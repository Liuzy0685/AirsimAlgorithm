# 多无人机协作接口与分工

本文件是 A（协同避障）、B（编队）、C（前端遥测）共同使用的接口约定。A 状态服务、B 命令 ACK、控制进程订阅、编队目标与控制网关已经接入 `launch_swarm.py` / `flight_mode.py` / `automatic_mode.py`。启用 `state_service.enabled` 时，状态服务必须在运行时通过真实坐标验证和完整快照检查，控制进程才会起飞。离线测试不替代两机、五机 AirSim 飞行验收。完整运行步骤见 [`docs/A_交付说明_状态服务与命令入口.md`](docs/A_交付说明_状态服务与命令入口.md)。

跨进程消息、命令生命周期、控制权、字段语义、重连、配置和测试责任的详细约定见 [`docs/多机通信与控制协议.md`](docs/多机通信与控制协议.md)。协议版本由 `common/constants.py::PROTOCOL_VERSION` 定义；当前为 v1 草案。

## 公共约定

| 内容 | 唯一来源 | 三方使用方式 |
| --- | --- | --- |
| 坐标与常量 | `common/constants.py` | 位置、线速度、目标和避障速度统一为 World NED，单位米、米/秒；角度为弧度。角速度字段明确为 body frame。 |
| 无人机状态 | `models/vehicle_state.py`，由 `common/models.py` 原类 re-export | 使用同一个 `VehicleState` 类；AirSim 原始 `timestamp` 为纳秒，`received_monotonic_seconds` 为本机单调时钟秒。 |
| 编队命令 | `common/models.py::FormationCommand`、`CommandLifecycle` | B 输出目标位置或期望速度（World NED），A 消费；跨进程发送时须携带命令 ID、序号、有效期、来源和任务 ID。 |
| 避障命令 | `common/models.py::AvoidanceCommand` | A 输出期望速度与来源；当前尚未接入最终速度仲裁。 |
| 遥测消息 | `common/models.py::TelemetryMessage` | C 从 A 的 WebSocket 订阅快照；转换到前端 TelemetryFrame 的服务仍由 C 负责。 |
| 状态快照 | `coordination/state_provider.py::SwarmStateSnapshot` | 包含 1～5 机各自采集时间、有效状态、错误、采集耗时和时差。算法需要全部邻机时必须检查 `collection_complete`。 |
| 坐标转换 | `coordination/coordinate_transform.py::CoordinateTransformer` | 先用实际 AirSim 位姿执行 `validate()`；验证失败禁止使用新转换结果。位置为 `R @ p + t`，速度为 `R @ v`。 |
| 状态读取 | `coordination/state_provider.py::StateProvider` | `get_snapshot()` 执行一轮顺序读取；其他查询只读最近一轮快照，不发起隐式 RPC。过期、失效或时间差过大的结果不可作为实时避碰输入。 |

## 三人的任务

### A：协同避障与公共状态（`common/`、`coordination/`、后续 `control/`）

1. 在目标 AirSim 场景运行 `scripts/validate_swarm_state.py`，记录各机位置误差、速度/姿态坐标约定及 2/5 机 RPC 耗时；确认 `simGetObjectPose(vehicle_name)` 在当前版本可靠。
2. 保留 `automatic_mode.py` 旧调用入口，先并行对照 `StateProvider` 与旧 `StateReader`、新旧坐标转换；验证后再逐步迁移 CPA、priority_hold、ORCA。
3. 建立 SafetyFilter、VelocityArbitrator 和唯一 AirSim 控制网关；对缺失、过期邻机状态采取安全处理。所有接口变更先更新本文件和对应测试。

### B：编队（自己的编队模块）

1. 从 `common.models` 导入 `VehicleState`、`FormationCommand`；只使用已验证、未过期的 World NED 状态计算编队槽位。
2. 输出 `FormationCommand`，明确 `vehicle_name`、`mode`、目标/速度、`formation_id` 和 `CommandLifecycle`；不得直接调用 AirSim 飞行控制 API。
3. 与 A 约定命令传递方式和失联/超时策略，并测试单机、双机、五机及避障优先级接管后的编队恢复。

B 的纯算法位于 `formation/controller.py`，配置在 `configs/formation.yaml`。`FormationController.plan()` 接受完整 `SwarmStateSnapshot`、B 本机记录的接收时间与协调状态，输出带生命周期的跟随机命令。支持纵队、横队和 V 字队形，槽位随领航机航向旋转；冲突/避障/紧急状态停止发布，冲突解除后限速回归，收敛后恢复位置目标。领航机继续由 A 的导航任务控制。

网络运行入口为 `scripts/formation_service.py`：订阅 A 快照并发布命令、读取 ACK；不直接调用 AirSim。B 的原有纯算法与测试保持不变，A 负责接收端期限、控制许可与最终安全检查。离线算法测试：`python -m pytest tests/test_formation_controller.py -q`。

B 的状态门控位于 `formation/runtime.py`：启动或重连后先收到全部配置车辆的近期 `coordination.status`，才从完整快照发布编队命令；状态过期、任务不匹配或紧急/避障状态时停止发布。`CONFLICT_CLEARED` 会保留到下一轮有效快照生成恢复命令，避免被后续 `NORMAL` 帧覆盖；新一轮冲突或状态超时会撤销旧的解除事件。对应离线测试为 `tests/test_formation_runtime.py`。

### C：前端与遥测（`frontend/` 及遥测服务）

1. 从 `common.models` 导入 `TelemetryMessage`、`VehicleState`，从快照读取有效状态并显示各机采集时间、失效原因和状态完整性。
2. 建立真实遥测发布/订阅通道，替换前端演示数据；先明确进程间传输格式、版本号和消息频率，不在前端直接读 AirSim RPC。
3. 显示编队与避障命令、协调状态；验证 1～5 机、掉线、过期、时间差和重新连接场景。前端不得修改飞行控制状态。

## 集成顺序与边界

状态服务独占权威多机采集；各机控制进程从它订阅邻机快照，本机高频传感器仍供本地控制使用。B 的位置槽位接入现有轨迹/APF 的目标输入，速度请求在既有安全/恢复优先级之后、ORCA 之前作为期望输入，最终速度再次经过网关的状态、CPA、三维 LiDAR 和期限检查。`command.ack` 表示入口接收；控制状态报告实际采用或失败。派发前通过 `command.request` / `command.permit` 取得 A 接收端剩余 TTL，并扣除控制进程本机测得的整个 RTT，禁止跨进程转发重置 TTL。`state_service.enabled:false` 保留独立导航实验，不能使用 B 编队。

离线验收：`python -m pytest tests/test_common_models.py tests/test_vehicle_state_import_compat.py tests/test_coordinate_transform.py tests/test_state_provider.py tests/test_swarm_snapshot.py tests/test_coordinate_transform_integration.py -q`。公共模型、阶段一和阶段二在开发机通过 53 项测试；AirSim 实测仍待完成。

## A 已交付模块（本分支新增，均带离线单测）

| 模块 | 作用 | 单测 |
| --- | --- | --- |
| `coordination/wire.py` | 协议封装/快照序列化/命令解析/ack/状态编码 | `tests/test_wire.py` |
| `coordination/command_ingress.py` | `formation.command` 身份/生命周期/去重/旧序号/超时 | `tests/test_command_ingress.py` |
| `coordination/state_publisher.py` | 权威 `swarm.snapshot` 发布；验证失败不发布 | `tests/test_state_publisher.py` |
| `coordination/ws.py` | 最小 RFC 6455 WebSocket 握手/帧编解码（stdlib） | `tests/test_ws.py` |
| `coordination/state_service.py` | `ws://127.0.0.1:8765/ws/swarm` 状态/命令服务、断连/重连 | `tests/test_state_service.py` |
| `control/command_authority.py` | 单机唯一控制者租约 | `tests/test_command_authority.py` |
| `control/safety_filter.py` | 编队目标安全检查（限速/机间间距/邻机完整） | `tests/test_safety_filter.py` |
| `control/velocity_arbitrator.py` | 速度源优先级仲裁 | `tests/test_velocity_arbitrator.py` |
| `control/vehicle_command_gateway.py` | 唯一飞行控制入口（权限+安全+派发） | `tests/test_vehicle_command_gateway.py` |

新增接线模块为 `coordination/runtime_config.py`、`coordination/client.py`、`coordination/controller_link.py`、`control/swarm_flight_bridge.py`；入口为 `scripts/swarm_state_service.py` 和 `scripts/formation_service.py`。`tests/test_swarm_wiring.py` 覆盖两/五机、真实回环 WebSocket、ACK、断连、重放、过期和最终 RPC 边界。统一网关作用于启用状态服务的自动飞行进程（含该进程起降）；其他旧手动/独立调试脚本仍须使用同一车辆 OS 锁，尚未全部迁移。详细说明与真实验收清单见 [`docs/A_交付说明_状态服务与命令入口.md`](docs/A_交付说明_状态服务与命令入口.md)。
