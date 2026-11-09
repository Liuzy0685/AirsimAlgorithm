# 多无人机协作接口与分工

本文件是 A（协同避障）、B（编队）、C（前端遥测）共同使用的接口约定。当前仓库中的公共代码覆盖阶段一、二；跨进程状态发布、编队命令接入、真实遥测发布和统一控制网关尚未接线。任何一方不得把尚未接线的模型当作已运行的服务。

跨进程消息、命令生命周期、控制权、字段语义、重连、配置和测试责任的详细约定见 [`docs/多机通信与控制协议.md`](docs/多机通信与控制协议.md)。协议版本由 `common/constants.py::PROTOCOL_VERSION` 定义；当前为 v1 草案。

## 公共约定

| 内容 | 唯一来源 | 三方使用方式 |
| --- | --- | --- |
| 坐标与常量 | `common/constants.py` | 位置、线速度、目标和避障速度统一为 World NED，单位米、米/秒；角度为弧度。角速度字段明确为 body frame。 |
| 无人机状态 | `models/vehicle_state.py`，由 `common/models.py` 原类 re-export | 使用同一个 `VehicleState` 类；AirSim 原始 `timestamp` 为纳秒，`received_monotonic_seconds` 为本机单调时钟秒。 |
| 编队命令 | `common/models.py::FormationCommand`、`CommandLifecycle` | B 输出目标位置或期望速度（World NED），A 消费；跨进程发送时须携带命令 ID、序号、有效期、来源和任务 ID。 |
| 避障命令 | `common/models.py::AvoidanceCommand` | A 输出期望速度与来源；当前尚未接入最终速度仲裁。 |
| 遥测消息 | `common/models.py::TelemetryMessage` | A/C 约定的数据结构；C 只读。当前没有发布/订阅通道。 |
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

### C：前端与遥测（`frontend/` 及遥测服务）

1. 从 `common.models` 导入 `TelemetryMessage`、`VehicleState`，从快照读取有效状态并显示各机采集时间、失效原因和状态完整性。
2. 建立真实遥测发布/订阅通道，替换前端演示数据；先明确进程间传输格式、版本号和消息频率，不在前端直接读 AirSim RPC。
3. 显示编队与避障命令、协调状态；验证 1～5 机、掉线、过期、时间差和重新连接场景。前端不得修改飞行控制状态。

## 集成顺序与边界

先完成真实 AirSim 坐标与性能验收，再按 v1 协议实现跨进程状态/命令传输，之后接入 B 的 `FormationCommand` 和 C 的 `TelemetryMessage`。当前只有 Python 内存中的共用数据类型和读取接口；三架或五架无人机各进程间尚无公共状态服务。现有 ORCA、APF、priority_hold 与飞行控制行为保持原样。

离线验收：`python -m pytest tests/test_common_models.py tests/test_vehicle_state_import_compat.py tests/test_coordinate_transform.py tests/test_state_provider.py tests/test_swarm_snapshot.py tests/test_coordinate_transform_integration.py -q`。公共模型、阶段一和阶段二在开发机通过 53 项测试；AirSim 实测仍待完成。
