# A 交付说明：状态通道、命令 ACK 与控制网关接线

更新：2026-11-18。开发源码来自 `D:\30817\Adrone\AirsimAlgorithm-feature-wang`。
Git 交付目录为同级 `AirsimAlgorithm-git`，基于远端 main，提交分支为 `feature/a-swarm-integration`。
原开发目录中的 `tmp/a_wiring_backup/` 备份保留在本地，不进入 Git 提交。

## 已完成的接线

A 的服务与网关已接入启动器和现有自动飞行循环。后续已在本机 Blocks 开展真实双机/五机试验，
结果与复现方式见 [Blocks 仿真验收](A_Blocks_仿真验收.md)。每次启动仍强制实时坐标验证及初始完整快照检查，失败不启动飞行。

| 内容 | 文件 |
| --- | --- |
| A 权威状态/命令服务入口 | `scripts/swarm_state_service.py` |
| 服务、线格式、订阅与重连 | `coordination/state_service.py`、`wire.py`、`client.py`、`controller_link.py` |
| 共享服务配置 | `coordination/runtime_config.py`、`configs/swarm.yaml` |
| 身份、任务/车辆、去重、序号、双时钟期限和 ACK | `coordination/command_ingress.py`、`state_service.py` |
| 编队目标接入与最终派发检查 | `control/swarm_flight_bridge.py`、`flight_modes/automatic_mode.py` |
| 统一飞行 API 边界 | `VehicleCommandGateway.bind_raw_client()` / `GatewayFlightClient` |
| B 网络运行入口 | `scripts/formation_service.py` |
| 五机示例 | `configs/swarm_5_drones.yaml`、`configs/formation_5_drones.yaml` |
| 真实只读验收证据 | `scripts/validate_swarm_state.py --output-json ...` |

本地原先没有 `formation/`，已从用户指定 GitHub 仓库 main 引入 B 的纯算法、配置及原测试，未改槽位算法。
`formation/controller.py` 的来源 SHA：`957b989e0118b8539f0044fcf3da4ca3dd881e1d`。

## 本机运行命令

本机 `python` 指向不可用的 WindowsApps 别名，因此以下使用已安装解释器。在项目根目录 PowerShell 执行：

```powershell
$pythonExe = 'C:\Users\Liuziyi\AppData\Local\Python\pythoncore-3.14-64\python.exe'
$env:AIRSIM_PYTHONCLIENT_PATH = 'D:\30817\Adrone\airsim\AirSim-main\PythonClient'
$settingsJson = "$env:USERPROFILE\Documents\AirSim\settings.json"
```

先确认目标场景和 settings 的车辆/LiDAR 名称、生成位置与配置标记一致。
两机示例需 Drone1/2、LidarSensor1/2、PlayerStart1/2、MissionEnd1/2；五机示例需相应 1～5 号名称。
现有自动飞行预检查仍读取各机初始任务标记，随后跟随机目标才由 B 的移动槽位接管。
示例不会修改真实 settings；请按场景调整 YAML。

### 只读两/五机验收

```powershell
& $pythonExe scripts/validate_swarm_state.py --settings-json $settingsJson --swarm-config configs/swarm.yaml --samples 10 --output-json runs/acceptance_2.json
& $pythonExe scripts/validate_swarm_state.py --settings-json $settingsJson --swarm-config configs/swarm_5_drones.yaml --samples 10 --output-json runs/acceptance_5.json
```

任意采样不完整或坐标验证失败，返回码为 2。JSON 记录位置/速度误差、姿态检查、各机读取耗时、
整轮耗时、接收/仿真时差与失效原因。不得用离线数据代替实测记录。

### 一次启动 A、各机控制进程与 B

确认仿真场景具备运行条件后：

```powershell
# 两机
& $pythonExe scripts/launch_swarm.py --settings-json $settingsJson --with-formation --confirm-simulation-clearance
# 五机
& $pythonExe scripts/launch_swarm.py --config configs/swarm_5_drones.yaml --formation-config configs/formation_5_drones.yaml --settings-json $settingsJson --with-formation --confirm-simulation-clearance
```

启动器等 A 坐标验证、完整快照和端口绑定成功后才启动飞行进程。
日志在 `runs/swarm_*/state_service.log`、`formation.log` 及各机日志。
正常退出先清理飞行进程，再停 A/B；Windows CTRL_BREAK 已接入 Python finally 清理。

### 独立启动与 C 接入

```powershell
# A：只读状态，不会起飞
& $pythonExe scripts/swarm_state_service.py --settings-json $settingsJson --swarm-config configs/swarm.yaml
# B：在另一个终端启动
& $pythonExe scripts/formation_service.py --swarm-config configs/swarm.yaml --formation-config configs/formation.yaml
```

只有 A 而无目标控制订阅者时，B 得到 `controller_unavailable`，不会假报可飞。
手动分进程部署须让 A 与 `flight_mode.py --swarm-config ...` 使用同一配置，勿重复占用服务端口。
C 的连接示例：

```text
ws://127.0.0.1:8765/ws/swarm?protocol_version=1&role=c_telemetry&producer_id=c-telemetry&stream_id=<每次进程启动的新UUID>
```

可复用 `coordination.client.SwarmConnection` 自动响应 ping。
C 订阅 `swarm.snapshot` 后自行转换为前端 `TelemetryFrame`；本次未修改 C 前端或把演示数据当作真实遥测。

## 控制与 ACK 语义

1. A 统一采集并验证 World NED；A/B/C 使用同一采集 ID，重复采集 ID 不续新鲜度。
2. 入口检查连接身份、车辆/跟随机、任务、控制者在线、lifecycle、重复 ID、旧序号和双时钟期限。
3. `command.ack accepted=true` 表示接收待执行，不表示飞行成功；最终可能被安全/恢复抢占。
4. 控制进程用 `command.request` 请求 A 的 `command.permit`，从剩余 TTL 扣除整个本机 RTT，
   再在 RPC 前重验期限并限制运动 duration。转发不会重启 A 的原始接收端 TTL。
5. 位置槽位和恢复导航目标进入现有轨迹/APF；恢复速度约束速度大小，保留 A 选定的静态绕障方向，再经过 ORCA。
6. 最终速度检查完整/新鲜快照、控制权、限速、CPA 与三维 LiDAR 间距，再提交到原始 AirSim API。
   实际采用、避障、冲突解除、恢复和紧急失败通过 `coordination.status` 返回。

常见原因：`envelope_identity_mismatch`、`unknown_vehicle`、`vehicle_not_formation_follower`、
`task_id_mismatch`、`controller_unavailable`、`no_authoritative_snapshot`、`missing_lifecycle`、
`stream_id_mismatch`、`envelope_sequence_mismatch`、`duplicate_command_id`、`stale_sequence`、`expired`。

断开 B 或控制连接会撤销相关命令；重连保留去重历史且不回放旧速度；仿真暂停时单调 TTL 仍会到期。
无法验证安全动作时进入 `EMERGENCY` / `MANUAL_INTERVENTION_REQUIRED`，不盲目悬停/降落，需要在仿真中人工接管。

## 验证和边界

`tests/test_swarm_wiring.py` 覆盖两/五机 B 规划→A ACK→控制许可→网关、真实回环 WebSocket、
后台订阅、断连、重放、旧快照、任务/车辆/权限、双时钟过期、RTT 扣减、租约、障碍和退出顺序。

接线初版全量离线回归：**1356 passed，248.15 秒**（Windows，Python 3.14，2026-11-18）。
后续 Blocks 实测及新增回归的结果见 [Blocks 仿真验收](A_Blocks_仿真验收.md)。
涉及本机网络和 Windows 多进程管道的测试在沙箱外执行；不连接或控制真实 AirSim 无人机。

```powershell
& $pythonExe -m pytest tests/ -q --basetemp=tmp/a_wiring_full_final
```

已附双机成功和五机失败的真实仿真记录；完整五机、队形切换、交叉避障、空中掉线与进程故障仍需逐项通过，不能以协议测试代替。
当前位置编队保持配置巡航高度，超出该高度 ±0.3 m 的槽位拒绝，不承诺任意三维槽位。
CPA / LiDAR 策略较保守，可能需在目标场景调参。
网关覆盖启用状态服务的自动飞行及其起降，旧手动/独立调试脚本尚未全部迁移。
`state_service.enabled:false` 保留旧独立导航实验，不支持 B 编队验收。
