# UAV AirSim 自主避障与路径规划（交接版）

基于 **Microsoft AirSim**（Unreal Engine 4.27 多旋翼仿真）的 Python 自主避障管线。
系统读取 LiDAR 点云、无人机状态与碰撞信息，输出安全的速度指令，完成「感知 → 规划 → 控制」闭环。

本仓库为**最新版本源码包**（交接用），已剔除历史目录、基准测试与日志产物，可直接克隆后在本地继续开发。

---

## 目录结构

```
new/
├── adapters/           # AirSim RPC 封装
│   └── airsim_client.py        # 惰性连接 / 重连的 AirSim 客户端
├── models/             # 数据模型（dataclass）
│   ├── lidar_frame.py          # LiDAR 帧数据（SensorLocalFrame）
│   ├── vehicle_state.py        # NED 机体状态
│   ├── collision_state.py      # 碰撞信息
│   ├── sector_measurement.py   # 单扇区距离 + FOV 可观测性
│   ├── directional_distances.py# 帧级结果（含 legacy 映射）
│   ├── local_planner_command.py# 局部规划指令
│   ├── fixture_result.py       # 测试夹具结果
│   └── lidar_frame.py
├── perception/         # 感知：点云 → 16 扇区
│   ├── perception_config.py    # 严格 YAML 配置加载（SectorDef）
│   ├── pointcloud_filter.py    # NaN/量程/自剔除/体素过滤
│   ├── pointcloud_to_sectors.py# 点云 → 16 扇区
│   └── sensor_fov.py           # LiDAR FOV 装载 + 扇区覆盖校验
├── sensors/            # 传感器读取器
│   ├── lidar_reader.py         # getLidarData() → LidarFrame
│   ├── state_reader.py         # getMultirotorState() → VehicleState
│   └── collision_reader.py     # simGetCollisionInfo() → CollisionState
├── mapping/            # 建图
│   ├── occupancy_grid.py       # 占用栅格
│   └── distance_field.py       # 距离场
├── transforms/         # 坐标变换
│   └── lidar_to_local_ned.py   # LiDAR 局部坐标系 → NED
├── planners/           # ★ 核心规划算法
│   ├── cbmba_astar.py          # CBMBA A* 全局路径规划（从旧 JS 移植）
│   ├── cbmba_guidance.py       # CBMBA 引导层
│   ├── improved_potential_field.py  # APF 人工势场反应式避障
│   ├── local_trajectory_planner.py  # 局部轨迹规划（确定性轨迹生成）
│   ├── trajectory_tracker.py   # 轨迹跟踪
│   ├── local_recovery.py       # 卡死 / 震荡检测（纯计算）
│   ├── recovery_commander.py   # 恢复指令 + 状态机
│   ├── goal_termination.py     # 目标终止判定
│   └── process_workers.py      # 多进程 worker（CBMBA/轨迹/建图）
├── control/            # 控制
│   ├── velocity_controller.py  # 安全速度发送（默认只读）
│   └── safety_supervisor.py    # 安全监督
├── flight_modes/       # 飞行模式（编排层）
│   ├── automatic_mode.py       # 自主避障主循环（编排上述全部模块）
│   ├── shared_flight_session.py# AirSim 生命周期（连接/解锁/起飞/降落）
│   ├── manual_mode.py          # 键盘手动
│   ├── manual_gamepad_mode.py   # 手柄手动（XInput）
│   ├── gamepad_config.py       # 手柄配置（严格校验）
│   ├── gamepad_reader.py       # 手柄读取（pygame，可无则降级）
│   ├── gamepad_state.py        # 手柄状态
│   ├── airsim_debug_draw.py    # AirSim 调试绘制
│   └── trajectory_flight_metrics.py  # 轨迹飞行指标
├── utils/              # 工具
│   └── consecutive_tracker.py  # 连续无效帧跟踪器
├── configs/            # 运行时配置（YAML / JSON）
│   ├── vehicle.yaml            # 车辆参数
│   ├── perception.yaml         # 16 扇区定义
│   ├── trajectory_planner.yaml # 轨迹规划参数
│   ├── trajectory_flight.yaml  # 轨迹飞行参数
│   ├── local_planner.yaml      # 局部规划参数
│   ├── manual_gamepad.yaml     # 手柄参数
│   ├── minimal_flight.yaml     # 最小飞行配置
│   ├── mission_recovery_test.yaml
│   ├── runtime_config.py       # 运行时配置加载
│   ├── airborne_fixture_config.py
│   └── settings.example.json   # AirSim settings 示例（勿覆盖真实文件）
├── scripts/            # 入口与只读调试脚本
│   ├── flight_mode.py          # ★ 正式实飞入口
│   ├── sensor_smoke_test.py    # 只读传感器采集测试
│   ├── sector_smoke_test.py    # 只读扇区冒烟测试（FOV 门控）
│   ├── lidar_axis_calibration.py# LiDAR 轴向标定
│   ├── minimal_lidar_flight.py # 最小飞行
│   ├── gamepad_debug.py        # 手柄调试
│   ├── gamepad_axes_test.py    # 手柄轴测试
│   ├── plot_flight_trace.py    # 航迹绘图
│   └── airborne_test_fixture.py# 空中测试夹具
├── tests/              # 单元测试（1000+ 个，自包含，无需 AirSim）
│   └── fixtures/       # 自包含测试 settings JSON
├── requirements.txt
└── .gitignore
```

---

## 依赖安装

```bash
pip install -r requirements.txt
```

> `airsim` 包**不通过 pip 安装**，而是从 AirSim 源码树通过 `PYTHONPATH` 加载（见下文）。

机间相遇、异常退出和运行时队形切换的最新完善见 [A 机间避让专项验收](docs/A_机间避让专项验收_2026-11-19.md)，
包括与最终 CPA 相容的速度候选、独立航线测试模式、队形物理收敛字段及可复现的串行仿真套件。
> `pygame` 为可选项，缺失时手柄读取自动降级为「已断开」，不影响运行。

---

## 运行单元测试（无需 UE4 / AirSim）

```bash
# 在仓库根目录执行（不要用裸 pytest，避免历史重名测试冲突）
python -m pytest tests/ -q
```

全部测试自包含，不依赖 AirSim、UE4、参考文件或真实 settings.json。

---

## 实飞入口

正式实飞入口为 `scripts/flight_mode.py`：

### 多无人机独立导航（第一阶段）

当前 `configs/swarm.yaml` 默认启用 A 权威状态服务；启动器先启动并验证 A，再启动各机控制进程。加入 `--with-formation` 可同时启动 B 的编队服务：

```powershell
python scripts/launch_swarm.py --settings-json "C:\path\to\settings.json" --with-formation --confirm-simulation-clearance
```

两机/五机配置、独立启动、ACK 与异常处理见 [A 状态服务与控制网关交付说明](docs/A_交付说明_状态服务与命令入口.md)。Blocks 不规则柱体地图已开展真实双机/五机测试，逐项结果、失败记录及复现命令见 [A Blocks 仿真验收](docs/A_Blocks_仿真验收.md)。将 `state_service.enabled` 设为 `false` 可运行下述旧独立导航实验；该模式不消费 B 编队命令。

基于 B 的 `f5b48b9`，新增可选的障碍反馈、承诺路线、槽位联合调整、领航历史路径跟随与原队形恢复，
使用独立的 `swarm_2_navigation.yaml` / `swarm_5_navigation.yaml`，保留原安全门控与控制网关。
协议、复现参数和本轮实测结果见 [A/B 障碍感知导航与恢复](docs/A_B_障碍感知导航与恢复.md)。

前端任务入口由独立 `scripts/operator_api.py` 提供，支持真实地图元数据、任务校验、初始队形设置、
启动、请求停止及执行状态查询；配置和示例见 [A 前端任务接口](docs/A_前端任务接口.md)。
飞行中队形切换需 B 接入 [运行时编队接口](docs/B_运行时编队接口接线清单.md)，未接入时明确拒绝。

B 提供的 `maze_2_mytest_03` 感知时效失败分析、A 原始扫描缓存/控制减负修复，以及不飞行的
单/双 UE 地面对照方法见 [A 感知时效与负载排查](docs/A_感知时效与负载排查_2026-11-19.md)。
新网格导航默认关闭旧诊断 worker；`navigation.run_legacy_diagnostics: true` 可恢复对照负载。

启用状态服务而未启动 B 时，跟随机等待有效编队命令；领航机继续自己的导航任务。需要完整编队时使用上述 `--with-formation`，或单独运行 B 服务。

`scripts/launch_swarm.py` 按 `configs/swarm.yaml` 为每架无人机启动独立的自动导航进程。双机默认将 Drone1 绑定到 `PlayerStart1`/`MissionEnd1`，Drone2 绑定到 `PlayerStart2`/`MissionEnd2`。建议在 UE 中放置空 Actor，并分别设置唯一的 Actor Tag 为这些名称；代码按 Actor Tag 或 Actor 内部名查询标记。程序读取 `PlayerStartN` 的场景位置作为各机共同坐标原点，并用它将对应的 `MissionEndN` 换算到本机导航坐标。AirSim 实际生成位置仍由 `settings.json` 的车辆 `X/Y/Z` 决定，因此请按场景中各 `PlayerStartN` 的位置设置这些数值，使生成位置与标记重合；程序会记录两者不一致的告警，但不会自动移动无人机。请确认 `settings.json` 已定义对应无人机和 LiDAR。默认以水平 2D ORCA 过滤本机局部规划器的期望速度，`priority_hold` 模式则按恒速 CPA 预测和优先级规则让行。`priority_hold` 模式下，优先级数值越高越有优先通行权；同优先级时按无人机名称确定唯一让行机。ORCA 模式中双方承担互惠避让责任，优先级不会豁免安全过滤。该实现用于仿真阶段的算法对比，不是经验证的飞行安全系统。

```powershell
python scripts/launch_swarm.py `
  --settings-json "C:\path\to\settings.json" `
  --confirm-simulation-clearance
```

也可以单独指定无人机及标记：`flight_mode.py --vehicle-name Drone2 --lidar-name LidarSensor2 --mission-start-actor PlayerStart2 --mission-end-actor MissionEnd2`。控制锁按无人机隔离，同一架无人机仍只能由一个 flight mode 控制。`configs/swarm.yaml` 默认启用水平 2D ORCA；将 `coordination.algorithm` 改为 `priority_hold` 可运行原有 CPA 加优先级悬停基线。首次验证时，请在空旷区域、低速和足够大的起始间距下运行，确认两架无人机都能读取对方状态；观察 `cbmba_mission_goal`、`inter_uav_prediction`、`inter_uav_orca_command` 和 `inter_uav_orca_fallback` 日志，再逐步测试交叉航线。ORCA 只处理无人机之间的水平速度冲突，LiDAR 仍处理静态障碍，垂直避障仍由原高度控制处理。当前实现没有建模机体加速度、旋翼下洗或通信延迟的不确定性；ORCA 无可行速度、对端状态过期或 ORCA 结果触及 LiDAR 硬安全间距时，所有优先级都会请求水平停车，并保留高度保持或必要的垂直逃逸。`passing_bias_mps` 默认 0.1 m/s，仅在预测冲突时调整期望速度以缓解对称相遇停滞；设为 0 可关闭。算法选型、修复内容和验证边界见 [多机避障选型说明](docs/multi_uav_avoidance_selection.md)。

```bash
# 手动（键盘）
python scripts/flight_mode.py --mode manual

# 手动（手柄）
python scripts/flight_mode.py --mode manual --manual-input gamepad

# 自主避障（auto 模式强制要求显式 --settings-json）
python scripts/flight_mode.py --mode auto --settings-json <你的 settings.json>
```

> 不要用 `python -m flight_modes.automatic_mode` 作为实飞入口。

### 加载 AirSim PythonClient

```bash
# 方式一（推荐）：PYTHONPATH
export PYTHONPATH="<AirSim源码>/PythonClient:$PYTHONPATH"

# 方式二：环境变量
export AIRSIM_PYTHONCLIENT_PATH="<AirSim源码>/PythonClient"
```

---

## 坐标系

| 坐标系 | 轴定义 | 用途 |
|--------|--------|------|
| 世界 NED | +X=北/前, +Y=东/右, +Z=下 | `VehicleState`、`send_velocity_world_ned()` |
| LiDAR SensorLocalFrame | +X=前, +Y=右, +Z=下 | `LidarFrame.point_cloud_sensor` |
| 机体 FRD | vx=前, vy=右, vz=下 | `send_velocity_body_frd()` |

航向=0 时，机体轴与世界 NED 轴对齐。

---

## 核心算法一览

- **CBMBA A\***：全局路径规划，从旧 JS 项目纯 Python 移植（`planners/cbmba_astar.py`）。
- **APF（人工势场）**：反应式避障（`planners/improved_potential_field.py`）。
- **局部轨迹规划**：确定性轨迹生成 + 跟踪（`local_trajectory_planner.py` / `trajectory_tracker.py`）。
- **恢复机制**：卡死（XY 位移 < 0.15m 持续 ≥2.5s）与震荡（vy 符号翻转）检测 → 后退/侧移脱困（`local_recovery.py` / `recovery_commander.py`）。
- **目标终止**：`goal_termination.py`。
- **多进程调度**：CBMBA/轨迹/建图以多进程 worker 运行，感知以线程运行（`process_workers.py`）。

---

## 交接说明（迁移范围）

本包只包含**最新版本**的导航系统源码，以下内容已被排除（如需要可向原作者索取）：

- 历史 `planning/` 目录（已被 `planners/` 取代）及其依赖测试
- `benchmark_*.py` 基准脚本及对应的 `test_*_benchmark.py` / `*_profile.py` 性能测试
- 日志、航迹 CSV、`.git`、`__pycache__`、`.pytest_cache`
- 开发过程报告（`*_REPORT.md`、`*_AUDIT.md` 等）

---

## 当前分支运行说明

单机导航基线源于 `feature-wang`，在原有轨迹规划和避障逻辑的基础上补充了终点相关控制：无人机接近 `MissionEnd` 时使用实时终点位置，避免继续沿缓存蓝线飞过终点；进入捕获范围后减速并等待稳定，再结束任务。巡航速度仍由现有配置控制，没有为了终点逻辑降低全程速度。

### Windows 启动步骤

1. 先启动 Unreal 地图并点击 Play，确认 AirSim 已监听 `127.0.0.1:41451`。
2. 打开 PowerShell，进入仓库并激活虚拟环境：

```powershell
cd "C:\Users\13520\Desktop\new\AirsimAlgorithm"
.\.venv\Scripts\Activate.ps1
$env:AIRSIM_PYTHONCLIENT_PATH = "D:\AirSimDev\AirSim\PythonClient"
$env:PYTHONPATH = "$env:AIRSIM_PYTHONCLIENT_PATH;$env:PYTHONPATH"
```

3. 确认 `settings.json` 中的车辆名、LiDAR 名称与实际地图一致，然后运行自动轨迹模式：

```powershell
python -u scripts\flight_mode.py --mode auto --confirm-simulation-clearance --settings-json "C:\Users\13520\Documents\AirSim\settings.json" --local-navigation-mode trajectory --flight-config configs\trajectory_flight.yaml --max-duration 480 --quiet --log-file runs\flight.log
```

其中 `--max-duration 480` 是最长运行时间，正常情况下检测到 `MissionEnd` 后会提前结束。终点控制参数位于 `configs/trajectory_planner.yaml` 的 `trajectory_tracking` 和 `goal_termination` 配置段。

### 连接测试

如果程序提示 `WSAECONNREFUSED`，说明 Unreal/AirSim 尚未启动或端口不一致。可先执行只读 LiDAR 测试：

```powershell
python -u scripts\sector_smoke_test.py --settings-json "C:\Users\13520\Documents\AirSim\settings.json" --frames 3 --interval 0.2
```

看到 `Connected!` 和 `FOV FULLY COMPATIBLE` 后，再运行自动飞行命令。详细日志保存在 `runs\flight.log`，该目录已被 `.gitignore` 忽略，不会上传到仓库。

## License

MIT（与 AirSim PythonClient 保持一致）。

## 2026-11-22 前端联调更新

统一双机/五机与 maze/FlyingExampleMap 启动方案、Ground 地面预检及失败任务恢复降落，
见 [联调与恢复说明](docs/前端联调与恢复_2026-11-22.md)。`recovery/check` 仍只读，
显式恢复降落由 A 的独立受控进程执行；浏览器不直接调用 AirSim 飞控。
