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

## License

MIT（与 AirSim PythonClient 保持一致）。
