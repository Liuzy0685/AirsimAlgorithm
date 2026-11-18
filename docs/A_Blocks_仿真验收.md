# A：Blocks 实测与复现

2026-11-18，主场景为用户现有 `Blocks/Content/maze.umap`（19 个不规则分布的柱体）。
本轮修改不保存地图资产，不覆盖用户 Documents/AirSim/settings.json。
验收代码分支：`feature/a-swarm-acceptance`，接续 `feature/a-swarm-integration`。

## 已实测的范围

实测结果见仓库 `artifacts/acceptance/2026-11-18/`。原始逐帧状态、A/B 日志、各机控制日志及失败试验保存在本地
`runs/acceptance/`，不将运行缓存整包上传。导出摘要记录原始文件 SHA256，轨迹 CSV 保留完整采样点。

最终代码双机复测 `maze_2_final` 通过：659/659 完整采样，领航 mission_complete，跟随机按全队完成信号
coordinated_stop，两机均确认落地/解锁/释放控制权。最小采样间距 2.09750 m，插值最小间距 2.09746 m，
零障碍碰撞；领航巡航约 100.20 s。全程编队 RMSE 4.12 m、最大 8.07 m，集结容差内采样 27 帧。
A 状态采集平均 1.13 ms、P95 1.57 ms。本轮记录了实际 Python 源码、运行配置和地图资产的 SHA256。

落地后独立运行的真实 A 服务协议检查 18/18 通过，包括有效 ACK/permit、重复 ID、旧序号、未知车辆、
错误任务、仿真期限、运行中的本地 TTL、B 断连撤销、重连不清空重放历史及 C 无法注入命令。
暂停同时会使未前进的状态时间戳失效，不能仅凭暂停期间拒绝命令来断言唯一原因是 TTL；因此另行验证了
模拟器正常运行且仿真期限未到时，本地 TTL 仍使 permit 失效。协议测试未执行任何飞行派发。
首次协议测试因没有跳过恢复后的短暂无效快照而提前退出，已保留为未完成尝试，修正等待条件后重新实测通过。

双机开发试验 `maze_2_run05` 完成领航任务、跟随机重新集结及两机落地/解锁/释放控制权：
703 帧完整快照，采样最小间距 2.15677 m，分段线性插值最小间距 2.15673 m，配置阈值 2 m，
未观测到障碍碰撞。领航巡航约 111.71 s；A 状态采集平均 1.06 ms、P95 1.51 ms。
避障期间槽位误差 RMSE 4.13 m、最大 7.12 m，因此本结果证明该任务能够避障后重新集结，
不代表全程保持 0.25 m 编队精度。该开发试验早于源码指纹记录功能；最终复测另列，不倒填源码哈希。

五张地图的双机只读状态检查各 25/25 完整采样通过。五机相同出生点配置在 `maze` 和 `regularMaze`
通过，在三张走廊地图失败：部分出生点不在有效地面上，验证器已阻止将其作为可飞配置。
这三张地图需按各自地面重新设置五机出生点。静止状态的位置变换误差约 0.1465 m（容差 0.5 m）；
零速变换检查不能证明所有动态姿态和非零航向场景均已验证。

五机密集横队尚未通过：最近一次 `maze_5_run05` 在约 55.88 s 因 Drone4 的
`emergency_distance` 退出，最小采样机间距 2.92429 m，未观测到障碍碰撞；
其余四机确认落地、解锁并释放控制权，Drone4 明确标为需人工处置。
此前 `maze_5_run04` 曾实际碰撞 Cylinder3，`maze_5_run03` 为诊断主动停止而非任务成功。
这些试验不能用于声称五机任务完成或五机避障算法已通过验收。
剩余瓶颈是密集柱体附近跟随机的导航/脱困与编队恢复，后续应在本日志的失败点验证修复，
并与 B 共同检查槽位是否落入障碍、是否需要收缩/切换队形及领航等待策略。

## 实测驱动的修复

- 旧 Blocks RPC 未发送 `ready`/`can_arm`，客户端类默认 False 不能当作服务端明确拒绝。
  `readiness_available` 明确区分“未报告”和“明确 false”；仅验收配置显式启用 `allow_legacy_readiness`，
  A 与 B 都需启用。保留真实 ready=false，不伪造 can_arm；正常配置默认拒绝未知 readiness。
- 起飞前识别有时间戳、向上法线和稳定速度的地面接触。旧服务端 landAsync 后仍可能报告 Flying，
  只有新接触事件、已验证的起始地面高度与法线、持续稳定等条件同时满足，才允许地面确认回退。
- 普通限速不再误报为 AVOIDING。冲突清除需连续安全帧和持续时间；有意等待编队不触发卡死恢复。
- B 的恢复速度先变成 A 局部规划器的前视目标，不能在规划之后覆盖速度而直接穿柱。
  新 B 额外发送可选 `recovery_target_ned_m`，让 A 直接规划到真实槽位；旧速度命令才使用方向前视。
  A/B 需同步更新，旧 A 的严格解析器会拒绝这个新字段。
  单帧可持有仍有效的旧 permit，新命令到来不会使其提前失效；原期限绝不延长。
- 若 permit 恰在派发前过期，不发送旧目标；另提议 A 的短时高度保持，仍逐项经过状态、CPA、
  控制权、LiDAR 检查。保持也不安全时拒绝派发并记录需要人工处置，不能假报清理完成。
- 最终 LiDAR 检查按传感器实测 pose 转为共同 NED 后对齐车辆，包含安装偏移与旋转。
  使用原始回波，避免导航自机排除框把近处障碍一起删掉。旧开发五机试验曾发生 `Cylinder3` 碰撞，
  该失败完整保留，不能用后续修复抹去。Blocks 的原始射线实现已忽略自身 Actor，验收专用感知配置
  因此关闭额外的自机排除框；其他传感器配置必须独立验证。
  静态点云检查覆盖观测延迟内的实测运动和实际有限时长指令；3 s 仍用于机间 CPA，
  不再把曲线路径当前切线强行外推 3 s。该分段恒速模型没有给出硬件制动距离保证。
- 全队使用停止文件协同退出，A/B 保持在线直至各机清理结束；领航完成后等待跟随机稳定，
  再统一结束。结果 JSON 的 `cleanup` 是最终清理结果；最新版本也在清理后更新顶层
  landing_confirmed/disarmed/api_control_released，landing_attempted 仅在实际发出 land RPC 后为 true。
  早期开发试验的顶层字段来自清理前，landing_attempted 则曾表示清理函数已调用；判断这些历史记录须看 cleanup.phase。

## 复现

在仓库根目录执行。`configs/acceptance/` 是独立的低速、有限时长仿真配置，
使用 settings 平移和显式局部任务目标，不依赖地图中不存在的 MissionEndN 标记。

```powershell
$py = 'C:\Users\Liuziyi\AppData\Local\Python\pythoncore-3.14-64\python.exe'
$env:AIRSIM_PYTHONCLIENT_PATH = 'D:\30817\Adrone\airsim\AirSim-main\PythonClient'
$project = 'D:\30817\Adrone\blocks\Blocks.uproject'
$settings = (Resolve-Path configs/acceptance/settings_2.json).Path
Start-Process 'D:\30817\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe' -WindowStyle Hidden -ArgumentList @(
  "`"$project`"", '/Game/maze', '-game', "-settings=`"$settings`"", '-windowed', '-ResX=960', '-ResY=540', '-NoSound', '-unattended')

# 模拟器加载完成、出生点稳定后，唯一 output 目录必须尚不存在。
& $py scripts/run_swarm_acceptance.py --blocks-project $project --settings-json configs/acceptance/settings_2.json --swarm-config configs/acceptance/swarm_2.yaml --formation-config configs/acceptance/formation_2.yaml --map-label maze --duration 240 --output runs/acceptance/maze_2_repeat --confirm-simulation-clearance

# 所有飞机已落地且 API 控制权已释放时，才允许切换地图做只读状态矩阵。
& $py scripts/validate_blocks_maps.py --blocks-project $project --settings-json configs/acceptance/settings_2.json --swarm-config configs/acceptance/swarm_2.yaml --output runs/acceptance/state_maps_2_repeat --samples 25

# 不起飞的真实协议异常检查：独立启动 A，测试 ACK、重放、断连和暂停中的本地 TTL。
& $py scripts/validate_live_swarm_protocol.py --settings-json configs/acceptance/settings_2.json --swarm-config configs/acceptance/swarm_2.yaml --output runs/acceptance/protocol_repeat

# 离线从真实运行导出证据与图；绘图额外依赖 matplotlib。
& $py scripts/export_swarm_acceptance.py --run runs/acceptance/maze_2_repeat --output artifacts/acceptance/maze_2_repeat
```

五机需先退出自己的双机仿真实例，用 `settings_5.json` 重启；将 swarm/formation 文件改为 `_5.yaml`，
观测上限改为 360 s。不能只改 YAML 而让运行中的模拟器仍保留两机，脚本会拒绝。
若故障结果明确显示 MANUAL_INTERVENTION_REQUIRED，先确认全部控制进程已退出，再恢复该测试实例；
不能将重置模拟器记作成功落地。

## 判定与局限

代码验证：全量 **1390 passed，253.54 s**（Windows/Python 3.14）。随后增加启动器拒绝复用旧运行目录的检查，
定向回归 **48 passed，0.54 s**。该后续变更只阻止已有日志目录启动，不改变本轮飞行控制算法。
当前共收集 1391 项测试；没有将 1391 写成同一轮全量执行数。

飞行通过要求：启动器正常退出、有效间距观测、零障碍碰撞、实际集结采样、领航 mission_complete、
所有飞机最终确认落地并释放控制权。单纯跑满时间、收到 ACK、控制进程启动或没有碰撞均不足以通过。
采样及线性插值不能给出连续时间安全证明；低速统计包含主动等待和地面时间。
离线故障回归与落地状态的协议故障检查，也不能代替空中断连、迎面/交叉/追越和动态障碍验收。
