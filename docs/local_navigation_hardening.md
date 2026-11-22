# 局部避障六方面改进与对照验收

目标是提高未知环境中的可用性和可解释性，不承诺任何地图、负载和任务下都优于旧版本。本改动不引入深度学习，不使用预先已知的完整障碍布局，也没有关闭碰撞、原始点云、状态时效或控制权门控。

## 已实现与边界

| 方面 | 本轮实现 | 仍需验证或不作保证 |
| --- | --- | --- |
| 感知与控制时效 | 记录 RPC、感知处理、导航记忆更新、寻路、网关、实际速度 RPC、控制工作和派发间隔，以及派发时点云/状态年龄。摄像头统一串行排队、限制总采集速率，并随读取耗时退避 | 真实单/双/五机、0/1/全部摄像头的 RPC 与 UE 负载矩阵尚未执行；摄像头降频不等于能消除 UE 延迟 |
| 障碍记忆 | 静态观测不再仅因 TTL 消失；需要多个不同扫描中的实测清空射线确认。已知队友附近回波进入短期层，避免长期幽灵障碍；保留此前确认的墙体 | 这是使用现有定位的观测记忆，不是 SLAM 或语义识别。最多8000格，容量满后仍会淘汰旧格；二维与近水平射线存在表示局限 |
| 绕柱脱困 | 将实际位移、目标进展和运动意图区分开；等待 B、让行、领航机等队友及状态不可用时不累积卡死。停滞给失败路线施加有时限的软代价，重新选择通道 | 不提供全局可达性保证；有些迷宫需要更大的观测范围或高层探索。没有把失败通道伪装成实体障碍 |
| 运动平滑 | 普通导航速度按实测速度限制变化，叠加延迟/净空和转弯前目标距离的限速。平滑结果重新检查短段净空，并保留最终原始点云和机间检查 | 停止/安全动作优先，可以打破平滑约束；不是完整的加速度或 jerk 最优控制。速度限制可能延长任务时间 |
| 多机通行 | 窄通道内采用与现有让行一致的优先级和名称顺序，记录 CLEAR/ENTERING/OCCUPYING/YIELDING/EXITING/BLOCKED_TIMEOUT。等待超时仅申请沿已走过且重新检查的短段退让，不能通过则继续明确等待 | 这是局部通行建议，不是集中式资源预约、分布式共识或无死锁证明。交叉口、多机拥堵和没有退让空间的通道仍可能无法自行解除；CPA/网关继续作最终裁决 |
| 异常与展示 | 使用原有协调状态的 reason 字段说明阻挡/等待/过期等原因；前端补中文解释和相机预算帧率上限。提供窗口时延日志的离线汇总 | 没有把 ACK 当作实际到达；正常降落、清理确认与人工处理继续由现有任务/恢复结果决定 |

后退不是盲目倒飞：只使用长度有限的历史路径，先对现在的障碍和队友缓冲区重新检查，再通过原有 CPA 和原始点云网关。超时不强制让另一架飞机进入仍被占用的通道。静止的同向领航机不会仅因速度为零被误判成迎面来机；有新鲜导航目标时会参考其运动意图。

## 启用与回退

新的运动策略由 swarm YAML 顶层 `local_navigation_hardening` 控制。现有共享 acceptance 配置、settings 和编队 YAML 不修改，缺省为关闭。主开关打开后，`memory`、`recovery`、`smoothing`、`passage` 可独立关闭，便于逐项定位收益与退步。只有 `navigation.enabled` 的多机栅格导航分支接受这些策略；上一轮短轨迹候选改动仍由其独立配置控制。

以下命令只生成试验副本，不启动仿真、不发送飞行命令，也不覆盖原配置：

```powershell
# 带相同诊断开销的对照：四项运动/地图策略均关闭
python scripts/prepare_navigation_trial.py --base configs/acceptance/swarm_2_navigation.yaml --output runs/trial/base_2.yaml
# 逐项启用，例如先检查稳定记忆
python scripts/prepare_navigation_trial.py --base configs/acceptance/swarm_2_navigation.yaml --output runs/trial/memory_2.yaml --features memory
# 集成试验
python scripts/prepare_navigation_trial.py --base configs/acceptance/swarm_2_navigation.yaml --output runs/trial/all_2.yaml --features memory recovery smoothing passage
```

五机使用同样流程，将输入换为 `swarm_5_navigation.yaml`。生成副本保留所有原有飞行、安全、传感器和队形参数，记录原文件及副本 SHA256；B 仍使用相应原编队 YAML。实际验收时只替换原验收命令的 `--swarm-config`，保留对应的 `--settings-json`、`--formation-config` 和地图。对照开启相同日志，用于比较策略本身；它不是字节完全不变的旧发布版本性能测量。

### 相机资源预算

相机服务默认总采集上限10 FPS，并以 `capture_utilization=0.5` 按读取耗时退避。十个视角请求20 FPS时，不会承诺每路20 FPS。`/api/cameras` 每路增加 `requested_fps`、`effective_fps_limit`、`capture_ms`、`throttled`；前端显示“单路预算上限”，该值不是实际测得的 FPS。视频过期提示按预算调整，**飞行状态与 LiDAR 的过期阈值不变**。

```powershell
python scripts/camera_service.py --vehicles Drone1 Drone2 --max-total-fps 10 --capture-utilization 0.5
# 做对照时取消速率上限与主动退避；同样只保留一个读取任务在执行
python scripts/camera_service.py --vehicles Drone1 Drone2 --max-total-fps 0 --capture-utilization 1
```

取消视角会等待已经发出的只读请求结束，避免重选后堆积线程；切换响应因此可能等待一次 RPC 超时。降频会降低图像流畅度，不能把控制负载下降直接称为整个系统全面变好。

## 时间预算与监控

路由重规划的两次搜索共享80毫秒协作式预算，候选检查与记忆更新另计。它不是整个控制周期的硬实时保证；Python、数组运算、RPC 和系统调度都可能超时。已有可行候选不会因可选第二次搜索超时而被删除；没有可行路线时保持安全处置。

日志写入本身也会耗时。网关记录诊断后，再以原来用于校验的快照截止时间、原点云采集起始时间和命令许可重新缩短派发时长，不能用新日志或更新的快照延长旧安全判断的有效期。

```powershell
python scripts/summarize_navigation_runtime.py runs/mytest/runtime/Drone1.log runs/mytest/runtime/Drone2.log --output runs/mytest/runtime_profile.json
```

`navigation_runtime` 是有界滑动窗口且约每秒输出一次。汇总中的 `observed_window_max_p95` 是观测窗口 P95 的最大值，不能当成全程帧级 P95，也不能代替碰撞、任务成功和落地清理验收。缺少诊断数据时报告证据不足，不能默认为零延迟。

## 怎样判断改动值得保留

`scripts/compare_navigation_trials.py` 对成对的试验记录作保守筛查。两组必须匹配地图/settings/编队哈希、机数、摄像头数与请求帧率、速度上限、硬净空和点云时效阈值，且 case_id/seed 一一对应。记录须包含真实结果和测量，缺值不会补成成功。

每条输入记录包含：

- `case_id`、实际使用的 `seed`；
- `environment`：`map_sha256`、`settings_sha256`、`formation_sha256`、`vehicle_count`、`camera_count`、`requested_fps`、`speed_limit_mps`、`safety_distance_m`、`hard_clearance_m`、`max_scan_age_s`；
- `metrics`：`success`、`collision_count`、`cleanup_verified`、`minimum_peer_distance_m`、`minimum_obstacle_clearance_m`、`planner_p95_ms`、`sensor_age_p95_ms`、`control_p95_ms`、`task_time_s`、`path_length_m`、`unresolved_deadlocks`。

不得把窗口百分位误填为全程百分位，也不能把局部轨迹对点云的预测净空当作全程实际最小障碍距离。既有验收 summary 没有某个字段时需要补采集；本工具不会凭空把缺失数据转换为可比证据。

```powershell
python scripts/compare_navigation_trials.py --baseline runs/trial/base_records.json --candidate runs/trial/new_records.json --output runs/trial/comparison.json
```

新增碰撞、安全距离不达标、清理未确认、原成功任务失败、未解决堵塞增加或关键时延明显退步会阻止正向结论。每个场景少于5组成对试验、候选仍有失败、证据缺失或条件不一致时不给出改进结论。正向也仅称 `promising` / `promising_with_tradeoffs`，不是统计显著性或普遍优越性的证明；建议实际每个场景至少10组，并记录视频质量损失。

## 本轮证据与后续验收

`benchmark_navigation_hardening.py` 提供1/2/5个 CPU 进程的8000点观测及“旧障碍离开视野”的记忆/路由对照，结果存放于 `artifacts/navigation_hardening/2026-11-23/`。旧 TTL 对照在虚拟观测时间超过有效期后可能选择穿过此前柱体位置的首段路线；新记忆在这些测试中保留了障碍，但增加了处理耗时。这个指标叫“此前静态障碍的首段路线漏检”，**不是实际碰撞次数**。

此次没有运行 UE 或无人机飞行，也没有声称新版本整体优于旧版本。还需在同一主分支提交、地图、settings 和编队配置下，完成单/双/五机、0/1/全部摄像头，以及绕柱、窄通道、队形恢复、B 断连、传感器延迟、控制进程退出和清理确认的实际闭环对照。单机配置需另行采用已经验证的单机部署配置，不能把双机 YAML 改个名字就当成单机试验。
