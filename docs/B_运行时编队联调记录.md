# B 运行时编队接线联调记录

日期：2026-11-19。代码范围：`formation/operator_link.py`、`scripts/formation_service.py`、`formation/navigation.py`。

## 已完成的离线联调

- 可选 `operator_api` 配置校验：无配置仍走原 WebSocket 编队服务；任务 ID、版本、回环地址与 B 专用令牌分别检查。
- 与 A 的真实本机 HTTP 服务替身完成心跳、请求拉取、`DEFERRED`、`APPLIED`、状态上报和重放拒绝测试。
- 请求在避障与配置通道失联时暂缓；重复、旧版本、过期、错误任务和无效队形不修改规划器。HTTP 409 时回退本地配置并停发编队命令。
- 更新规划器与导航器的名义/临时跟随间距发生在快照处理边界；HTTP 调用在独立任务中，不阻塞 WebSocket 规划循环。
- 状态上报使用本轮 A 快照的目标、跟踪误差和仿真时间；`APPLIED` 只代表 B 算法采用，状态中的 `nominal`、`trail`、`recovering`、`holding` 单独表示实际运行模式。

运行：`python -m pytest tests/test_formation_operator_link.py tests/test_formation_runtime.py tests/test_formation_controller.py tests/test_operator_api.py tests/test_swarm_wiring.py tests/test_obstacle_navigation.py -q`

结果：100 passed；`python -m py_compile formation/operator_link.py scripts/formation_service.py formation/navigation.py` 通过。

## 尚待真实飞行验收

本机 `127.0.0.1:41451` AirSim RPC 未监听，无法执行飞行中的队形切换。本次没有伪造飞行通过记录。待仿真实例运行后，按 `docs/B_运行时编队接口接线清单.md` 使用 operator API 发起 `line → vee` 或 `column → line` 请求，保留任务事件、B 日志、A 快照/状态、轨迹 CSV 与仿真版本。验收需观察 `PENDING → DEFERRED（避障时）→ APPLIED`、随后真实目标与误差逐步收敛；`APPLIED` 本身不算物理队形恢复。
