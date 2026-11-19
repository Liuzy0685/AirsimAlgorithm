# A 机间避让专项证据，2026-11-19

[实现、命令和验证边界](../../../docs/A_机间避让专项验收_2026-11-19.md)。

| 真实试验 | 报告 | 最小采样距离 |
| --- | --- | ---: |
| 双机迎面 | [摘要](head_on/summary.json)、[轨迹](head_on/trajectory.png) | 2.200 m |
| 双机交叉 | [摘要](crossing/summary.json)、[轨迹](crossing/trajectory.png) | 2.214 m |
| 双机追越 | [摘要](overtaking/summary.json)、[轨迹](overtaking/trajectory.png) | 2.170 m |
| 两次运行时切队形 | [报告](formation_switch/report.json)、[轨迹](formation_switch/trajectory.png) | 2.923 m |
| B 主动断开 | [摘要](b_disconnect/summary.json) | 3.000 m |
| 单控制进程协同退出 | [摘要](controller_stop/summary.json) | 3.000 m |
| B 五机编队绕柱 | [摘要](formation_five/summary.json)、[轨迹](formation_five/trajectory.png) | 2.242 m |
| 原 Corridor 双机会车 | [摘要](narrow_corridor/summary.json)、[轨迹](narrow_corridor/trajectory.png) | 2.264 m |

表内均通过对应任务/退出标准，全部确认落地、解锁和释放 API。
三项独立双机 maze 场景增加了临时地面，另附实际地面网格和物理探测来源；
五机编队、Corridor、切队形和退出试验使用原地图地面。
切队形通过真实 HTTP 客户端代替 C 页面发起，未验证浏览器页面或视频链路。

**[五机独立交换终点](five_way_stress_failed/summary.json)未通过**，
保留 [失败轨迹](five_way_stress_failed/trajectory.png)、未验证清理与原始碰撞分类，不能用正常 B 编队成功替代它。
[开发历史](development_history.json) 包含设置拒绝、停滞、高度保护、CPA 保护与清理未确认的尝试。
原始数据和令牌留在本机 `runs/coordination_hardening/`；此目录不包含私有 operator 配置或令牌。

报告记录真实启动时 Python/配置/地图哈希及原始报告哈希。CSV 为全部完整采样轨迹，
图中的黑叉仅是柱体中心；采样、插值与单次通过均不能证明连续时间安全或普遍收敛。

[全量回归](tests.xml)：**1555 passed，254.69 s**；[首轮与最终测试记录](test_history.json)。
[真实只读预检](preflight_live.json)：健康起点接受，0.828 m 近墙起点拒绝，始终未解锁/占用 API。
[八项实测矩阵](matrix.json)、[逐文件源码核对](source_validation.json)、[最终来源指纹](provenance.json)
区分实际架次版本与后补预检验证，没有回填早期实飞哈希。
