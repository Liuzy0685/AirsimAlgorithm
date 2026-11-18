# A：前端任务、地图与执行状态接口（v1）

本轮基于 `372aac5`（已合并障碍感知导航与恢复）。新增独立、本机 HTTP 操作入口，
浏览器不直接发送 AirSim 飞行 API，也不改变原 `/ws/swarm` 中 `c_telemetry` 的只读权限。
既有 B→A `formation.command`、ACK、双时钟期限、permit、ORCA、CPA、LiDAR 和控制网关继续生效。

## 已实现的范围

- 读取能力、实际场景地图与地图版本；校验任务、预览名义槽位。
- 创建任务、修改起飞前队形、启动、请求停止、查询任务与操作结果。
- 持久化请求去重、任务版本检查、单任务运行、重启不重放任务。
- 调用现有独立进程启动器，收集真实起飞/飞行/清理结果；SSE 发布任务阶段变化。
- A 侧运行时队形设置入口、B 专用心跳/拉取/结果/状态回报接口。
  **B 当前网络服务尚未接入这些新增接口**，因此飞行中修改队形会返回
  `503 b_control_unavailable`，不会伪报成功。起飞前选择队形已经可用。

第一版固定使用服务启动时的完整机群与领航机、配置速度和巡航高度；
不支持飞行中修改终点、暂停后继续、编辑出生点或换机群。`capabilities.supports` 明确返回支持情况。
默认从当前实测位置起飞；不会按网页坐标瞬移飞机。名义终点槽位预览不是无碰撞路径保证。

## 启动

先启动自己的 Blocks `/Game/maze`，加载与双机或五机配置对应的 settings。
不要覆盖 Documents/AirSim/settings.json，也不要在任务飞行时切地图。

```powershell
$py='C:\Users\Liuziyi\AppData\Local\Python\pythoncore-3.14-64\python.exe'
$env:AIRSIM_PYTHONCLIENT_PATH='D:\30817\Adrone\airsim\AirSim-main\PythonClient'

# 只读导出真实柱体网格包围盒；更换或修改地图后重新导出。
& $py scripts/export_operator_map.py --blocks-project D:\30817\Adrone\blocks\Blocks.uproject --map-id maze --bounds -5 45 -16 16 --output configs/operator/maze.json

& $py scripts/operator_api.py --settings-json configs/acceptance/settings_2.json --swarm-config configs/acceptance/swarm_2_navigation.yaml --formation-config configs/acceptance/formation_2.yaml --map-manifest configs/operator/maze.json --blocks-project D:\30817\Adrone\blocks\Blocks.uproject --runtime-dir runs/operator --allow-origin http://127.0.0.1:5173 --allow-flight
```

默认监听 `http://127.0.0.1:8790/api/v1`；省略 `--allow-flight` 时启动请求被拒绝。
五机替换三份机群/settings/formation 配置，地图 manifest 可在相同未变更地图下共用。
绑定仅允许 IPv4 回环；C 若使用 `localhost:5173`，须显式增加对应 `--allow-origin`。
令牌自动生成于 `runs/operator/operator.token`，不打印、不上传 Git。
前端通过 `Authorization: Bearer <令牌>` 调用；不要把令牌放在 URL、截图或公开前端源码中。
后台运行可指定 `--stop-file runs/operator/api.stop`；创建该文件会请求 API 正常退出，
仍会等待活动任务清理，超时不会强杀控制进程。前台运行使用 Ctrl+C。

## 通用约定

- JSON 采用 UTF-8；写请求使用 `Content-Type: application/json`，最大 64 KiB。
- 所有位置是公共 World NED 米：X 北、Y 东、Z 下。正数高度不能直接填入 Z。
- `request_id` 是前端生成的唯一编号，只允许字母、数字、`_`、`-`，1～80 字符。
  同编号同路由同载荷返回原接收凭据；内容不同返回 `409 request_id_conflict`。
- 同一运行目录最多保留 128 个任务、2048 份接收凭据；超限拒绝新建，不静默遗忘去重历史。
- `revision` 是任务配置版本，写起飞/队形设置请求时必须填写 `expected_revision`。
- JSON 中拒绝重复字段、NaN、Infinity、不支持的字段及越界参数。
- `accepted=true` 仅表示操作请求进入处理，不代表完成起飞、恢复队形或落地。
  重连后查询任务/操作，不重新发送新的启动请求。

## 对 C 的 HTTP 接口

| 方法与路径（均相对 `/api/v1`） | 输入 | 输出要点 |
| --- | --- | --- |
| `GET /capabilities` | 无 | 机群、领航机、队形、可调范围、固定飞行限制、可用功能 |
| `GET /maps/current` | 无 | `map_id`、`map_revision`、NED 范围、障碍 AABB、覆盖类型 |
| `POST /missions/validate` | 下述完整任务请求 | 实测起点、名义终点/初始槽位预览、警告；不启动 |
| `POST /missions` | 完整任务请求 | `accepted`、`mission`；成功 201 |
| `GET /missions/{mission_id}` | 无 | 阶段、配置版本、运行状态、各机清理结果 |
| `POST /missions/{id}/start` | `request_id, expected_revision` | 操作接收凭据；成功 202 |
| `POST /missions/{id}/stop` | `request_id` | `completed=false`；等待实际清理结果 |
| `PUT /missions/{id}/formation` | `request_id, expected_revision, formation` | 草稿设置结果，或运行时请求待处理状态 |
| `GET /operations/{request_id}` | 无 | 启动/停止/编队操作的最新结果 |
| `GET /events?after={sequence}` | SSE 游标 | `update` 事件及最新序号；必要时要求重载任务 |

任务请求例子（`map_revision` 必须读取接口，不能写死）：

```json
{
  "request_id": "create-unique-001",
  "map_id": "maze",
  "map_revision": "从 maps/current 获取",
  "vehicle_names": ["Drone1", "Drone2"],
  "leader": "Drone1",
  "leader_goal_ned_m": [15.5, -3.0, -2.0],
  "formation": {"shape": "line", "spacing_m": 3.0}
}
```

`shape` 支持 `line / column / vee`。间距范围为机间安全距离加 0.7 m 至 8 m，
还须满足名义终点槽位的地图边界和已知障碍净空检查。某一种队形在某个终点可能被拒绝，
例如不能把横队可用的领航终点直接假定为纵队也可用。
任务校验和创建执行只读实测预检；启动时再次预检，不使用旧验证结果作为飞行许可。

```json
{"request_id":"start-unique-001","expected_revision":1}
```

```json
{"request_id":"shape-unique-001","expected_revision":1,"formation":{"shape":"vee","spacing_m":3.0}}
```

起飞前队形修改返回 `operation.state=APPLIED_TO_DRAFT`，任务版本递增，之后用新版本启动。
飞行中队形设置须等 B 接线并持续心跳，才会返回 `PENDING`。收到 B 的 `APPLIED` 仅说明 B
采用了配置，**不是物理编队已经恢复**；物理状态仍看 A 的 `coordination.status` 以及 B 新增状态回报。

## 任务与执行结果

阶段：`READY → STARTING → TAKING_OFF → RUNNING → LANDING → COMPLETED`；
主动停止经过 `STOPPING → STOPPED`，异常为 `FAILED`，API 重启中断且缺少清理证据为 `INTERRUPTED`。
传感器 `landed_state` 在起飞早期就可能是 Flying，因此 TAKING_OFF/RUNNING 是粗粒度展示；
不代替控制进程内部阶段，也不是“控制循环已经开始”的许可。

`runtime` 含 `timestamp_unix_ms`、`snapshot_id`、`stream_id`、`collection_complete`、
公共坐标车辆状态及原 A 协调状态。API 状态采集是独立观察连接，不抢占权威状态服务。
超过 2 秒显示 `stale=true`；该时间是展示判断，不能延长控制指令期限。
SSE 推送阶段变化；C 按需轮询当前任务或继续订阅现有遥测适配服务以更新连续运动画面。
浏览器原生 EventSource 不支持自定义 Authorization，使用 `fetch` 流读取 SSE；
连接最长约 25 秒，带 `after` 游标重连，`reset_required=true` 时重新查询任务。

完成判定必须包含：启动器退出、领航 `mission_complete`、各机终止原因符合预期，
全部确认落地、解锁、释放 API 控制与控制租约。单独退出码 0 不算完成。
API 记录到的 `applied_formation` 是配置状态；实际飞行可以因避障进入临时跟随模式。
没有 B 运行时状态时应显示“编队模式未接入”，不能猜测 `nominal/trail`。

失败清理若被安全网关拒绝，`manual_intervention_required=true`，禁止后续启动。
不能为了让任务显示成功而绕过感知校验或强制清除记录。模拟器恢复/重启也不算落地成功。
API 意外退出后不自动重启任务，保留停止文件入口，并在原 runner 最终结果出现时重新核对清理。

## 地图的准确含义

`configs/operator/maze.json` 为真实 19 个柱体的渲染网格 AABB 导出；只使用三角形实际引用的顶点，
并与全部 Actor 位姿校准 UE 世界厘米到公共 NED 米的转换，不依据中心坐标猜尺寸。
启动前复核场景、地图资产 SHA256、柱体位姿/旋转/缩放及车辆配置，变化后要求重新导出。
目前 `coverage=partial_cylinder_render_meshes_only`，**不包含全部碰撞体、地形或临时新增物体**。
编辑范围不是已确认可飞区域，渲染包围盒也不是碰撞认证；`collision_certified=false`。
未观测区域与已知柱体分层展示，实际执行仍以实时感知和 A 网关为准。

## 主要拒绝原因

| HTTP | error | 含义 |
| --- | --- | --- |
| 401 / 403 | `unauthorized / origin_denied / flight_disabled` | 凭据、页面来源或飞行开关不允许 |
| 409 | `request_id_conflict / revision_conflict / map_revision_mismatch` | 请求或配置版本冲突 |
| 409 | `vehicle_not_released / state_service_in_use / mission_active` | 飞机或运行资源仍被占用 |
| 409 | `runtime_map_mismatch / map_geometry_changed` | 当前仿真与地图记录不一致 |
| 422 | `goal_near_known_obstacle / goal_out_of_bounds / goal_outside_geofence` | 目标或名义槽位不适用 |
| 422 | `unsupported_altitude / unsupported_fleet` | 第一版不支持该高度或机群修改 |
| 503 | `simulator_unavailable / b_control_unavailable` | 仿真不可用或 B 新接口未接线 |

错误格式统一为 `{"error":"机器可读原因","message":"可展示说明"}`。
视频服务独立由 C 接入；本 API 不读取/编码十路视频，也不提供网页直接飞行控制权限。

## B 的接线说明与验证

B 按 [B 运行时编队接口接线清单](B_运行时编队接口接线清单.md) 补充。
已有纯算法和门控不需要重写；初始任务通过 A 生成的独立 YAML 绑定真实任务 ID。
运行目录和令牌不得提交 Git。实测、回归结果见本轮 `artifacts/operator_api/`。

真实 HTTP / AirSim 最终验收 17/17 通过：完整双机任务领航巡航约 97.02 s，
跟随机协同结束；另一次起飞早期请求停止，两机也均确认落地、解锁、释放控制。
API 采样最小机间距 2.27782 m。实测后补充了围栏按本次实际起点计算的预检修正，
修正后的真实只读预检通过，控制/清理路径源码不变。开发失败记录和实际源码指纹均保留，
详见 [本轮证据](../artifacts/operator_api/2026-11-18/README.md)。

最终全量回归：`python -m pytest -q`，**1450 passed，251.70 s**；
报告位于 `artifacts/operator_api/2026-11-18/tests.xml`。
新增 32 项回归覆盖接口鉴权、JSON/数值/任务版本、请求去重、并发任务、
真实起点围栏、B 配置/状态时效与权限、持久化恢复、停止清理及网格无效尾部数据。
