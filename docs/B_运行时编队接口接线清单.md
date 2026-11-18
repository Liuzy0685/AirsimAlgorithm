# 给 B：运行时编队接口接线清单（对接 A operator API v1）

A 已补任务、地图、启动/停止与队形设置入口。你已有的三种槽位计算、状态门控、
避障恢复和 `formation.command` 保留；本次主要补网络接线和执行状态反馈。
**没有接线前，A 飞行中切换队形明确返回 503；起飞前选择三种队形可直接通过生成配置生效。**

## 1. 读取 A 为当前任务生成的配置

A 启动器继续用 `--swarm-config` 和 `--formation-config` 启动现有 `scripts/formation_service.py`。
生成的 swarm YAML 新增可选字段：

```yaml
operator_api:
  version: 1
  base_url: http://127.0.0.1:8790/api/v1
  mission_id: 本次任务编号
  initial_revision: 任务初始配置版本
  bearer_token: 只用于当前B进程的随机令牌
```

没有该段时按旧方式运行，不启动配置接口轮询。
有该段时核对 `mission_id == state_service.task_id`；初始队形仍从 formation YAML 读取。
内部接口使用 `Authorization: Bearer <bearer_token>`；不要使用前端的 operator.token，
不要打印、截图、提交生成的 YAML 或令牌。A 的公开任务/事件数据不会包含 B 令牌。

以下 `{base}` 为 `operator_api.base_url`，`{mid}` 为 `mission_id`。

## 2. 声明新配置通道在线

```http
POST {base}/internal/missions/{mid}/formation/heartbeat
```

```json
{"protocol_version":1,"capabilities":["formation.configure"]}
```

建议每 0.5 秒发送，A 在 2 秒未收到后视为配置通道失联。只有真正安装好接收和应用处理器后
才能声明 capability，不能为让前端按钮亮起就只发心跳。失败后重连不得重放旧飞行速度。
这条心跳与原 `/ws/swarm` 的 B 心跳独立，两条都要维护。

## 3. 拉取待处理队形配置

```http
GET {base}/internal/missions/{mid}/formation/request
```

无待处理时返回 `{"request":null}`；有请求时例如：

```json
{
  "request": {
    "request_id":"shape-unique-001",
    "mission_id":"本次任务编号",
    "revision":4,
    "formation":{"shape":"vee","spacing_m":3.0},
    "expires_at_unix_ms":1795010000000
  }
}
```

建议 5 Hz 拉取，HTTP 超时不要阻塞原 WebSocket 接收和飞行规划循环。
检查任务 ID、配置版本、请求 ID 和 A 给出的配置期限；期限已到不应用，A 也会拒绝过期结果。
同请求只能应用一次，重复拉取不能反复重置规划器或重放旧配置。
这里是持久任务意图，**不是速度命令**；原 FormationCommand 的仿真 ns + 本地 TTL 期限保持原样。

## 4. 应用或暂缓，并反馈结果

```http
POST {base}/internal/missions/{mid}/formation/result
```

```json
{
  "request_id":"shape-unique-001",
  "revision":4,
  "state":"DEFERRED",
  "formation":{"shape":"vee","spacing_m":3.0},
  "reason":"avoiding"
}
```

字段必须齐全，不能额外添加；`reason` 最多 256 字符。

- `DEFERRED`：状态门控不允许切换，如 AVOIDING/冲突未解除；保留旧配置并等待，不能绕过门控。
- `APPLIED`：已实际更新 B 规划器配置；只是“算法采用了新配置”，不代表飞机已组成新队形。
- `REJECTED`：参数或当前状态不允许，填写具体原因。

`request_id/revision/formation` 必须与 A 请求一致。A 返回 409 时停止应用该请求，重新拉取当前状态。
更新应在原规划循环的明确边界原子完成：`FormationConfig`、`FormationController`、
`FormationNavigation` 的名义间距及临时跟随间距必须一致。不能只改 `switch_shape()` 而漏掉间距，
也不能只修改磁盘 YAML。对待处理恢复状态与历史路径缓存作一致处理，避免半帧使用两套配置。

应用后仍按原 `FormationController.plan → FormationCommand → A ingress/ACK/permit → A 安全网关`
输出目标，不直接调用 AirSim，不把前端参数当成最终速度。

## 5. 发布当前编队状态，供 C 正确展示

```http
POST {base}/internal/missions/{mid}/formation/status
```

```json
{
  "revision":4,
  "nominal_shape":"vee",
  "effective_mode":"trail",
  "slot_targets_ned_m":{"Drone2":[10.0,-3.0,-2.0]},
  "tracking_error_m":{"Drone2":1.2},
  "sim_timestamp_ns":1795010000000000000,
  "reason":"obstacle_constrained"
}
```

以上数值仅说明格式。只填真实本轮目标、误差和 A 快照时间，不能使用固定示例值。
建议 2 Hz，revision 使用 B 实际已采用的版本；不能用尚未应用的 pending revision。
`effective_mode` 为 `nominal / trail / recovering / holding`，暂时沿历史路径跟随时报告 trail，
不把用户选中的横队直接当作当前运行模式。槽位和误差必须恰好覆盖本任务全部跟随机。
A 校验当前任务凭据、采用版本及相对权威快照的仿真时效；重复旧时间戳不刷新状态年龄。
没有反馈或过期时，前端显示“实际编队模式未知”，而不是推断已经恢复。

## 交付与验收

1. 无 operator_api 配置时，原启动和门控测试继续通过。
2. 初始三种队形与任务 ID 对应正确；配置 HTTP 异常不阻塞规划循环。
3. 避障时请求切换，返回 DEFERRED；安全状态恢复后再 APPLIED，并观察实际重新集结。
4. 重复 ID、旧版本、过期请求、错误任务、错误令牌均不产生重复配置或飞行派发。
5. B 配置通道断连时，A 的 pending 操作失败或过期，不能显示切换完成。
6. 前端可同时区分请求队形、采用配置、临时跟随模式和物理恢复状态。
7. 新增单元测试与 A/B/C 联调记录，保留未通过和超时记录。

A 侧接口已有测试替身验证；**仍需你接入真实 B 后做飞行中队形切换验收**，
不能用当前两机绕柱通过记录代替这项尚未完成的联调。
