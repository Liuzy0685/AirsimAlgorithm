# FlyingExampleMap 五机零 XY：任务终点与地图清单排查

## B 日志中已经确认的事实

输入为 `formation_live_20261123_v_open(2).zip`。Drone1 约50.84秒、343帧，日志位姿 XY 位移为0；59条路线事件中58条不可用、目标 `(30.5,-6,-2)` 不变，目标阻挡标志均为false。路线曾短暂可用，随后在有效点云到达后反复无路。ORCA记录的首选与实际XY速度都为零。

其余四架主要停在自身初始目标等待，不能将它们全部描述成各自路线不可用。B命令有1028次accepted ACK、0次拒绝。五架最终均以coordinated_stop结束，确认落地、解除武装和API释放。该轮未完成任务，但没有留下B所报那次未确认清理的问题。

Drone1全程平均控制频率6.74 Hz，周期P95约188毫秒，有234次控制超时警告；性能仍需另行排查。旧的342次hover/0次trajectory是轻量导航分支的来源计数，不能独立当成实际悬停/未规划的证明；本次零位移由位姿与指令另外确认。

压缩包没有提交SHA、实际settings/YAML内容、地图资产哈希或完整规划点云，不能凭日志唯一断定B现场的搜索拒绝原因，也不能判定是否同时运行两个UE实例。

## 本机地图发现：该任务目标位于方块内

本机资产 `/Game/FlyingCPP/Maps/FlyingExampleMap` 的SHA256：

`a0fa58a356583f8642dba9585315a670e86f719af5358a8fbb1120a067068a84`

只在五机落地、API均关闭时读取静态渲染网格，未改地图、车辆位姿或资产。`TemplateCube_Rounded_75` 的实测AABB为：

| 轴 | 最小值（米） | 最大值（米） | 任务目标 |
| --- | ---: | ---: | ---: |
| X | 23.10 | 33.10 | 30.50 |
| Y | -11.50 | -1.50 | -6.00 |
| Z | -3.98 | 1.02 | -2.00 |

对该网格204个三角面进行六个方向射线奇偶检验，目标均被判定在闭合渲染网格内，不仅是AABB边缘重叠。碰撞网格未导出，渲染几何不等同于完整碰撞认证。

旧清单只包含圆柱体，这些圆柱体远离本任务编辑区域；地图里大量TemplateCube方块没有显示或参与已知障碍终点预检。因此不能把地图名称或空白前端显示当成“航路无障碍”的证据。

用实测前表面范围构造的**合成观测夹具**，可得到 `goal_blocked=false`、`no_grid_path`、零位移的判定：表面点地图不会把实体内部自动标为占据，目标阻挡false不代表目标可达。这不是B原始LiDAR的精确回放。

用户已核对B的 `D:\blocks\blocks\Content\FlyingCPP\Maps\FlyingExampleMap.umap`，哈希与上述一致。仍需核对本次运行真正加载的关卡、对象变换与依赖静态网格，不能只凭 `.umap` 相同就断定运行场景完全相同。本机的默认终点已经判为无效；不应降低净空、延长点云时效或关闭碰撞检查让它飞进去。B现场的精确路由原因仍需完整快照或运行时几何核验。

## 修复范围与复测方式

- 扩展只读地图导出器，可显式选择Cylinder/TemplateCube前缀并保留与编辑范围相交的完整AABB；默认圆柱体导出方式兼容原调用。
- 更新 `configs/operator/flying_example.json`，保留原地图资产哈希、Ground身份参考与编辑范围，补入32个实测静态方块。清单仍是部分渲染几何，不承诺无未知障碍。
- 增加 `scripts/check_mission_endpoints.py`，只读检查共享配置中的领航机世界终点及当前队形终端槽位；不连接AirSim，不自动换目标，不授权起飞。
- `launch_swarm.py --map-manifest` 会在任何飞控RPC/控制进程启动前执行上述额外拒绝门。检查通过不能替代实际地图、状态、传感器、控制权及碰撞预检。前端任务预览使用新版清单后，也会通过现有接口拒绝这个目标。

B先核对实际 `.umap` 哈希，再在同一提交下执行：

```powershell
python scripts/check_mission_endpoints.py --settings-json configs/acceptance/settings_5.json --swarm-config configs/acceptance/swarm_5_navigation.yaml --formation-config configs/acceptance/formation_5.yaml --map-manifest configs/operator/flying_example.json
```

对于当前默认目标，预期退出码2，原因 `goal_near_known_obstacle`，对象 `TemplateCube_Rounded_75`。这说明任务输入应被拒绝，不是飞行验收通过。已知地图的直接启动命令增加 `--map-manifest configs/operator/flying_example.json` 后同样拒绝，仍不需改settings/导航/编队YAML。

若要核对B运行时关卡和依赖网格，保持所有飞机落地、静止且API关闭，在UE运行但未启动控制器时执行（engine-log须换成本次UE的真实日志）：

```powershell
python scripts/check_runtime_goal_geometry.py --settings-json configs/acceptance/settings_5.json --swarm-config configs/acceptance/swarm_5_navigation.yaml --formation-config configs/acceptance/formation_5.yaml --map-manifest configs/operator/flying_example.json --blocks-project "D:\blocks\blocks\Blocks.uproject" --engine-log "D:\blocks\blocks\Saved\Logs\Blocks.log" --output runs/flying_runtime_geometry_B.json
```

此命令不获取API控制权、不移动飞机，只读校验加载关卡、资产哈希、清单中的对象位置/缩放/朝向，重新测量目标方块的渲染AABB，焊接渲染接缝检查网格闭合性后进行六方向射线判定。预期本机结果为 `active_map_verified=true`、`render_aabb_matches_manifest=true`、`inside_render_mesh=true`，退出码2表示目标应拒绝。如果几何不匹配、网格不闭合或报错，应保留JSON/错误重新核对，不能将不确定结果当作飞行许可。即使退出码0，`flight_authorized`仍为false。

需由任务制定方通过前端任务接口明确选择合法终点并预检，不能偷偷替换默认目标。C重启后端加载新清单后刷新地图和预览，无需新增相机；B无需提交另一份settings。合法目标下如果仍无路，再取新诊断快照排查A的记忆/栅格/预算，控制负载问题也继续保留。

## 本次验证与限制

本分支已合入 `main` 的 `686cd94`，包含B的 `4901621` 编队更新；无文本冲突。相关回归294项通过，覆盖五机空地/队友几何、地图导出、端点拒绝、启动前零RPC拒绝、编队过渡、运行时只读检查与既有网关。运行时核验命令在本机落地/API关闭实例上验证成功，返回上述目标在渲染网格内部的结果。

本机尝试的五机飞行诊断在进入巡航前因 `authoritative_snapshot_unavailable` 失败，没有成功复现B的原场景。自动清理曾未确认；后续显式恢复结合已有解除武装确认/未进入武装阶段的审计与新鲜地面观测，最终确认五机落地、API释放并关闭实例。原失败结果未改写，单独保留恢复记录。这一事件不归因于B的槽位算法，也不算五机飞行通过。

几何读取全部在地面无API控制时完成，现有碰撞门控没有关闭。运行证据、原始B压缩包、恢复记录、网格射线检验和合成夹具见 `artifacts/open_map_five/2026-11-24/`。代码在测试分支/PR中，尚未宣称合入main或解决全部五机验收问题。
