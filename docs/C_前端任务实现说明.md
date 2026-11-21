# C 前端任务实现说明

本目录的前端已经按最新版 `main` 的 [A 前端任务接口](A_前端任务接口.md) 改为真实任务控制台。

## 责任边界

- 页面只调用 A 的 `http://127.0.0.1:8790/api/v1`，不直接调用 AirSim 飞行 API。
- 预检、创建、启动、停止和队形修改都使用后端 `request_id`。重复点击会复用同一个编号；断线后页面只查询任务和操作结果，不自动重发飞行请求。
- 地图点击只设置领航机的 World NED 终点。起点由 A 预检返回的 `actual_start` 提供，网页不会瞬移或修改无人机出生点。
- 跟随机目标由 B/A 计算，页面展示预检返回的 `terminal_slots_preview`，不把槽位线画成安全路径。
- 任务状态和操作结果以 A 的回报为准。收到“请求已接受”时不会直接显示已经起飞或已经落地。
- 第一视角和第三视角由独立的 `scripts/camera_service.py` 提供，页面不读 AirSim RPC。没有图像服务时显示“未配置”，不伪造画面。

## 运行顺序

以下命令统一在最新版工作区 `D:\大创无人机\AirsimAlgorithm-main` 执行。旧目录 `D:\大创无人机\AirsimAlgorithm` 和 `D:\大创无人机\AirsimAlgorithm-C` 保留着此前的分支和修改，不要从旧目录启动这套新版服务。

```powershell
# 1. 先在 Unreal/AirSim 中启动 Blocks 地图；不要在任务飞行中切地图。

# 2. 在最新版仓库启动 A 操作 API（PowerShell）
cd D:\大创无人机\AirsimAlgorithm-main
$env:AIRSIM_PYTHONCLIENT_PATH = 'D:\AirSimDev\AirSim\PythonClient'
$env:PYTHONPATH = 'D:\AirSimDev\AirSim\PythonClient'
python scripts/operator_api.py `
  --settings-json configs/acceptance/settings_2.json `
  --swarm-config configs/acceptance/swarm_2_navigation.yaml `
  --formation-config configs/acceptance/formation_2.yaml `
  --map-manifest configs/operator/maze.json `
  --blocks-project 'D:\AirSimDev\AirSim\Unreal\Environments\Blocks\Blocks.uproject' `
  --runtime-dir runs/operator `
  --allow-origin http://127.0.0.1:8000 `
  --allow-flight
```

如果 Python 不在 PATH 中，先把命令里的 `python` 换成这台电脑已安装的 Python 3.11 可执行文件。A API 启动后会生成 `runs/operator/operator.token`。把文件内容输入页面的“操作令牌”框；令牌不要放进 URL、截图或 Git。页面会把令牌保存在当前浏览器的 localStorage 中，刷新或重新打开网页后仍保留；需要清除时请在浏览器站点数据中删除。

```powershell
# 3. 另开 PowerShell 窗口，在同一新版工作区启动只读遥测网关和网页
cd D:\大创无人机\AirsimAlgorithm-main
python -m telemetry.server --config configs/telemetry.yaml
```

页面地址是 `http://127.0.0.1:8000`。任务运行后，A 的运行时状态服务会提供 `/ws/swarm`，本服务按协议版本 1 订阅并转发到页面 `/ws/telemetry`。没有任务运行时，页面可以打开，但连续运动遥测可能为空或显示离线；页面不会生成假飞行数据。

```powershell
# 4. 另开 PowerShell 窗口启动独立、只读的低帧率图像服务
cd D:\大创无人机\AirsimAlgorithm-main
$env:AIRSIM_PYTHONCLIENT_PATH = 'D:\AirSimDev\AirSim\PythonClient'
$env:PYTHONPATH = 'D:\AirSimDev\AirSim\PythonClient'
python scripts/camera_service.py --vehicles Drone1 Drone2 --interval 0.5
```

图像服务成功启动后，在页面填写 `http://127.0.0.1:8780`。双机/五机 settings 必须与所选机群一致，并包含每架机命名为 `{vehicle}Third` 的外部相机；服务会在启动时检查配置，缺少相机定义时拒绝采集，避免向 AirSim 查询不存在的相机。页面的 A 操作令牌保存在当前浏览器的 localStorage 中，不写入 URL。

如果相机画面离线，先检查服务窗口给出的错误和 AirSim 相机名称。图像服务使用机载相机 `0` 读取每架机前向视角，并分别读取、移动 `Drone1Third`、`Drone2Third` 等外部相机到对应无人机后上方；它只读位姿/图像，不负责起飞、移动或降落。图像为 320×240，默认采集间隔 0.5 秒。

## 页面使用流程

1. 点击“连接并读取能力”，页面读取 `/capabilities` 和 `/maps/current`。
2. 点击地图选择领航机终点，或直接填写 X/Y/Z。Z 必须是 NED 的向下坐标，不能把正高度直接填入 Z。
3. 选择 `line`、`column` 或 `vee`，填写间距。
4. 点击“预检任务”，确认实测起点、名义槽位和警告；这一步不会启动飞行。
5. 点击“创建任务”，再点击“请求启动任务”。页面会显示请求已接受，随后根据任务 API 查询 `STARTING`、`TAKING_OFF`、`RUNNING` 等真实状态。
6. 需要结束时点击“请求停止并降落”，直到后端确认 `STOPPED` 或 `COMPLETED`，并查看各机清理结果。
7. 浏览器断线或刷新后，点击“连接并读取能力”；页面会使用浏览器保留的令牌查询现有任务和未完成操作，不会自动重发启动或停止请求。点击“断开连接”只停止网页连接，不停止 Unreal、A API 或相机服务。

运行中队形切换只有在 `/capabilities` 表示 B 运行时通道可用时才有意义。收到 `503 b_control_unavailable` 时页面会显示原因，不会显示切换成功。

## 双视角接口

`camera_service.py` 暴露：

- `GET /stream/{vehicle}/first`：该机机载前向相机。
- `GET /stream/{vehicle}/third`：独立外部相机的后上方跟随视角。
- `GET /api/cameras`：每架机两个视角的最近采集状态。

图像服务使用单独的只读 AirSim RPC 客户端，第三视角采用缓存后的低帧率图像，避免浏览器刷新或视频采集抢占飞行控制连接。AirSim 中没有可用外部相机时，第三视角会显示离线原因；这时需要在 `settings.json` 中配置 ExternalCamera 或改用已有外部相机名称。

地图的 `coverage` 如果是 `partial_cylinder_render_meshes_only`，只表示部分柱体渲染网格包围盒已导出。它不是完整碰撞地图，也不是可飞区域认证；实际避障仍由 A 的实时感知和控制网关负责。
