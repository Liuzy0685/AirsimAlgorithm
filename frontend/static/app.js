(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const COLORS = ["#55c9f4", "#a78bfa", "#8df0bf", "#ffd166", "#ff8d7a"];
  const TERMINAL_MISSION_STATES = new Set(["COMPLETED", "STOPPED", "FAILED", "INTERRUPTED"]);
  const ACTIVE_MISSION_STATES = new Set(["STARTING", "TAKING_OFF", "RUNNING", "STOPPING", "LANDING"]);
  const TERMINAL_OPERATION_STATES = new Set(["COMPLETED", "FAILED", "APPLIED", "APPLIED_TO_DRAFT", "REJECTED"]);

  function storedValue(key, fallback = "") {
    try {
      const value = localStorage.getItem(key);
      if (value !== null) return value;
    } catch (_) { /* Browser storage may be blocked; use the session fallback. */ }
    try { return sessionStorage.getItem(key) || fallback; } catch (_) { return fallback; }
  }

  function setStoredValue(key, value) {
    try {
      if (value) localStorage.setItem(key, value);
      else localStorage.removeItem(key);
    } catch (_) { /* Keep the in-memory value when persistent storage is unavailable. */ }
    try {
      if (value) sessionStorage.setItem(key, value);
      else sessionStorage.removeItem(key);
    } catch (_) { /* Best effort only. */ }
  }

  function storedJson(key, fallback) {
    try { return JSON.parse(storedValue(key, JSON.stringify(fallback))); }
    catch (_) { return fallback; }
  }

  const savedCameraSelection = storedJson("uav.camera.selection", { vehicle: "", view: "first" });
  const initialCameraSelection = Array.isArray(savedCameraSelection?.cameras)
    ? savedCameraSelection.cameras
    : [{ vehicle: savedCameraSelection?.vehicle || "", view: savedCameraSelection?.view || "first" }];

  const state = {
    apiBase: storedValue("uav.operator.apiBase", "http://127.0.0.1:8790/api/v1"),
    token: storedValue("uav.operator.token").trim(),
    cameraBase: storedValue("uav.camera.base"),
    savedVehicleNames: storedJson("uav.camera.vehicleNames", []),
    cameraSelection: { cameras: initialCameraSelection },
    cameraFps: Number(storedValue("uav.camera.fps", "5")),
    cameraStatus: {},
    cameraServiceOnline: false,
    cameraNamesKey: "",
    cameraRefreshBusy: false,
    cameraFrameAbort: null,
    cameraSelectionBusy: false,
    cameraLightboxOpen: false,
    capabilities: null,
    map: null,
    mission: null,
    validation: null,
    validationSignature: "",
    operation: null,
    savedMissionId: storedValue("uav.operator.missionId"),
    requestIds: storedJson("uav.operator.requestIds", {}),
    requestSignatures: storedJson("uav.operator.requestSignatures", {}),
    operationTimer: null,
    missionTimer: null,
    eventAbort: null,
    telemetrySocket: null,
    telemetryDisconnected: false,
    telemetryFrame: null,
    runtimeFrame: null,
    trails: new Map(),
    lastStream: null,
    selectedVehicle: "",
    selectedGoal: null,
    formMissionId: "",
    // The map is rendered in the map's World NED bounds.  mapZoom changes the
    // visible projection as well as the click-to-NED conversion; it is not a
    // cosmetic button state.
    mapZoom: 1,
    eventSequence: Number(storedValue("uav.operator.eventSequence", "0")),
    backendLoading: false,
    recoveryNotice: storedJson("uav.operator.recoveryNotice", null),
  };

  $("api-base").value = state.apiBase;
  $("operator-token").value = state.token;
  $("camera-base").value = state.cameraBase;

  function setPill(id, online, text) {
    const node = $(id);
    node.textContent = text;
    node.className = `status-pill ${online ? "online" : "offline"}`;
  }

  function message(text, kind = "") {
    const node = $("operator-message");
    node.textContent = text;
    node.className = `message ${kind}`;
  }

  function requestMessage(text, kind = "") {
    const node = $("request-message");
    node.textContent = text;
    node.className = `message ${kind}`;
  }

  function renderRecoveryNotice() {
    const node = $("recovery-notice");
    if (!node) return;
    const notice = state.recoveryNotice;
    if (!notice) {
      node.hidden = true;
      node.replaceChildren();
      return;
    }
    const when = new Date(Number(notice.at || Date.now())).toLocaleString();
    node.hidden = false;
    node.className = `recovery-notice ${notice.kind === "error" ? "error" : "success"}`;
    node.innerHTML = `<div class="recovery-notice-heading"><strong>${notice.kind === "error" ? "安全恢复检查未通过" : "安全恢复检查记录"}</strong><span>${escapeHtml(when)}</span></div><div><strong>${escapeHtml(notice.code || "recovery")}</strong>：${escapeHtml(notice.message || "—")}</div>${notice.detail ? `<p class="muted">${escapeHtml(notice.detail)}</p>` : ""}<button id="clear-recovery-notice" class="text-button" type="button">清除这条记录</button>`;
    $("clear-recovery-notice").addEventListener("click", clearRecoveryNotice);
  }

  function setRecoveryNotice(kind, code, text, detail = "") {
    state.recoveryNotice = { kind, code, message: text, detail, at: Date.now() };
    setStoredValue("uav.operator.recoveryNotice", JSON.stringify(state.recoveryNotice));
    renderRecoveryNotice();
  }

  function clearRecoveryNotice() {
    state.recoveryNotice = null;
    setStoredValue("uav.operator.recoveryNotice", "");
    renderRecoveryNotice();
  }

  function persist() {
    setStoredValue("uav.operator.apiBase", state.apiBase);
    setStoredValue("uav.camera.base", state.cameraBase);
    setStoredValue("uav.operator.token", state.token);
    setStoredValue("uav.camera.vehicleNames", JSON.stringify(state.savedVehicleNames));
    setStoredValue("uav.camera.selection", JSON.stringify(state.cameraSelection));
    setStoredValue("uav.camera.fps", String(state.cameraFps));
    setStoredValue("uav.operator.requestIds", JSON.stringify(state.requestIds));
    setStoredValue("uav.operator.requestSignatures", JSON.stringify(state.requestSignatures));
    setStoredValue("uav.operator.eventSequence", String(state.eventSequence));
    if (state.recoveryNotice) setStoredValue("uav.operator.recoveryNotice", JSON.stringify(state.recoveryNotice));
    if (state.mission?.mission_id) setStoredValue("uav.operator.missionId", state.mission.mission_id);
  }

  function normalizeBase(value) {
    const trimmed = String(value || "").trim().replace(/\/+$/, "");
    if (!trimmed) throw new Error("任务 API 地址不能为空");
    return trimmed.endsWith("/api/v1") ? trimmed : `${trimmed}/api/v1`;
  }

  function apiError(status, body) {
    const error = new Error(body?.message || `任务 API 返回 HTTP ${status}`);
    error.status = status;
    error.code = body?.error || "http_error";
    error.payload = body;
    return error;
  }

  async function api(path, options = {}) {
    if (!state.token) throw new Error("请先输入操作令牌");
    const headers = { Authorization: `Bearer ${state.token}` };
    if (options.body !== undefined) {
      headers["Content-Type"] = "application/json";
      options = { ...options, body: JSON.stringify(options.body) };
    }
    let response;
    try {
      response = await fetch(`${state.apiBase}${path}`, { ...options, headers: { ...headers, ...(options.headers || {}) } });
    } catch (error) {
      throw new Error(`无法连接任务 API：${error.message}`);
    }
    const text = await response.text();
    let body = null;
    try { body = text ? JSON.parse(text) : null; } catch (_) { body = { message: text || "空响应" }; }
    if (!response.ok) throw apiError(response.status, body);
    return body;
  }

  function newRequestId(kind) {
    if (!state.requestIds[kind]) {
      state.requestIds[kind] = `${kind}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      persist();
    }
    return state.requestIds[kind];
  }

  function resetRequest(kind) {
    delete state.requestIds[kind];
    delete state.requestSignatures[kind];
    persist();
  }

  function formationRequestId(missionId, form, revision) {
    const key = `formation-${missionId}`;
    const signature = JSON.stringify({
      expected_revision: revision,
      formation: { shape: form.shape, spacing_m: form.spacing },
    });
    if (state.requestSignatures[key] !== signature) {
      state.requestIds[key] = `${key}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      state.requestSignatures[key] = signature;
      persist();
    }
    return state.requestIds[key];
  }

  function formatVector(value, digits = 2) {
    return Array.isArray(value) ? value.map((v) => Number(v).toFixed(digits)).join(", ") : "—";
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[character]);
  }

  function formatFormation(value) {
    if (!value) return "—";
    return `${value.shape || "未知"} / ${Number(value.spacing_m || 0).toFixed(1)} m`;
  }

  function formatAge(receivedAt) {
    if (!Number.isFinite(Number(receivedAt))) return "未知";
    const ageMs = Math.max(0, Date.now() - Number(receivedAt));
    return ageMs < 1000 ? `${ageMs} ms 前` : `${(ageMs / 1000).toFixed(1)} s 前`;
  }

  function readableState(value) {
    return ({ READY: "待启动", STARTING: "启动中", TAKING_OFF: "起飞中", RUNNING: "飞行中", STOPPING: "停止中", LANDING: "降落中", COMPLETED: "已完成", STOPPED: "已停止", FAILED: "失败", INTERRUPTED: "已中断" })[value] || value || "未知";
  }

  function setButton(id, enabled) { $(id).disabled = !enabled; }

  function updateMissionEditingState() {
    const locked = ACTIVE_MISSION_STATES.has(state.mission?.state);
    ["goal-x", "goal-y", "goal-z", "formation-shape", "formation-spacing"].forEach((id) => {
      $(id).disabled = locked;
    });
    setButton("validate-mission", Boolean(state.capabilities) && !locked);
    setButton("create-mission", Boolean(state.capabilities) && !locked);
  }

  function formValues() {
    const caps = state.capabilities;
    const zDefault = caps?.cruise_z_ned_m ?? -2;
    const x = Number($("goal-x").value);
    const y = Number($("goal-y").value);
    const z = $("goal-z").value === "" ? Number(zDefault) : Number($("goal-z").value);
    const spacing = Number($("formation-spacing").value);
    if (![x, y, z, spacing].every(Number.isFinite)) throw new Error("终点坐标和队形间距必须是有效数字");
    return { x, y, z, spacing, shape: $("formation-shape").value };
  }

  function missionBody(kind = "create") {
    if (!state.capabilities || !state.map) throw new Error("请先读取能力和地图");
    const form = formValues();
    const body = {
      request_id: newRequestId(kind),
      map_id: state.map.map_id,
      map_revision: state.map.map_revision,
      vehicle_names: state.capabilities.vehicle_names,
      leader: state.capabilities.leader,
      leader_goal_ned_m: [form.x, form.y, form.z],
      formation: { shape: form.shape, spacing_m: form.spacing },
    };
    return body;
  }

  function bodySignature(body) {
    return JSON.stringify({ ...body, request_id: "" });
  }

  function renderCapabilities() {
    const caps = state.capabilities;
    if (!caps) return;
    $("fleet-summary").textContent = `${caps.vehicle_names.length} 机 / ${caps.leader} 领航`;
    $("fleet-fields").innerHTML = caps.vehicle_names.map((name) => `<span class="fleet-chip">${escapeHtml(name)}${name === caps.leader ? " · 领航" : " · 跟随"}</span>`).join("");
    $("formation-spacing").min = String(caps.spacing_m.min);
    $("formation-spacing").max = String(caps.spacing_m.max);
    $("formation-spacing").value = Math.max(caps.spacing_m.min, Math.min(caps.spacing_m.max, Number($("formation-spacing").value) || caps.spacing_m.min));
    const cruiseZ = Number(caps.cruise_z_ned_m);
    $("goal-z").value = Number.isFinite(cruiseZ) ? cruiseZ.toFixed(1) : "";
    const altitudeEditable = Boolean(caps.supports?.speed_altitude_edit);
    $("goal-z").readOnly = !altitudeEditable;
    $("goal-z-note").textContent = altitudeEditable
      ? "当前后端允许修改 Z；直接点击地图会填写 X/Y，起点由后端预检返回 actual_start。"
      : `Z 由后端能力接口固定为巡航高度 ${Number.isFinite(cruiseZ) ? cruiseZ.toFixed(1) : "—"} m；当前版本不支持自由修改。直接点击地图会填写 X/Y。`;
    $("formation-limit").textContent = `安全距离 ${Number(caps.safety_distance_m).toFixed(1)} m；间距 ${Number(caps.spacing_m.min).toFixed(1)}–${Number(caps.spacing_m.max).toFixed(1)} m；巡航 Z=${Number.isFinite(cruiseZ) ? cruiseZ.toFixed(1) : "—"} m`;
    const supports = caps.supports || {};
    $("capabilities-detail").innerHTML = `
      <div class="capability-row"><span>飞行开关</span><strong>${caps.flight_enabled ? "已开启" : "未开启 --allow-flight"}</strong></div>
      <div class="capability-row"><span>机群与领航机</span><strong>${caps.vehicle_names.join(", ")} / ${caps.leader}</strong></div>
      <div class="capability-row"><span>队形</span><strong>${(caps.shapes || []).join(" / ")}</strong></div>
      <div class="capability-row"><span>可用功能</span><span class="capability-list">${Object.entries(supports).filter(([, v]) => v).map(([k]) => `<span class="capability-chip">${k}</span>`).join("") || "—"}</span></div>
      <div class="capability-row"><span>暂不支持</span><strong>${Object.entries(supports).filter(([, v]) => !v).map(([k]) => k).join(", ") || "—"}</strong></div>`;
    setButton("validate-mission", true);
    setButton("create-mission", true);
    $("recovery-check").hidden = false;
    setButton("recovery-check", !ACTIVE_MISSION_STATES.has(state.mission?.state));
    updateMissionEditingState();
  }

  function renderMap() {
    const mapNode = $("map");
    mapNode.querySelectorAll(".map-marker, .trail-line, .map-empty").forEach((node) => node.remove());
    if (!state.map) {
      const empty = document.createElement("div"); empty.className = "map-empty"; empty.textContent = "连接任务 API 后读取真实地图"; mapNode.appendChild(empty); return;
    }
    const [dataMinX, dataMaxX, dataMinY, dataMaxY] = state.map.bounds_xy_m;
    const dataWidth = Math.max(0.001, dataMaxX - dataMinX);
    const dataHeight = Math.max(0.001, dataMaxY - dataMinY);
    const aspect = mapNode.clientWidth > 0 && mapNode.clientHeight > 0 ? mapNode.clientWidth / mapNode.clientHeight : 1;
    const centerX = (dataMinX + dataMaxX) / 2;
    const centerY = (dataMinY + dataMaxY) / 2;
    const viewWidth = Math.max(dataWidth, dataHeight * aspect) / state.mapZoom;
    const viewHeight = viewWidth / aspect;
    const minX = centerX - viewWidth / 2;
    const maxX = centerX + viewWidth / 2;
    const minY = centerY - viewHeight / 2;
    const maxY = centerY + viewHeight / 2;
    $("map-coverage").textContent = `数据覆盖：X ${dataMinX.toFixed(1)}–${dataMaxX.toFixed(1)} m / Y ${dataMinY.toFixed(1)}–${dataMaxY.toFixed(1)} m · 视图 ${state.mapZoom.toFixed(2)}× · ${state.map.coverage || "partial"}`;
    function point(value) { return [((value[0] - minX) / (maxX - minX)) * 100, ((maxY - value[1]) / (maxY - minY)) * 100]; }
    function place(node, value) { const [left, top] = point(value); node.style.left = `${left}%`; node.style.top = `${top}%`; }
    for (const obstacle of state.map.obstacles || []) {
      const [loX, loY] = obstacle.min_ned_m; const [hiX, hiY] = obstacle.max_ned_m;
      const node = document.createElement("div"); node.className = "map-marker obstacle-marker";
      node.style.left = `${((loX + hiX) / 2 - minX) / viewWidth * 100}%`;
      node.style.top = `${(maxY - (loY + hiY) / 2) / viewHeight * 100}%`;
      node.style.width = `${(hiX - loX) / viewWidth * 100}%`; node.style.height = `${(hiY - loY) / viewHeight * 100}%`; node.title = `${obstacle.id || "obstacle"} · measured render mesh AABB`; mapNode.appendChild(node);
    }
    const starts = state.validation?.actual_start || {};
    for (const [name, item] of Object.entries(starts)) {
      if (!Array.isArray(item.position_ned_m)) continue;
      const node = document.createElement("div"); node.className = "map-marker start-marker"; place(node, item.position_ned_m); node.title = `${name} 实测起点`; mapNode.appendChild(node);
    }
    const preview = state.validation?.terminal_slots_preview || state.mission?.terminal_slots_preview || {};
    for (const [name, value] of Object.entries(preview)) {
      const node = document.createElement("div"); node.className = "map-marker slot-marker"; place(node, value); node.innerHTML = `<span class="marker-label">槽位 ${escapeHtml(name)}</span>`; mapNode.appendChild(node);
    }
    // A terminal mission can retain its last runtime snapshot after Unreal
    // has been restarted.  Keep that history in the status cards, but never
    // draw stale slot targets as if they were current aircraft positions.
    const currentSlots = state.mission?.formation_runtime_stale ? {} : (state.mission?.formation_runtime?.slot_targets_ned_m || {});
    for (const [name, value] of Object.entries(currentSlots)) {
      const node = document.createElement("div"); node.className = "map-marker slot-marker current-slot"; place(node, value); node.innerHTML = `<span class="marker-label">实际目标 ${escapeHtml(name)}</span>`; node.title = "B/A 当前回报的编队或临时跟随目标"; mapNode.appendChild(node);
    }
    const formGoal = ["goal-x", "goal-y", "goal-z"].map((id) => Number($(id).value));
    const formGoalIsValid = formGoal.every(Number.isFinite);
    const missionGoal = state.mission?.config?.leader_goal_ned_m;
    // Before a mission is running, the marker follows the editable form. This
    // prevents an old READY/FAILED mission from pinning the marker to its goal
    // after the operator selects a new point.
    const goal = state.selectedGoal
      || state.validation?.config?.leader_goal_ned_m
      || (formGoalIsValid && !ACTIVE_MISSION_STATES.has(state.mission?.state) ? formGoal : missionGoal);
    if (goal) {
      const node = document.createElement("div");
      // The goal marker is a visual guide.  It must not block selecting a new
      // point on the map, including clicking directly on the old goal.
      node.className = "map-marker goal-marker";
      node.style.pointerEvents = "none";
      place(node, goal);
      node.title = "当前领航机任务终点 · 点击地图可重新选择";
      mapNode.appendChild(node);
    }
    const telemetryLive = state.telemetryFrame && state.telemetryFrame.status !== "OFFLINE" && Object.keys(state.telemetryFrame.drones || {}).length;
    const runtimeLive = state.runtimeFrame && state.runtimeFrame.status !== "OFFLINE" && !state.runtimeFrame.stale && Object.keys(state.runtimeFrame.drones || {}).length;
    const liveFrame = telemetryLive ? state.telemetryFrame : (runtimeLive ? state.runtimeFrame : null);
    const drones = liveFrame?.drones || {};
    const names = Object.keys(drones);
    for (const [name, drone] of Object.entries(drones)) {
      if (!Array.isArray(drone.position_ned_m)) continue;
      const node = document.createElement("button"); node.type = "button"; node.className = `map-marker drone-marker ${drone.valid ? "" : "invalid"} ${state.selectedVehicle === name ? "selected" : ""}`; node.style.background = COLORS[names.indexOf(name) % COLORS.length]; place(node, drone.position_ned_m); node.title = `${name}：定位地图`; node.innerHTML = `<span class="marker-label">${escapeHtml(name)}</span>`; node.addEventListener("click", (event) => { event.stopPropagation(); state.selectedVehicle = name; $("map-coordinates").textContent = `${name} 当前位置：${formatVector(drone.position_ned_m)} NED · 数据${formatAge(drone.received_at_unix_ms)}`; renderMap(); }); mapNode.appendChild(node);
    }
    renderTrails(mapNode, point);
  }

  function renderTrails(mapNode, point) {
    for (const [name, points] of state.trails) {
      for (let i = 1; i < points.length; i += 1) {
        const a = point(points[i - 1]); const b = point(points[i]); const line = document.createElement("div"); line.className = "trail-line"; line.style.background = COLORS[Array.from(state.trails.keys()).indexOf(name) % COLORS.length];
        const dx = (b[0] - a[0]) * mapNode.clientWidth / 100; const dy = (b[1] - a[1]) * mapNode.clientHeight / 100; line.style.left = `${a[0]}%`; line.style.top = `${a[1]}%`; line.style.width = `${Math.sqrt(dx * dx + dy * dy)}px`; line.style.transform = `rotate(${Math.atan2(dy, dx)}rad)`; mapNode.appendChild(line);
      }
    }
  }

  function renderPreflight(result) {
    state.validation = result;
    state.validationSignature = result?.config ? bodySignature({ ...result.config, request_id: "" }) : "";
    const node = $("preflight-result");
    node.className = "preflight-result ok";
    const starts = Object.entries(result.actual_start || {}).map(([name, value]) => `<div class="preview-item"><strong>${escapeHtml(name)} 实测起点</strong>${formatVector(value.position_ned_m)}<br><span class="muted">yaw ${Number(value.yaw_rad || 0).toFixed(2)} rad</span></div>`).join("");
    const slots = Object.entries(result.terminal_slots_preview || {}).map(([name, value]) => `<div class="preview-item"><strong>${escapeHtml(name)} 名义槽位</strong>${formatVector(value)}</div>`).join("");
    node.innerHTML = `<strong>预检通过：只表示任务配置可提交，尚未启动飞行。</strong><div class="preview-grid">${starts}${slots}</div><p class="muted">${escapeHtml(result.preview_semantics || "名义槽位预览不等于无碰撞路径")}</p><p class="muted">${(result.warnings || []).map(escapeHtml).join(" ")}</p>`;
    $("preflight-status").textContent = "预检通过";
    renderMap();
  }

  function renderPreflightError(error) {
    const node = $("preflight-result"); node.className = "preflight-result error"; node.textContent = `${error.code || "请求失败"}：${error.message}`; $("preflight-status").textContent = "预检未通过"; state.validation = null; renderMap();
  }

  function renderMission(mission) {
    if (!mission) return;
    if (state.mission?.mission_id !== mission.mission_id) state.selectedGoal = null;
    state.mission = mission;
    state.savedMissionId = mission.mission_id;
    persist();
    if (state.formMissionId !== mission.mission_id) {
      const goal = mission.config?.leader_goal_ned_m || [];
      if (goal.length >= 2) {
        $("goal-x").value = Number(goal[0]).toFixed(2);
        $("goal-y").value = Number(goal[1]).toFixed(2);
      }
      if (goal.length >= 3 && state.capabilities?.supports?.speed_altitude_edit) {
        $("goal-z").value = Number(goal[2]).toFixed(2);
      }
      const formation = mission.config?.formation || mission.requested_formation || {};
      if (formation.shape) $("formation-shape").value = formation.shape;
      if (Number.isFinite(Number(formation.spacing_m))) $("formation-spacing").value = Number(formation.spacing_m);
      state.formMissionId = mission.mission_id;
    }
    if (TERMINAL_MISSION_STATES.has(mission.state)) {
      resetRequest("create");
      $("create-mission").textContent = "创建新任务";
    } else {
      $("create-mission").textContent = "创建任务";
    }
    $("mission-state").textContent = readableState(mission.state);
    $("mission-id").textContent = mission.mission_id ? mission.mission_id.slice(0, 12) : "—";
    $("mission-revision").textContent = mission.revision ?? "—";
    $("requested-formation").textContent = formatFormation(mission.requested_formation);
    $("applied-formation").textContent = formatFormation(mission.applied_formation);
    const runtimeFormation = mission.formation_runtime;
    $("effective-formation").textContent = mission.formation_runtime_stale || !runtimeFormation ? "未知" : (runtimeFormation.effective_mode || "未知");
    const effectiveMode = runtimeFormation?.effective_mode;
    $("formation-settled").textContent = mission.formation_runtime_stale || !runtimeFormation ? "未知" : (effectiveMode === "nominal" ? "已回报 nominal" : `未恢复（${effectiveMode || "未知"}）`);
    $("formation-status-reason").textContent = runtimeFormation?.reason || (mission.formation_runtime_stale ? "实际编队状态未接入或已过期。" : "等待真实编队状态回报。");
    if (mission.reason) requestMessage(`后端：${mission.reason}`, "error");
    if (mission.manual_intervention_required) {
      requestMessage("历史任务清理未确认，请先点击“安全恢复检查”，通过后才能再次启动。", "error");
    }
    renderRuntime(mission.runtime);
    renderMap();
    const active = ACTIVE_MISSION_STATES.has(mission.state);
    setButton("start-mission", mission.state === "READY" && Boolean(state.capabilities?.flight_enabled) && !mission.manual_intervention_required);
    setButton("stop-mission", active || mission.state === "INTERRUPTED");
    $("recovery-check").hidden = false;
    setButton("recovery-check", !active);
    updateMissionEditingState();
    const supports = state.capabilities?.supports || {};
    setButton("apply-formation", Boolean(mission.mission_id) && (mission.state === "READY" || (mission.state === "RUNNING" && supports.runtime_formation === "requires_B_heartbeat")));
  }

  function renderOperation(operation) {
    if (!operation) return;
    state.operation = operation;
    $("operation-state").textContent = `${operation.kind || "操作"}：${operation.state || "未知"}`;
    const log = $("operation-log");
    const item = document.createElement("div"); item.className = "operation-item"; item.innerHTML = `<strong>${escapeHtml(operation.kind || "operation")}</strong><span>${escapeHtml(operation.state || "—")}${operation.reason ? ` · ${escapeHtml(operation.reason)}` : ""}</span>`; log.replaceChildren(item);
    if (TERMINAL_OPERATION_STATES.has(operation.state)) requestMessage(`后端已回报：${operation.state}${operation.reason ? `（${operation.reason}）` : ""}`, operation.state === "FAILED" || operation.state === "REJECTED" ? "error" : "success");
  }

  function renderRuntime(runtime) {
    if (!runtime || typeof runtime !== "object") return;
    const frame = { type: "telemetry", timestamp: runtime.timestamp_unix_ms, status: runtime.stale ? "PARTIAL_STATE" : "NORMAL", source: "operator-api/runtime", sequence_id: runtime.snapshot_id || "—", stream_id: runtime.stream_id || "runtime", drones: {}, coordination_state: "未知", snapshot: { collection_complete: runtime.collection_complete, collection_duration_s: null, max_receive_skew_s: null }, diagnostics: { received_at_unix_ms: runtime.timestamp_unix_ms, transport_latency_ms: null } };
    const coordination = runtime.coordination || {};
    const states = Object.values(coordination).map((item) => item?.state).filter(Boolean);
    if (states.length) frame.coordination_state = new Set(states).size === 1 ? states[0] : "MIXED";
    for (const [name, value] of Object.entries(runtime.vehicles || {})) {
      const stateValue = value.state || value;
      const status = coordination[name] || {};
      frame.drones[name] = { vehicle_name: name, valid: value.valid !== false && !value.validity_reason, validity_reason: value.validity_reason || "", position_ned_m: stateValue.position_ned_m || null, linear_velocity_ned_mps: stateValue.linear_velocity_ned_mps || stateValue.velocity_ned_mps || null, sim_timestamp_ns: stateValue.timestamp || stateValue.sim_timestamp_ns || null, received_at_unix_ms: runtime.timestamp_unix_ms, coordination_state: status.state || "UNKNOWN", coordination_reason: status.reason || "", active_command_id: status.active_command_id || null };
    }
    state.runtimeFrame = { ...frame, stale: Boolean(runtime.stale) };
    if (!state.telemetryFrame || state.telemetryFrame.status === "OFFLINE" || !Object.keys(state.telemetryFrame.drones || {}).length) renderTelemetry(frame, true);
  }

  function renderTelemetry(frame, fromRuntime = false) {
    if (!frame) return;
    if (!fromRuntime) state.telemetryFrame = frame;
    if (frame.status === "OFFLINE") {
      $("data-status").textContent = "遥测离线";
      if (!fromRuntime) renderMap();
      return;
    }
    const frameFresh = Number.isFinite(Number(frame.timestamp)) && Date.now() - Number(frame.timestamp) <= 2000;
    $("data-status").textContent = !frameFresh ? "数据已过期" : (frame.status === "NORMAL" ? "状态正常" : "部分状态");
    $("telemetry-source").textContent = `来源：${frame.source || "—"}`;
    $("sequence").textContent = frame.sequence_id ?? "—";
    const snapshot = frame.snapshot || {}; const diagnostics = frame.diagnostics || {};
    $("collection-complete").textContent = snapshot.collection_complete == null ? "—" : (snapshot.collection_complete ? "是" : "否");
    $("collection-duration").textContent = snapshot.collection_duration_s == null ? "—" : `${Number(snapshot.collection_duration_s).toFixed(3)} s`;
    $("max-skew").textContent = snapshot.max_receive_skew_s == null ? "—" : `${Number(snapshot.max_receive_skew_s).toFixed(3)} s`;
    $("transport-latency").textContent = diagnostics.transport_latency_ms == null ? "—" : `${diagnostics.transport_latency_ms} ms`;
    const drones = frame.drones || {};
    if (state.lastStream !== frame.stream_id) { state.trails.clear(); state.lastStream = frame.stream_id; }
    for (const [name, drone] of Object.entries(drones)) { if (Array.isArray(drone.position_ned_m)) { const points = state.trails.get(name) || []; points.push(drone.position_ned_m); if (points.length > 120) points.shift(); state.trails.set(name, points); } }
    const list = $("drone-list"); const entries = Object.entries(drones); list.replaceChildren();
    if (!entries.length) { list.innerHTML = '<div class="empty-state">还没有收到无人机状态</div>'; } else for (const [name, drone] of entries) {
      const stale = Number.isFinite(Number(drone.received_at_unix_ms)) && Date.now() - Number(drone.received_at_unix_ms) > 2000;
      const card = document.createElement("article"); card.className = `drone-card ${state.selectedVehicle === name ? "selected" : ""}`; card.title = "点击在地图上定位"; card.addEventListener("click", () => { state.selectedVehicle = name; if (Array.isArray(drone.position_ned_m)) $("map-coordinates").textContent = `${name} 当前位置：${formatVector(drone.position_ned_m)} NED · 数据${formatAge(drone.received_at_unix_ms)}`; renderMap(); renderTelemetry(frame, fromRuntime); });
      const collectedAt = drone.received_at_unix_ms ? new Date(drone.received_at_unix_ms).toLocaleTimeString() : (drone.state_received_monotonic_seconds == null ? "—" : `${Number(drone.state_received_monotonic_seconds).toFixed(2)} s`);
      const badgeText = stale ? "过期" : (drone.valid ? "有效" : "失效");
      const reasonLines = [
        drone.active_command_id ? `当前命令：${drone.active_command_id}` : "",
        drone.coordination_reason || "",
        drone.validity_reason || "",
      ].filter(Boolean).slice(0, 2);
      card.innerHTML = `<div class="drone-card-head"><span class="drone-name">${escapeHtml(name)}</span><span class="badge ${stale || !drone.valid ? "invalid" : "valid"}">${badgeText}</span></div><div class="drone-values"><div>位置 NED<strong>${formatVector(drone.position_ned_m)}</strong></div><div>速度 NED<strong>${formatVector(drone.linear_velocity_ned_mps)} m/s</strong></div><div>仿真时间<strong>${drone.sim_timestamp_ns == null ? "—" : escapeHtml(`${drone.sim_timestamp_ns} ns`)}</strong></div><div>采集时间<strong>${escapeHtml(collectedAt)}</strong></div><div>数据时效<strong>${escapeHtml(formatAge(drone.received_at_unix_ms))}</strong></div><div>协调/避障<strong>${escapeHtml(drone.coordination_state || "未知")}</strong></div></div><div class="drone-reasons">${reasonLines.map((line) => `<p class="reason" title="${escapeHtml(line)}">${escapeHtml(line)}</p>`).join("")}</div>`;
      list.appendChild(card);
    }
    renderMap(); renderCameras(state.capabilities?.vehicle_names || Object.keys(drones));
  }

  function cameraCardMarkup(name, view, base) {
    const safeName = escapeHtml(name); const encodedName = encodeURIComponent(name);
    const viewLabel = view === "first" ? "第一视角 · 机载前向" : "第三视角 · 后上方跟随";
    return `<div class="camera-card available" data-camera-name="${safeName}" data-camera-view="${view}"><span>${viewLabel}</span><div class="camera-frame-shell"><div class="camera-placeholder" data-camera-placeholder>正在连接相机服务…</div><img hidden loading="lazy" decoding="async" alt="${safeName} ${viewLabel}" data-camera-title="${safeName} · ${viewLabel}" data-camera-frame="${base}/frame/${encodedName}/${view}" /></div><span class="camera-time" data-camera-time>等待画面</span></div>`;
  }

  function renderCameras(names = []) {
    const list = $("camera-list"); const base = state.cameraBase.replace(/\/+$/, "");
    const vehicleNames = [...new Set(names.length ? names : (state.capabilities?.vehicle_names || state.savedVehicleNames || []))];
    const available = vehicleNames.flatMap((vehicle) => ["first", "third"].map((view) => ({ vehicle, view })));
    const availableKeys = new Set(available.map(({ vehicle, view }) => `${vehicle}:${view}`));
    let selected = (Array.isArray(state.cameraSelection.cameras) ? state.cameraSelection.cameras : [])
      .filter((item) => item && availableKeys.has(`${item.vehicle}:${item.view}`))
      .map((item) => ({ vehicle: item.vehicle, view: item.view }));
    const seen = new Set();
    selected = selected.filter((item) => {
      const key = `${item.vehicle}:${item.view}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    if (!selected.length && available.length) selected = [available[0]];
    state.cameraSelection.cameras = selected;
    if (![1, 5, 10, 20].includes(state.cameraFps)) state.cameraFps = 5;
    persist();
    const selectionKey = selected.map(({ vehicle, view }) => `${vehicle}:${view}`).join(",");
    const key = `${base}|${vehicleNames.join(",")}|${selectionKey}|${state.cameraFps}`;
    if (key === state.cameraNamesKey) return;
    state.cameraNamesKey = key;
    const fps = $("camera-fps"); fps.value = String(state.cameraFps);
    const selectedKeys = new Set(selectionKey.split(",").filter(Boolean));
    $("camera-options").replaceChildren();
    for (const { vehicle, view } of available) {
      const label = document.createElement("label");
      label.className = "camera-option";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.dataset.cameraVehicle = vehicle;
      checkbox.dataset.cameraView = view;
      checkbox.checked = selectedKeys.has(`${vehicle}:${view}`);
      const text = document.createElement("span");
      text.textContent = `${vehicle} · ${view === "first" ? "第一视角" : "第三视角"}`;
      label.append(checkbox, text);
      $("camera-options").appendChild(label);
    }
    if (!base) { list.innerHTML = '<div class="camera-card unavailable">请配置独立图像服务地址；A 任务 API 不提供视频流。</div>'; $("camera-service-status").textContent = "未配置图像服务"; return; }
    if (!vehicleNames.length) { list.innerHTML = '<div class="camera-card unavailable">能力接口尚未返回无人机列表。</div>'; $("camera-service-status").textContent = "等待无人机列表"; return; }
    list.querySelectorAll("img[data-object-url]").forEach((image) => {
      URL.revokeObjectURL(image.dataset.objectUrl);
    });
    list.innerHTML = `<article class="camera-pair selected-camera camera-count-${selected.length}"><h3><span>已选 ${selected.length} 个摄像头</span><span class="muted">只采集勾选项</span></h3><div class="camera-views">${selected.map(({ vehicle, view }) => cameraCardMarkup(vehicle, view, base)).join("")}</div></article>`;
    $("camera-refresh-label").textContent = `${selected.length} 路按需采集 · ${state.cameraFps} FPS`;
    $("camera-service-status").textContent = state.cameraSelectionBusy ? "正在切换摄像头…" : `已选择 ${selected.length} 路 · ${state.cameraFps} FPS`;
    void activateCameraSelection();
    void refreshCameraStatus();
    void refreshCameraFrames();
  }

  async function refreshCameraFrames() {
    if (state.cameraRefreshBusy || state.cameraLightboxOpen || document.hidden) return;
    const images = [...document.querySelectorAll("img[data-camera-frame]")];
    if (!images.length) return;
    const controller = new AbortController();
    state.cameraFrameAbort = controller;
    state.cameraRefreshBusy = true;
    try {
      await Promise.all(images.map(async (image) => {
        try {
          const response = await fetch(`${image.dataset.cameraFrame}?t=${Date.now()}`, { cache: "no-store", signal: controller.signal });
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const blob = await response.blob();
          if (state.cameraLightboxOpen || controller.signal.aborted) return;
          const nextUrl = URL.createObjectURL(blob);
          const previousUrl = image.dataset.objectUrl;
          image.src = nextUrl;
          image.hidden = false;
          const placeholder = image.closest(".camera-frame-shell")?.querySelector("[data-camera-placeholder]");
          if (placeholder) placeholder.hidden = true;
          image.dataset.objectUrl = nextUrl;
          if (previousUrl) URL.revokeObjectURL(previousUrl);
        } catch (error) {
          if (error.name === "AbortError") return;
          const placeholder = image.closest(".camera-frame-shell")?.querySelector("[data-camera-placeholder]");
          if (!image.dataset.objectUrl && placeholder) { placeholder.hidden = false; placeholder.textContent = "正在等待相机画面…"; }
        }
      }));
    } finally {
      if (state.cameraFrameAbort === controller) state.cameraFrameAbort = null;
      state.cameraRefreshBusy = false;
    }
  }

  async function refreshCameraStatusLegacy() {
    const base = state.cameraBase.replace(/\/+$/, ""); if (!base) return;
    try {
      const response = await fetch(`${base}/api/cameras`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const cameras = await response.json();
      for (const [name, views] of Object.entries(cameras)) for (const view of ["first", "third"]) {
        const key = `${name}-${view}`.replace(/[&<>"']/g, "_"); const node = document.querySelector(`[data-camera-time="${key}"]`); const item = views?.[view];
        if (!node) continue;
        node.textContent = item?.captured_at_unix_ms ? `${item.status === "online" ? "画面" : "断流"}：${new Date(item.captured_at_unix_ms).toLocaleTimeString()}${item.error ? ` · ${item.error}` : ""}` : (item?.status === "offline" ? `离线 · ${item.error || "无画面"}` : "等待画面");
      }
    } catch (error) {
      document.querySelectorAll(".camera-time").forEach((node) => { node.textContent = `图像服务离线：${error.message}`; });
    }
  }

  async function refreshCameraStatus() {
    const base = state.cameraBase.replace(/\/+$/, ""); if (!base) return;
    try {
      const response = await fetch(`${base}/api/cameras`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const cameras = await response.json();
      state.cameraStatus = cameras;
      state.cameraServiceOnline = true;
      const activeFps = Number(response.headers.get("X-Camera-Fps") || cameras?._meta?.fps);
      const fpsLabel = [1, 5, 10, 20].includes(activeFps) ? ` · ${activeFps} FPS` : "";
      const cards = document.querySelectorAll(".camera-card[data-camera-name]");
      cards.forEach((card) => {
        const item = cameras?.[card.dataset.cameraName]?.[card.dataset.cameraView];
        const node = card.querySelector("[data-camera-time]");
        const capturedAt = Number(item?.captured_at_unix_ms);
        const ageMs = Number.isFinite(capturedAt) ? Math.max(0, Date.now() - capturedAt) : Infinity;
        const stale = ageMs > 3000;
        if (!node) return;
        if (item?.status === "offline") node.textContent = `断流 · ${item.error || "相机采集失败"}`;
        else if (item?.active && !item?.captured_at_unix_ms) node.textContent = "采集中 · 等待第一帧";
        else if (item?.active && stale) node.textContent = `画面过期 · ${formatAge(capturedAt)}`;
        else if (item?.active && capturedAt) node.textContent = `在线 · ${new Date(capturedAt).toLocaleTimeString()} · ${formatAge(capturedAt)}`;
        else node.textContent = "未采集 · 选择后启动";
      });
      const selected = state.cameraSelection.cameras || [];
      const activeCount = selected.filter(({ vehicle, view }) => cameras?.[vehicle]?.[view]?.active).length;
      const offlineCount = selected.filter(({ vehicle, view }) => cameras?.[vehicle]?.[view]?.status === "offline").length;
      const statusText = offlineCount ? `${offlineCount} 路断流` : `${activeCount}/${selected.length} 路采集中`;
      $("camera-service-status").textContent = `${statusText}${fpsLabel}`;
    } catch (error) {
      state.cameraServiceOnline = false;
      document.querySelectorAll(".camera-time").forEach((node) => { node.textContent = `图像服务离线 · ${error.message}`; });
      document.querySelectorAll("[data-camera-placeholder]").forEach((node) => { node.hidden = false; node.textContent = "暂时无法连接相机服务，请检查服务是否启动"; });
      $("camera-service-status").textContent = "图像服务离线";
    }
  }

  async function activateCameraSelection() {
    const base = state.cameraBase.replace(/\/+$/, "");
    const cameras = state.cameraSelection.cameras || [];
    if (!base || !cameras.length) return;
    state.cameraSelectionBusy = true;
    try {
      const response = await fetch(`${base}/api/cameras/active`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "custom", cameras, fps: state.cameraFps }),
      });
      const text = await response.text();
      let body = null; try { body = text ? JSON.parse(text) : null; } catch (_) { body = null; }
      if (!response.ok) throw new Error(body?.error || body?.message || `HTTP ${response.status}`);
      state.cameraSelectionBusy = false;
      $("camera-service-status").textContent = `已启动 ${cameras.length} 路采集 · ${state.cameraFps} FPS`;
      await refreshCameraStatus();
      await refreshCameraFrames();
    } catch (error) {
      state.cameraSelectionBusy = false;
      $("camera-service-status").textContent = `视角切换失败 · ${error.message}`;
      document.querySelectorAll(".camera-time").forEach((node) => { node.textContent = "尚未开始采集，请确认相机服务已启动"; });
      document.querySelectorAll("[data-camera-placeholder]").forEach((node) => { node.hidden = false; node.textContent = "暂时无法加载画面，请检查相机服务是否启动"; });
    }
  }

  function closeCameraLightbox() {
    const lightbox = $("camera-lightbox");
    state.cameraLightboxOpen = false;
    lightbox.hidden = true;
    const preview = $("camera-lightbox-image");
    preview.style.display = "none";
    preview.removeAttribute("src");
    $("camera-lightbox-message").textContent = "正在加载画面…";
  }

  $("camera-options").addEventListener("change", (event) => {
    const checkbox = event.target.closest("input[data-camera-vehicle]");
    if (!checkbox) return;
    const checked = [...$("camera-options").querySelectorAll("input[data-camera-vehicle]:checked")];
    if (!checked.length) {
      checkbox.checked = true;
      $("camera-service-status").textContent = "至少选择一个摄像头";
      return;
    }
    state.cameraSelection.cameras = checked.map((item) => ({ vehicle: item.dataset.cameraVehicle, view: item.dataset.cameraView }));
    state.cameraNamesKey = "";
    persist();
    renderCameras(state.capabilities?.vehicle_names || state.savedVehicleNames);
  });
  $("camera-fps").addEventListener("change", (event) => {
    state.cameraFps = [1, 5, 10, 20].includes(Number(event.target.value)) ? Number(event.target.value) : 5;
    state.cameraNamesKey = "";
    persist();
    renderCameras(state.capabilities?.vehicle_names || state.savedVehicleNames);
  });
  $("camera-list").addEventListener("click", (event) => {
    const image = event.target.closest("img[data-camera-title]");
    if (!image) return;
    if (state.cameraFrameAbort) state.cameraFrameAbort.abort();
    state.cameraLightboxOpen = true;
    $("camera-lightbox-title").textContent = image.dataset.cameraTitle;
    const preview = $("camera-lightbox-image");
    preview.style.display = "none";
    preview.src = image.src;
    preview.onload = () => { preview.style.display = "block"; $("camera-lightbox-message").textContent = "单帧预览 · 点击右上角关闭"; };
    preview.onerror = () => { preview.style.display = "none"; $("camera-lightbox-message").textContent = "图像服务暂时没有可用画面，请确认相机服务正在运行"; };
    $("camera-lightbox").hidden = false;
    $("camera-lightbox-close").focus();
  });
  $("camera-lightbox-close").addEventListener("click", closeCameraLightbox);
  $("camera-lightbox").addEventListener("click", (event) => { if (event.target === $("camera-lightbox")) closeCameraLightbox(); });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape" && !$("camera-lightbox").hidden) closeCameraLightbox(); });

  async function loadBackend() {
    if (state.backendLoading) return;
    state.backendLoading = true;
    try {
      state.telemetryDisconnected = false;
      state.apiBase = normalizeBase($("api-base").value); state.token = $("operator-token").value.trim(); state.cameraBase = $("camera-base").value.trim(); persist();
      const [caps, map] = await Promise.all([api("/capabilities"), api("/maps/current")]);
      state.capabilities = caps; state.map = map; state.savedVehicleNames = [...(caps.vehicle_names || [])]; persist(); renderCapabilities(); renderMap(); renderCameras();
      setPill("operator-connection", true, "任务 API 已连接"); message("能力和真实地图已读取。", "success");
      if (state.savedMissionId) {
        await loadMission(state.savedMissionId);
        startMissionPolling();
        await recoverPendingOperation(state.savedMissionId);
      }
      startEventLoop();
    } catch (error) {
      setPill("operator-connection", false, "任务 API 连接失败"); message(`${error.code || "连接失败"}：${error.message}`, "error");
    } finally {
      state.backendLoading = false;
    }
  }

  function disconnectOperator() {
    if (state.eventAbort) {
      state.eventAbort.abort();
      state.eventAbort = null;
    }
    if (state.missionTimer) {
      clearInterval(state.missionTimer);
      state.missionTimer = null;
    }
    if (state.operationTimer) {
      clearTimeout(state.operationTimer);
      state.operationTimer = null;
    }
    state.telemetryDisconnected = true;
    if (state.telemetrySocket) {
      state.telemetrySocket.close();
      state.telemetrySocket = null;
    }
    state.capabilities = null;
    state.map = null;
    state.mission = null;
    state.selectedGoal = null;
    state.formMissionId = "";
    state.validation = null;
    state.operation = null;
    state.telemetryFrame = null;
    state.runtimeFrame = null;
    setPill("operator-connection", false, "任务 API 未连接");
    setPill("telemetry-connection", false, "遥测未连接");
    $("mission-state").textContent = "未连接";
    $("fleet-summary").textContent = "—";
    $("mission-id").textContent = "—";
    $("mission-revision").textContent = "—";
    $("data-status").textContent = "等待数据";
    $("drone-list").innerHTML = '<div class="empty-state">已断开，重新连接后读取无人机状态</div>';
    $("camera-list").innerHTML = '<div class="camera-card unavailable">已断开任务连接；相机服务仍可独立运行。</div>';
    $("recovery-check").hidden = true;
    setButton("recovery-check", false);
    updateMissionEditingState();
    message("已断开网页与任务 API、遥测的连接。令牌仍保留在浏览器中，重新连接时无需再次输入。", "success");
  }

  async function loadMission(missionId) {
    if (!missionId) return;
    try { renderMission(await api(`/missions/${encodeURIComponent(missionId)}`)); } catch (error) { if (error.status === 404) { state.savedMissionId = ""; setStoredValue("uav.operator.missionId", ""); } else message(`查询任务失败：${error.message}`, "error"); }
  }

  async function recoverPendingOperation(missionId) {
    const requestIds = ["stop", "start", "formation"]
      .map((kind) => state.requestIds[`${kind}-${missionId}`])
      .filter(Boolean);
    for (const requestId of requestIds) {
      try {
        const operation = await api(`/operations/${encodeURIComponent(requestId)}`);
        renderOperation(operation);
        if (!TERMINAL_OPERATION_STATES.has(operation.state)) {
          requestMessage("已恢复未完成操作的状态查询，没有重新发送飞行指令。", "success");
          void pollOperation(requestId);
          return;
        }
      } catch (error) {
        if (error.status !== 404) requestMessage(`恢复操作状态失败：${error.message}`, "error");
      }
    }
  }

  function startMissionPolling() {
    if (state.missionTimer) clearInterval(state.missionTimer);
    state.missionTimer = setInterval(() => { if (state.mission?.mission_id) loadMission(state.mission.mission_id); }, 1000);
  }

  async function pollOperation(requestId) {
    if (state.operationTimer) clearInterval(state.operationTimer);
    const poll = async () => {
      try {
        const operation = await api(`/operations/${encodeURIComponent(requestId)}`); renderOperation(operation);
        if (TERMINAL_OPERATION_STATES.has(operation.state)) { clearInterval(state.operationTimer); state.operationTimer = null; if (state.mission?.mission_id) await loadMission(state.mission.mission_id); }
      } catch (error) { requestMessage(`查询操作结果失败：${error.message}`, "error"); }
    };
    await poll(); state.operationTimer = setInterval(poll, 700);
  }

  async function validateMission() {
    try {
      const body = missionBody("validate"); const result = await api("/missions/validate", { method: "POST", body }); renderPreflight(result); message("后端预检通过，可以创建任务。", "success");
    } catch (error) { renderPreflightError(error); message(`${error.code || "预检失败"}：${error.message}`, "error"); }
  }

  async function createMission() {
    try {
      const body = missionBody("create"); const result = await api("/missions", { method: "POST", body }); renderMission(result.mission); requestMessage("请求已接受，任务仍处于后端回报的状态。", "success"); message("任务已创建，尚未启动飞行。", "success"); startMissionPolling();
    } catch (error) { message(`${error.code || "创建失败"}：${error.message}`, "error"); }
  }

  async function startMission() {
    if (!state.mission) return;
    try {
      const requestId = newRequestId(`start-${state.mission.mission_id}`); const reply = await api(`/missions/${encodeURIComponent(state.mission.mission_id)}/start`, { method: "POST", body: { request_id: requestId, expected_revision: state.mission.revision } });
      renderOperation({ request_id: requestId, mission_id: state.mission.mission_id, kind: "start", state: "ACCEPTED" }); requestMessage("请求已接受，等待后端回报启动中、起飞中和飞行中状态。", "success"); await pollOperation(requestId);
    } catch (error) { requestMessage(`${error.code || "启动请求失败"}：${error.message}`, "error"); }
  }

  async function stopMission() {
    if (!state.mission) return;
    try {
      const requestId = newRequestId(`stop-${state.mission.mission_id}`); await api(`/missions/${encodeURIComponent(state.mission.mission_id)}/stop`, { method: "POST", body: { request_id: requestId } });
      renderOperation({ request_id: requestId, mission_id: state.mission.mission_id, kind: "stop", state: "ACCEPTED" }); requestMessage("请求已接受，等待后端确认降落、解锁和控制释放。", "success"); await pollOperation(requestId);
    } catch (error) { requestMessage(`${error.code || "停止请求失败"}：${error.message}`, "error"); }
  }

  async function verifyRecovery() {
    try {
      const requestId = newRequestId(`recovery-${state.mission?.mission_id || "all"}`);
      requestMessage("正在检查无人机是否已落地、解锁并释放控制权，请等待后端回报。", "success");
      const result = await api("/recovery/check", { method: "POST", body: { request_id: requestId } });
      const text = `已解除 ${result.cleared_mission_ids?.length || 0} 个旧任务的启动锁。`;
      setRecoveryNotice("success", "recovery_check_passed", text, "后端已完成只读检查，未发送飞行指令。");
      requestMessage(`安全恢复检查通过，${text}`, "success");
      if (state.savedMissionId) await loadMission(state.savedMissionId);
    } catch (error) {
      const code = error.code || "recovery_check_failed";
      setRecoveryNotice("error", code, error.message, "请先在 Unreal/AirSim 中确认所有无人机已落地、速度接近 0 且 API 控制已释放；前端不会绕过此安全条件。");
      requestMessage(`${code}：${error.message}`, "error");
    }
  }

  async function applyFormation() {
    if (!state.mission) return;
    try {
      const form = formValues(); const revision = state.mission.revision; const requestId = formationRequestId(state.mission.mission_id, form, revision); const reply = await api(`/missions/${encodeURIComponent(state.mission.mission_id)}/formation`, { method: "PUT", body: { request_id: requestId, expected_revision: revision, formation: { shape: form.shape, spacing_m: form.spacing } } });
      if (reply.operation) renderOperation(reply.operation); requestMessage("请求已接受。B 已采用配置与飞机物理恢复是两个状态。", "success"); if (reply.terminal_slots_preview) { state.validation = { ...(state.validation || {}), terminal_slots_preview: reply.terminal_slots_preview, config: { ...(state.validation?.config || state.mission.config), formation: form } }; renderMap(); }
      await loadMission(state.mission.mission_id); if (reply.operation?.request_id) await pollOperation(reply.operation.request_id);
    } catch (error) { requestMessage(`${error.code || "队形请求失败"}：${error.message}`, "error"); }
  }

  async function startEventLoop() {
    if (state.eventAbort) state.eventAbort.abort();
    const controller = new AbortController(); state.eventAbort = controller;
    while (!controller.signal.aborted) {
      try {
        const response = await fetch(`${state.apiBase}/events?after=${state.eventSequence}`, { headers: { Authorization: `Bearer ${state.token}` }, signal: controller.signal });
        if (!response.ok) throw apiError(response.status, await response.json().catch(() => null));
        const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = "";
        while (!controller.signal.aborted) {
          const part = await reader.read(); if (part.done) break; buffer += decoder.decode(part.value, { stream: true });
          const blocks = buffer.split("\n\n"); buffer = blocks.pop() || "";
          for (const block of blocks) { const line = block.split("\n").find((item) => item.startsWith("data: ")); if (!line) continue; const payload = JSON.parse(line.slice(6)); state.eventSequence = payload.sequence || state.eventSequence; persist(); if (payload.reset_required) await loadMission(state.savedMissionId); for (const event of payload.events || []) if (event.mission?.mission_id === state.savedMissionId) renderMission(event.mission); }
        }
      } catch (error) { if (!controller.signal.aborted) await new Promise((resolve) => setTimeout(resolve, 1000)); }
    }
  }

  function connectTelemetry() {
    if (state.telemetrySocket) state.telemetrySocket.close();
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${location.host}/ws/telemetry`); state.telemetrySocket = socket;
    socket.onopen = () => setPill("telemetry-connection", true, "遥测已连接");
    socket.onmessage = (event) => { try { const frame = JSON.parse(event.data); if (frame.source?.includes("demo")) $("demo-badge").classList.remove("hidden"); renderTelemetry(frame); } catch (_) { setPill("telemetry-connection", false, "收到无效遥测"); } };
    socket.onclose = () => { setPill("telemetry-connection", false, state.telemetryDisconnected ? "遥测未连接" : "遥测重连中"); if (!state.telemetryDisconnected) setTimeout(connectTelemetry, 1000); };
    socket.onerror = () => socket.close();
  }

  function resetFormRequests() { state.validation = null; state.validationSignature = ""; resetRequest("validate"); resetRequest("create"); $("preflight-status").textContent = "尚未预检"; $("preflight-result").className = "preflight-result empty-state"; $("preflight-result").textContent = "预检结果会显示实测起点、名义终点槽位和后端警告。"; updateMissionEditingState(); renderMap(); }

  $("connect-operator").addEventListener("click", loadBackend);
  $("disconnect-operator").addEventListener("click", disconnectOperator);
  $("operator-token").addEventListener("input", () => {
    state.token = $("operator-token").value.trim();
    persist();
  });
  ["goal-x", "goal-y", "goal-z", "formation-shape", "formation-spacing"].forEach((id) => $(id).addEventListener("input", () => {
    if (["goal-x", "goal-y", "goal-z"].includes(id)) state.selectedGoal = null;
    resetFormRequests();
  }));
  $("validate-mission").addEventListener("click", validateMission); $("create-mission").addEventListener("click", createMission); $("start-mission").addEventListener("click", startMission); $("stop-mission").addEventListener("click", stopMission); $("recovery-check").addEventListener("click", verifyRecovery); $("apply-formation").addEventListener("click", applyFormation);
  $("camera-base").addEventListener("change", () => { state.cameraBase = $("camera-base").value.trim(); state.cameraNamesKey = ""; persist(); renderCameras(Object.keys(state.telemetryFrame?.drones || {})); });
  $("clear-trails").addEventListener("click", () => { state.trails.clear(); renderMap(); });
  $("zoom-in").addEventListener("click", () => { state.mapZoom = Math.min(8, state.mapZoom * 1.25); renderMap(); }); $("zoom-out").addEventListener("click", () => { state.mapZoom = Math.max(0.25, state.mapZoom / 1.25); renderMap(); }); $("zoom-reset").addEventListener("click", () => { state.mapZoom = 1; renderMap(); });
  $("map").addEventListener("click", (event) => {
    // Vehicle markers are interactive for map定位.  The goal marker is
    // pointer-transparent so clicking it still selects a new endpoint.
    if (event.target.closest(".drone-marker") || ACTIVE_MISSION_STATES.has(state.mission?.state)) return;
    if (!state.map) return;
    const rect = $("map").getBoundingClientRect(); const [dataMinX, dataMaxX, dataMinY, dataMaxY] = state.map.bounds_xy_m; const centerX = (dataMinX + dataMaxX) / 2; const centerY = (dataMinY + dataMaxY) / 2; const aspect = rect.width / rect.height; const viewWidth = Math.max(dataMaxX - dataMinX, (dataMaxY - dataMinY) * aspect) / state.mapZoom; const viewHeight = viewWidth / aspect; const minX = centerX - viewWidth / 2; const maxY = centerY + viewHeight / 2; const x = minX + ((event.clientX - rect.left) / rect.width) * viewWidth; const y = maxY - ((event.clientY - rect.top) / rect.height) * viewHeight; if (x < dataMinX || x > dataMaxX || y < dataMinY || y > dataMaxY) { $("map-coordinates").textContent = "选点超出地图数据覆盖范围"; return; } const z = Number($("goal-z").value); state.selectedGoal = [x, y, Number.isFinite(z) ? z : Number(state.capabilities?.cruise_z_ned_m ?? -2)]; $("goal-x").value = x.toFixed(2); $("goal-y").value = y.toFixed(2); $("map-coordinates").textContent = `已选择领航机终点：${x.toFixed(2)}, ${y.toFixed(2)} NED`; resetFormRequests();
  });
  window.addEventListener("online", () => { if (state.savedMissionId) loadMission(state.savedMissionId); });
  document.addEventListener("visibilitychange", () => { if (!document.hidden && state.savedMissionId) loadMission(state.savedMissionId); });

  startMissionPolling();
  connectTelemetry();
  renderRecoveryNotice();
  renderCameras(state.savedVehicleNames);
  // Reconnect only for read-only capability, map, mission, and camera state.
  // No flight operation is sent during page restoration.
  if (state.token) setTimeout(() => { void loadBackend(); }, 250);
  // Frames run at about 5 FPS; the status endpoint stays at 1 Hz to keep
  // camera metadata and error polling from competing with image requests.
  setInterval(refreshCameraFrames, 50);
  setInterval(refreshCameraStatus, 1000);
})();
