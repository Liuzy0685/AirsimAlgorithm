"""
Shared flight session — AirSim lifecycle management for all flight modes.

Lifecycle phases:
    INITIALIZED → CONTROL_ACQUIRED → ARM_REQUESTED → ARMED
    → TAKEOFF_STARTED → AIRBORNE → LANDING_REQUESTED → LANDING
    → LANDED → DISARMED → CONTROL_RELEASED

All modes call the same landing function.  No duplicated landing code.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("flight_session")

_LOCK_DIR = Path(tempfile.gettempdir()) / "adrone_flight_modes"
_LOCK_FILE = _LOCK_DIR / "flight_mode.lock"


class SessionPhase(Enum):
    UNINITIALIZED = auto()
    INITIALIZED = auto()
    CONTROL_ACQUIRED = auto()
    ARM_REQUESTED = auto()
    ARMED = auto()
    TAKEOFF_STARTED = auto()
    AIRBORNE = auto()
    LANDING_REQUESTED = auto()
    LANDING = auto()
    LANDED = auto()
    DISARMED = auto()
    CONTROL_RELEASED = auto()
    MANUAL_INTERVENTION_REQUIRED = auto()


@dataclass
class SessionState:
    phase: SessionPhase = SessionPhase.UNINITIALIZED
    vehicle_validated: bool = False
    lidar_validated: bool = False


class SessionError(RuntimeError):
    """Raised when a lifecycle transition is invalid."""


class ManualInterventionRequired(SessionError):
    """State unknown — human must resolve before automated control resumes."""


@dataclass
class SharedFlightSession:
    """Manages the full AirSim flight lifecycle.

    landing is IDEMPOTENT — second call is a no-op if already done.
    """

    settings_json: str
    mode: str = "manual"
    target_z_ned: float = -1.0
    takeoff_timeout_s: float = 20.0
    max_vertical_speed_mps: float = 0.5

    _adapter: Any = field(default=None, repr=False)
    _client: Any = field(default=None, repr=False)
    _state: SessionState = field(default_factory=SessionState, repr=False)
    _lock_fh: Any = field(default=None, repr=False)
    _owns_lock: bool = field(default=False, repr=False)
    _cleaned_up: bool = field(default=False, repr=False)
    _takeoff_called: bool = field(default=False, repr=False)
    altitude_confirmed: bool = field(default=False, repr=False)
    _landing_done: bool = field(default=False, repr=False)
    _startup_floor_ts: int = field(default=0, repr=False)

    # ── lock ──

    @staticmethod
    def acquire_lock(mode_label: str) -> Optional[Any]:
        _LOCK_DIR.mkdir(parents=True, exist_ok=True)
        try:
            fh = open(str(_LOCK_FILE), "w", encoding="utf-8")
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            fh.write(f"{mode_label}\n{os.getpid()}\n"); fh.flush()
            return fh
        except (IOError, OSError):
            if fh: fh.close()
            return None

    @staticmethod
    def release_lock(fh: Any) -> None:
        if fh is None: return
        try:
            import msvcrt
            fh.seek(0); fh.truncate()
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception: pass
        finally:
            try: fh.close()
            except Exception: pass
        try: _LOCK_FILE.unlink(missing_ok=True)
        except Exception: pass

    # ── public API ──

    def initialize(self) -> SharedFlightSession:
        from adapters.airsim_client import AirSimClientAdapter
        logger.info("Initializing %s mode session …", self.mode)
        self._adapter = AirSimClientAdapter(readonly=False)
        self._adapter.connect()
        self._client = self._adapter.get_raw_client()
        vn = self._adapter.vehicle_name
        vehicles = [str(v) for v in self._adapter.list_vehicles()]
        if vn not in vehicles:
            raise SessionError(f"Vehicle '{vn}' not found. Available: {vehicles}")
        self._state.vehicle_validated = True
        self._state.lidar_validated = True
        self._state.phase = SessionPhase.INITIALIZED
        return self

    def takeoff_and_climb(self, target_z: Optional[float] = None) -> None:
        if self._takeoff_called:
            raise SessionError("takeoff_and_climb() already called.")
        if self._state.phase != SessionPhase.INITIALIZED:
            raise SessionError(f"Cannot takeoff in phase {self._state.phase.name}.")
        self._takeoff_called = True
        if target_z is not None:
            self.target_z_ned = target_z
        vn = self._adapter.vehicle_name

        try:
            self._client.enableApiControl(True, vehicle_name=vn)
            self._state.phase = SessionPhase.CONTROL_ACQUIRED
            logger.info("API control acquired.")
        except Exception as e:
            self._state.phase = SessionPhase.MANUAL_INTERVENTION_REQUIRED
            raise SessionError(f"enableApiControl failed: {e}") from e

        self._state.phase = SessionPhase.ARM_REQUESTED
        try:
            self._client.armDisarm(True, vehicle_name=vn)
            self._state.phase = SessionPhase.ARMED
            logger.info("Vehicle armed.")
        except Exception as e:
            self._handle_arm_failure(vn, e)

        self._state.phase = SessionPhase.TAKEOFF_STARTED
        try:
            self._client.takeoffAsync(timeout_sec=self.takeoff_timeout_s, vehicle_name=vn).join()
            logger.info("Takeoff completed.")
        except Exception as e:
            self._handle_post_arm_failure(vn, "takeoffAsync", e)

        # SimpleFlight takeoff 后处于 AUTO 起飞锁定，moveToZ/hoverAsync 都被忽略。
        # 用 moveByVelocityAsync 直接控制爬升速度（NED 负 Z = 向上），
        # 轮询 actual_z 直到到达目标高度，然后 hover 停止。
        time.sleep(2.0)

        try:
            vz_up = -abs(self.max_vertical_speed_mps)  # NED: 负 = 向上
            self._client.moveByVelocityAsync(
                vx=0.0, vy=0.0, vz=vz_up,
                duration=8.0, vehicle_name=vn,
            )
            # 轮询高度直到到达目标或超时
            _climb_deadline = time.monotonic() + 12.0
            _reached = False
            while time.monotonic() < _climb_deadline:
                time.sleep(0.3)
                try:
                    _st = self._client.getMultirotorState(vehicle_name=vn)
                    _z = float(_st.kinematics_estimated.position.z_val)
                    if _z <= self.target_z_ned + 0.3:  # 到达目标高度（±0.3m）
                        logger.info("altitude_climb_reached  z=%.2f", _z)
                        _reached = True
                        break
                except Exception:
                    pass
            # 停止爬升，悬停
            self._client.hoverAsync(vehicle_name=vn).join()
            self.altitude_confirmed = _reached or self._confirm_altitude(vn)
        except Exception as e:
            self._handle_post_arm_failure(vn, "moveByVelocityAsync", e)

        try:
            self._client.hoverAsync(vehicle_name=vn).join()
            self._state.phase = SessionPhase.AIRBORNE
            logger.info("Airborne — hovering.")
        except Exception as e:
            self._handle_post_arm_failure(vn, "hoverAsync", e)

    # ── LANDING — single entry point, idempotent ──

    def land_and_disarm(self, landing_timeout_s: float = 30.0,
                         poll_timeout_s: float = 30.0) -> bool:
        """Single landing entry for ALL modes (manual G, auto time_limit, cleanup).

        Idempotent: second call is a no-op.
        Logs every step for diagnostics.
        """
        if self._landing_done:
            logger.info("land_and_disarm already completed — skipping.")
            return self._state.phase == SessionPhase.CONTROL_RELEASED
        self._landing_done = True

        vn = self._adapter.vehicle_name
        ph = self._state.phase
        logger.info("landing_requested  current_phase=%s", ph.name)

        # ── INITIALIZED / CONTROL_ACQUIRED: never airborne ──
        if ph == SessionPhase.INITIALIZED:
            logger.info("Phase=INITIALIZED — no control calls to undo.")
            return True
        if ph == SessionPhase.CONTROL_ACQUIRED:
            return self._release_control(vn)

        # ── ARM_REQUESTED: arm state uncertain ──
        if ph == SessionPhase.ARM_REQUESTED:
            return self._cleanup_arm_requested(vn)

        # ── ARMED / TAKEOFF_STARTED / AIRBORNE / LANDING_REQUESTED / LANDING ──
        # → full landing sequence
        if ph in (SessionPhase.ARMED, SessionPhase.TAKEOFF_STARTED,
                   SessionPhase.AIRBORNE, SessionPhase.LANDING_REQUESTED,
                   SessionPhase.LANDING):
            return self._execute_landing(vn, landing_timeout_s, poll_timeout_s)

        # ── LANDED: disarm → release ──
        if ph == SessionPhase.LANDED:
            if not self._disarm(vn): return False
            return self._release_control(vn)

        # ── DISARMED: release ──
        if ph == SessionPhase.DISARMED:
            return self._release_control(vn)

        # ── CONTROL_RELEASED / MANUAL_INTERVENTION_REQUIRED ──
        logger.info("Phase=%s — no further cleanup actions.", ph.name)
        return ph == SessionPhase.CONTROL_RELEASED

    def cleanup(self) -> None:
        if self._cleaned_up: return
        self._cleaned_up = True
        logger.info("Cleaning up session …")
        try:
            self.land_and_disarm()
        except Exception as e:
            logger.warning("Cleanup error: %s", e)
        if self._adapter is not None:
            try: self._adapter.close()
            except Exception: pass
        if self._owns_lock and self._lock_fh is not None:
            self.release_lock(self._lock_fh); self._owns_lock = False
        logger.info("Session cleanup complete (phase=%s).", self._state.phase.name)

    # ── context manager ──

    def __enter__(self) -> SharedFlightSession:
        self.initialize(); return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.cleanup(); return False

    # ── properties ──

    @property
    def adapter(self): return self._adapter
    @property
    def client(self): return self._client
    @property
    def vehicle_name(self) -> str:
        return self._adapter.vehicle_name if self._adapter else "Drone1"
    @property
    def state(self) -> SessionState: return self._state
    @property
    def is_airborne(self) -> bool:
        return self._state.phase == SessionPhase.AIRBORNE
    @property
    def takeoff_called(self) -> bool:
        return self._takeoff_called
    @property
    def landing_done(self) -> bool:
        return self._landing_done

    def set_startup_floor_baseline(self, timestamp: int) -> None:
        """Record the startup floor contact timestamp from preflight warm-up.
        Used by landing poll to distinguish new landing-phase floor contacts."""
        self._startup_floor_ts = timestamp
        logger.info("startup_floor_baseline_timestamp=%d", timestamp)

    # ── internal: arm failure ──

    def _confirm_altitude(self, vn: str) -> bool:
        """Read back actual NED Z after ``moveToZ`` and log a confirmation.

        Phase C2: "Climbed to Z" must reflect a real state read-back, not just
        the command completing.  The drone can settle short of target (e.g.
        z ≈ -0.42 vs -1.0).  If short, retry ``moveToZ`` once and re-read.

        The 0.3 m tolerance here is a climb sanity bound — independent of the
        goal-termination altitude tolerance.

        Returns:
            ``True`` if the read-back confirms the drone is within tolerance of
            ``target_z_ned``; ``False`` if confirmation failed after the retry
            (the caller can then hold an altitude-position command until the
            drone climbs the rest of the way — the Phase C3-R startup gate).
        """
        _TOL = 0.3
        actual_z = float("nan")
        for _attempt in (1, 2):
            try:
                st = self._client.getMultirotorState(vehicle_name=vn)
                actual_z = float(st.kinematics_estimated.position.z_val)
            except Exception as exc:
                logger.warning("altitude_climb_confirm_error: %s", exc)
                actual_z = float("nan")
            error = abs(actual_z - self.target_z_ned) if math.isfinite(actual_z) else float("inf")
            confirmed = error <= _TOL
            logger.info(
                "altitude_climb_confirm  attempt=%d  requested_z=%.2f  "
                "actual_z=%.2f  error=%.2f  confirmed=%s",
                _attempt, self.target_z_ned, actual_z, error,
                "true" if confirmed else "false",
            )
            if confirmed:
                logger.info("Climbed to Z=%.1f m.", self.target_z_ned)
                return True
            logger.warning(
                "altitude_climb_not_confirmed  requested_z=%.2f  actual_z=%.2f  error=%.2f",
                self.target_z_ned, actual_z, error,
            )
            if _attempt == 1:
                try:
                    self._client.moveToZAsync(
                        z=self.target_z_ned, velocity=self.max_vertical_speed_mps,
                        timeout_sec=30, vehicle_name=vn,
                    )
                    # moveToZAsync.join() 在 takeoff 后可能瞬间返回（PX4 offboard
                    # 切换延迟导致命令被忽略），改为轮询 actual_z 直到到达或超时。
                    _climb_deadline = time.monotonic() + 8.0
                    while time.monotonic() < _climb_deadline:
                        time.sleep(0.3)
                        try:
                            _st = self._client.getMultirotorState(vehicle_name=vn)
                            _z = float(_st.kinematics_estimated.position.z_val)
                            if abs(_z - self.target_z_ned) <= _TOL:
                                logger.info("altitude_climb_retry_reached  z=%.2f", _z)
                                break
                        except Exception:
                            pass
                except Exception as exc:
                    logger.warning("altitude_climb_retry_moveToZ_error: %s", exc)
        logger.warning(
            "altitude_climb_failed_confirmation  requested_z=%.2f  actual_z=%.2f",
            self.target_z_ned, actual_z,
        )
        return False

    def _handle_arm_failure(self, vn: str, original_error: Exception) -> None:
        logger.error("armDisarm(True) failed: %s.", original_error)
        try:
            st = self._client.getMultirotorState(vehicle_name=vn)
            on_ground = (int(st.landed_state) == 0)
        except Exception:
            logger.warning("Cannot read landed_state after arm failure.")
            self._state.phase = SessionPhase.MANUAL_INTERVENTION_REQUIRED
            raise SessionError(f"armDisarm failed and cannot read state: {original_error}") from original_error

        if on_ground:
            try:
                self._client.armDisarm(False, vehicle_name=vn)
                logger.info("Disarmed after arm failure (on ground).")
            except Exception as e2:
                logger.warning("Disarm after arm failure also failed: %s", e2)
                self._state.phase = SessionPhase.MANUAL_INTERVENTION_REQUIRED
                raise SessionError(f"armDisarm failed, disarm also failed: {original_error}") from original_error
            self._release_control(vn)
            raise SessionError(f"armDisarm failed (cleaned up): {original_error}") from original_error
        else:
            logger.warning("Not on ground after arm failure — manual intervention required.")
            self._state.phase = SessionPhase.MANUAL_INTERVENTION_REQUIRED
            raise SessionError(f"armDisarm failed and vehicle may be airborne: {original_error}") from original_error

    def _handle_post_arm_failure(self, vn: str, step: str, original_error: Exception) -> None:
        logger.error("%s failed: %s.", step, original_error)
        try:
            st = self._client.getMultirotorState(vehicle_name=vn)
            on_ground = (int(st.landed_state) == 0)
        except Exception:
            logger.warning("Cannot read landed_state after %s failure.", step)
            self._state.phase = SessionPhase.MANUAL_INTERVENTION_REQUIRED
            raise SessionError(f"{step} failed and cannot read state: {original_error}") from original_error

        if on_ground:
            try:
                self._client.armDisarm(False, vehicle_name=vn)
                logger.info("Disarmed after %s failure (on ground).", step)
            except Exception as e2:
                logger.warning("Disarm after %s failure also failed: %s", step, e2)
                self._state.phase = SessionPhase.MANUAL_INTERVENTION_REQUIRED
                raise SessionError(f"{step} failed, disarm also failed: {original_error}") from original_error
            self._release_control(vn)
            raise SessionError(f"{step} failed (on ground, cleaned up): {original_error}") from original_error
        else:
            logger.info("Possibly airborne after %s failure — executing landing.", step)
            if not self._execute_landing(vn, 30.0):
                self._state.phase = SessionPhase.MANUAL_INTERVENTION_REQUIRED
                raise SessionError(f"{step} failed and landing unconfirmed: {original_error}") from original_error
            raise SessionError(f"{step} failed (airborne, cleaned up): {original_error}") from original_error

    # ── phased cleanup ──

    def _cleanup_arm_requested(self, vn: str) -> bool:
        logger.info("Phase=ARM_REQUESTED — confirming ground before disarm.")
        try:
            st = self._client.getMultirotorState(vehicle_name=vn)
            if int(st.landed_state) != 0:
                logger.warning("Not on ground in ARM_REQUESTED — manual intervention required.")
                self._state.phase = SessionPhase.MANUAL_INTERVENTION_REQUIRED
                return False
        except Exception:
            logger.warning("Cannot read state in ARM_REQUESTED — refusing to release.")
            self._state.phase = SessionPhase.MANUAL_INTERVENTION_REQUIRED
            return False
        if not self._disarm(vn):
            self._state.phase = SessionPhase.MANUAL_INTERVENTION_REQUIRED
            return False
        return self._release_control(vn)

    # ── core landing sequence (detailed logging) ──

    def _execute_landing(self, vn: str, landing_timeout_s: float, poll_timeout_s: float = 30.0) -> bool:
        """Full landing: hover → land → poll state → confirm → disarm → release.

        Logs every step for diagnostics.  This is the ONLY landing code path.
        """
        self._state.phase = SessionPhase.LANDING_REQUESTED
        logger.info("landing_requested  phase=%s", self._state.phase.name)

        # 1. Hover to stabilize
        logger.info("hover_command_sent")
        try:
            self._client.hoverAsync(vehicle_name=vn).join()
            logger.info("hover_command_completed")
        except Exception as e:
            logger.warning("hover_command_failed: %s", e)

        # 2. Land
        self._state.phase = SessionPhase.LANDING
        logger.info("land_command_sent  timeout=%.1fs", landing_timeout_s)
        try:
            self._client.landAsync(timeout_sec=landing_timeout_s, vehicle_name=vn).join()
            logger.info("land_command_completed")
        except Exception as e:
            logger.warning("land_command_failed: %s", e)

        # 3. Poll landed state (detailed logging)
        logger.info("landed_state_poll_start")
        landed = self._poll_landed_detailed(vn, max_wait_s=poll_timeout_s, interval_s=0.2)
        if landed:
            self._state.phase = SessionPhase.LANDED
            logger.info("landing_confirmed  phase=%s", self._state.phase.name)
        else:
            logger.warning("landing_not_confirmed  phase=%s", self._state.phase.name)
            return False

        # 4. Disarm
        if not self._disarm(vn):
            logger.warning("disarm_failed")
            return False
        logger.info("disarm_completed")

        # 5. Release API control
        if not self._release_control(vn):
            logger.warning("release_failed")
            return False
        logger.info("api_control_released")
        return True

    def _poll_landed_detailed(self, vn: str, max_wait_s: float, interval_s: float) -> bool:
        """Poll landed_state.  Falls back to latched Floor-contact + stability
        for UE4+AirSim where landed_state may not update after touchdown and
        collision.has_collided only fires once on initial contact.

        Standard:  landed_state == Landed for 3 consecutive frames.
        Fallback:  Latch a NEW landing-phase Floor timestamp, then check
                   velocity + position stability for 2 s regardless of has_collided.
        """
        deadline = time.monotonic() + max_wait_s
        poll_n = 0
        landed_consecutive = 0

        # ── latched floor contact state ──
        floor_latched: bool = False
        floor_latch_ts: int = 0
        floor_latch_obj: str = ""
        stable_consecutive: int = 0
        prev_pz: Optional[float] = None
        stable_since: Optional[float] = None

        last_logged_landed = None
        last_logged_latched = None
        last_logged_stable = None
        confirmed_source = ""

        try:
            import airsim
            LANDED_ENUM = airsim.LandedState.Landed
        except Exception:
            LANDED_ENUM = 0

        _FLOOR_OK = frozenset({"Floor", "Floor_3"})

        while time.monotonic() < deadline:
            poll_n += 1
            try:
                st = self._client.getMultirotorState(vehicle_name=vn)
                col = self._client.simGetCollisionInfo(vehicle_name=vn)
                raw_ls = st.landed_state
                ls_int = int(raw_ls)
                pz = float(st.kinematics_estimated.position.z_val)
                vx = float(st.kinematics_estimated.linear_velocity.x_val)
                vy = float(st.kinematics_estimated.linear_velocity.y_val)
                vz = float(st.kinematics_estimated.linear_velocity.z_val)
                has_col = bool(col.has_collided)
                col_obj = str(col.object_name) if col.object_name else ""
                col_ts = int(col.time_stamp)

                # ── Latch detection: new landing-phase Floor timestamp ──
                if not floor_latched and has_col and col_obj in _FLOOR_OK and col_ts != 0:
                    if self._startup_floor_ts == 0 or col_ts != self._startup_floor_ts:
                        floor_latched = True
                        floor_latch_ts = col_ts
                        floor_latch_obj = col_obj
                        logger.info(
                            "simulation_floor_contact_latched  object=%r  timestamp=%d  "
                            "startup_ts=%d",
                            col_obj, col_ts, self._startup_floor_ts,
                        )

                # ── Non-floor collision after latch → cancel fallback ──
                if floor_latched and has_col and col_ts != 0 and col_ts != floor_latch_ts:
                    if col_obj not in _FLOOR_OK:
                        logger.warning(
                            "landing_non_floor_collision  object=%r  timestamp=%d  "
                            "latched_ts=%d — refusing fallback",
                            col_obj, col_ts, floor_latch_ts,
                        )
                        return False

                # ── Standard check ──
                if ls_int == int(LANDED_ENUM):
                    landed_consecutive += 1
                    stable_consecutive = 0
                    if landed_consecutive >= 3:
                        confirmed_source = "landed_state"
                        break
                # ── Fallback: latched floor + stability ──
                elif floor_latched:
                    is_stable = (abs(vx) < 0.05 and abs(vy) < 0.05 and abs(vz) < 0.05
                                 and (prev_pz is not None and abs(pz - prev_pz) < 0.02))
                    if is_stable:
                        if stable_consecutive == 0:
                            stable_since = time.monotonic()
                            logger.info("landing_stability_started  pos_z=%.3f", pz)
                        stable_consecutive += 1
                        elapsed = time.monotonic() - stable_since if stable_since else 0
                        if elapsed >= 2.0 and stable_consecutive >= 1:
                            confirmed_source = "simulation_floor_contact_fallback"
                            logger.info("landing_stability_elapsed  %.1fs  frames=%d", elapsed, stable_consecutive)
                            break
                    else:
                        stable_consecutive = 0
                        stable_since = None
                    landed_consecutive = 0
                # ── Non-floor collision, no latch → reject ──
                elif has_col and col_obj not in _FLOOR_OK:
                    logger.warning("landed_state_poll_%d  non_floor_collision=%r  refusing fallback", poll_n, col_obj)
                    landed_consecutive = 0
                # ── No latch, no standard: reset ──
                else:
                    landed_consecutive = 0

                # ── throttle logging ──
                current_landed = (ls_int == int(LANDED_ENUM))
                current_latched = floor_latched
                current_stable = bool(stable_consecutive > 0)
                log_this = (poll_n % 10 == 1) or (current_landed != last_logged_landed) or \
                           (current_latched != last_logged_latched) or (current_stable != last_logged_stable)
                if log_this:
                    logger.info(
                        "landed_state_poll_%d  raw=%r  int=%d  landed_enum=%s  "
                        "pos_z=%.3f  vel=(%.3f,%.3f,%.3f)  "
                        "collision=%s  obj=%r  ts=%d  latched=%s",
                        poll_n, raw_ls, ls_int, repr(LANDED_ENUM),
                        pz, vx, vy, vz,
                        has_col, col_obj, col_ts, floor_latched,
                    )
                    last_logged_landed = current_landed
                    last_logged_latched = current_latched
                    last_logged_stable = current_stable

                prev_pz = pz
            except Exception as e:
                logger.warning("landed_state_poll_%d_error: %s", poll_n, e)
                landed_consecutive = 0
                stable_consecutive = 0
                stable_since = None
                prev_pz = None
            time.sleep(interval_s)

        if confirmed_source:
            logger.info("landing_confirmed  source=%s  polls=%d", confirmed_source, poll_n)
            return True
        logger.warning("landed_state_poll_timeout  polls=%d  max_wait=%.1fs", poll_n, max_wait_s)
        return False

    # ── disarm / release ──

    def _disarm(self, vn: str) -> bool:
        try:
            self._client.armDisarm(False, vehicle_name=vn)
            self._state.phase = SessionPhase.DISARMED
            logger.info("disarm_completed")
            return True
        except Exception as e:
            logger.warning("disarm_error: %s", e)
            return False

    def _release_control(self, vn: str) -> bool:
        try:
            self._client.enableApiControl(False, vehicle_name=vn)
            self._state.phase = SessionPhase.CONTROL_RELEASED
            logger.info("api_control_released")
            return True
        except Exception as e:
            logger.warning("release_error: %s", e)
            return False
