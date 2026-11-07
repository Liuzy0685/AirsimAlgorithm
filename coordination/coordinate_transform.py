"""CoordinateTransformer — vehicle-home-relative NED <-> common World NED.

Phase 1 keeps the existing spawn/marker **offset** scheme (pure translation
between each vehicle's home frame and a common PlayerStart frame) and adds two
things the single-drone pipeline did not need:

1. Optional per-vehicle **rotation** — the existing scheme assumes each home
   frame's axes are parallel to the common World-NED axes.  ``validate()``
   checks that assumption against real AirSim poses and, when a rotation is
   needed, the caller can supply it via ``set_rotation()`` / ``rotations=``.
2. **Fail-closed validation** — with ``require_validation=True`` (default),
   every transform raises ``CoordinateTransformUnvalidated`` until
   ``validate()`` has completed successfully.  Unconfirmed coordinate
   transforms must never be used for inter-vehicle coordination.

Conventions
-----------
- ``to_common_ned(v)``      = R @ v + t     (home -> common, position)
- ``from_common_ned(v)``    = R^T @ (v - t) (common -> home, position)
- ``to_common_velocity(v)`` = R @ v         (velocity: translation drops out)
- ``from_common_velocity``  = R^T @ v

With identity rotation this is exactly the existing ``collision_predictor``
``to_common_ned(local, spawn_offset)`` translation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from common.constants import (
    DEFAULT_TRANSFORM_ORIENTATION_TOLERANCE_RAD,
    DEFAULT_TRANSFORM_POSITION_TOLERANCE_M,
)

# A 3x3 rotation matrix (row-major, tuple of 3 rows).
RotationMatrix = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]

Vec3 = Tuple[float, float, float]

IDENTITY_ROTATION: RotationMatrix = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


class CoordinateTransformUnvalidated(RuntimeError):
    """Raised when a transform is used before ``validate()`` has succeeded."""


# ---------------------------------------------------------------------------
# Pure rotation helpers (no AirSim dependency — fully unit-testable)
# ---------------------------------------------------------------------------


def _matvec(m: RotationMatrix, v: Sequence[float]) -> Vec3:
    return tuple(
        m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] for i in range(3)
    )


def _transpose(m: RotationMatrix) -> RotationMatrix:
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))


def _matmul(a: RotationMatrix, b: RotationMatrix) -> RotationMatrix:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _det(m: RotationMatrix) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def is_proper_rotation(m: RotationMatrix, tol: float = 1e-6) -> bool:
    """Return ``True`` when ``m`` is orthonormal with determinant +1."""
    rtr = _matmul(_transpose(m), m)
    for i in range(3):
        for j in range(3):
            if abs(rtr[i][j] - (1.0 if i == j else 0.0)) > tol:
                return False
    return abs(_det(m) - 1.0) <= tol


def yaw_rotation_matrix(yaw_rad: float) -> RotationMatrix:
    """Rotation about the NED Z (Down) axis by ``yaw_rad``.

    Matches the body->world yaw rotation used by ``velocity_controller``:
        vx_world = vx*cos - vy*sin
        vy_world = vx*sin + vy*cos
    """
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def quaternion_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract the Z-axis (yaw) angle in radians from an XYZW quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# ---------------------------------------------------------------------------
# Validation report types
# ---------------------------------------------------------------------------


@dataclass
class VehicleTransformCheck:
    vehicle_name: str
    position_error_m: float = float("inf")
    velocity_error_mps: float = float("inf")
    yaw_rad: Optional[float] = None
    rotation_applied: bool = False
    ok: bool = False
    message: str = ""


@dataclass
class CoordinateTransformValidation:
    """Result of ``CoordinateTransformer.validate()``.

    ``validated`` is ``True`` only when every configured vehicle passed all
    checks.  A ``False`` result leaves the transformer fail-closed.
    """

    validated: bool = False
    vehicles: Dict[str, VehicleTransformCheck] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CoordinateTransformer
# ---------------------------------------------------------------------------


class CoordinateTransformer:
    """Translate/rotate vehicle-home-relative NED to common World NED.

    Parameters
    ----------
    spawn_offsets:
        ``{vehicle_name: (north, east, down)}`` — the translation from each
        vehicle's home origin to the common origin (existing offset scheme).
    rotations:
        Optional ``{vehicle_name: RotationMatrix}``.  Defaults to identity
        (pure translation).  Identity reproduces the existing behaviour.
    require_validation:
        When ``True`` (default), transforms raise until ``validate()`` passes.
    """

    def __init__(
        self,
        spawn_offsets: Mapping[str, Sequence[float]],
        rotations: Optional[Mapping[str, RotationMatrix]] = None,
        require_validation: bool = True,
    ) -> None:
        if not spawn_offsets:
            raise ValueError("spawn_offsets must not be empty")
        self._spawn_offsets: Dict[str, Vec3] = {}
        self._rotations: Dict[str, RotationMatrix] = {}
        for name, offset in spawn_offsets.items():
            if len(offset) != 3:
                raise ValueError(f"spawn offset for {name!r} must be 3D")
            off = tuple(float(c) for c in offset)
            if not all(math.isfinite(c) for c in off):
                raise ValueError(f"spawn offset for {name!r} must be finite")
            self._spawn_offsets[name] = off
            rot = (rotations or {}).get(name, IDENTITY_ROTATION)
            if not is_proper_rotation(rot):
                raise ValueError(f"rotation for {name!r} is not a proper rotation matrix")
            self._rotations[name] = rot
        self._require_validation = require_validation
        self._validated = False
        self._last_validation: Optional[CoordinateTransformValidation] = None

    # ------------------------------------------------------------------
    # Convenience constructors (reuse the existing collision_predictor loaders)
    # ------------------------------------------------------------------

    @classmethod
    def from_settings_json(
        cls,
        settings_json: str,
        vehicle_names: Sequence[str],
        rotations: Optional[Mapping[str, RotationMatrix]] = None,
        require_validation: bool = True,
    ) -> "CoordinateTransformer":
        """Build from AirSim ``settings.json`` spawn offsets."""
        from coordination.collision_predictor import load_spawn_offsets

        return cls(
            load_spawn_offsets(settings_json, vehicle_names),
            rotations=rotations,
            require_validation=require_validation,
        )

    @classmethod
    def from_scene_markers(
        cls,
        client,
        actor_names: Mapping[str, str],
        rotations: Optional[Mapping[str, RotationMatrix]] = None,
        require_validation: bool = True,
    ) -> "CoordinateTransformer":
        """Build from AirSim PlayerStart marker positions (by actor name/tag)."""
        from coordination.collision_predictor import load_scene_marker_offsets

        return cls(
            load_scene_marker_offsets(client, actor_names),
            rotations=rotations,
            require_validation=require_validation,
        )

    # ------------------------------------------------------------------
    # Rotation management
    # ------------------------------------------------------------------

    def set_rotation(self, vehicle_name: str, rotation: RotationMatrix) -> None:
        """Set (or replace) the rotation matrix for one vehicle."""
        if vehicle_name not in self._spawn_offsets:
            raise KeyError(f"unknown vehicle {vehicle_name!r}")
        if not is_proper_rotation(rotation):
            raise ValueError("rotation is not a proper rotation matrix")
        self._rotations[vehicle_name] = rotation
        # Changing the transform invalidates any previous validation result.
        self._validated = False
        self._last_validation = None

    def rotation(self, vehicle_name: str) -> RotationMatrix:
        return self._rotations[vehicle_name]

    # ------------------------------------------------------------------
    # Transform API (fail-closed)
    # ------------------------------------------------------------------

    def to_common_ned(self, vehicle_name: str, local_ned: Sequence[float]) -> Vec3:
        """Home-relative NED -> common World NED (position)."""
        self._assert_ready(vehicle_name)
        return self._to_common_ned_unchecked(vehicle_name, local_ned)

    def from_common_ned(self, vehicle_name: str, common_ned: Sequence[float]) -> Vec3:
        """Common World NED -> home-relative NED (position)."""
        self._assert_ready(vehicle_name)
        return self._from_common_ned_unchecked(vehicle_name, common_ned)

    def to_common_velocity(self, vehicle_name: str, local_velocity: Sequence[float]) -> Vec3:
        """Home-relative velocity -> common velocity (rotation only)."""
        self._assert_ready(vehicle_name)
        return _matvec(self._rotations[vehicle_name], local_velocity)

    def from_common_velocity(self, vehicle_name: str, common_velocity: Sequence[float]) -> Vec3:
        """Common velocity -> home-relative velocity (rotation only)."""
        self._assert_ready(vehicle_name)
        return _matvec(_transpose(self._rotations[vehicle_name]), common_velocity)

    # ------------------------------------------------------------------
    # Internal (unchecked) transform helpers used by validate()
    # ------------------------------------------------------------------

    def _to_common_ned_unchecked(self, vehicle_name: str, local_ned: Sequence[float]) -> Vec3:
        self._require_known(vehicle_name)
        self._require_3d(local_ned, "local_ned")
        rotated = _matvec(self._rotations[vehicle_name], local_ned)
        offset = self._spawn_offsets[vehicle_name]
        return (rotated[0] + offset[0], rotated[1] + offset[1], rotated[2] + offset[2])

    def _from_common_ned_unchecked(self, vehicle_name: str, common_ned: Sequence[float]) -> Vec3:
        self._require_known(vehicle_name)
        self._require_3d(common_ned, "common_ned")
        offset = self._spawn_offsets[vehicle_name]
        translated = _sub(common_ned, offset)
        return _matvec(_transpose(self._rotations[vehicle_name]), translated)

    # ------------------------------------------------------------------
    # Validation (fail-closed)
    # ------------------------------------------------------------------

    def validate(
        self,
        client,
        vehicle_names: Optional[Sequence[str]] = None,
        markers: Optional[Mapping[str, str]] = None,
        *,
        position_tolerance_m: float = DEFAULT_TRANSFORM_POSITION_TOLERANCE_M,
        orientation_tolerance_rad: float = DEFAULT_TRANSFORM_ORIENTATION_TOLERANCE_RAD,
        velocity_tolerance_mps: float = 1e-3,
    ) -> CoordinateTransformValidation:
        """Validate the transform against actual AirSim poses.

        For each vehicle it:
          1. checks the configured rotation matrix is a proper rotation;
          2. compares ``to_common_ned(getMultirotorState().position)`` against
             the actual global pose from ``simGetObjectPose(vehicle)``;
          3. checks the velocity transform preserves magnitude and round-trips
             (a proper rotation must do both; translation drops out);
          4. records the vehicle yaw and, when a ``markers`` mapping is given,
             checks the spawn marker orientation — a rotated marker with an
             identity transform that also fails the position check means a
             rotation matrix is required.

        ``client`` needs two methods (injected fake in unit tests):

        - ``client.getMultirotorState(name)`` returning an object with
          ``.kinematics_estimated.position`` / ``.linear_velocity`` (``Vector3r``
          with ``.x_val/.y_val/.z_val``) and ``.orientation`` (``Quaternionr``).
        - ``client.simGetObjectPose(name)`` returning an object with
          ``.position`` and ``.orientation``.

        Returns a ``CoordinateTransformValidation`` and sets the internal
        fail-closed flag: ``validated=True`` only when every check passed.
        """
        names = list(vehicle_names) if vehicle_names else list(self._spawn_offsets)
        result = CoordinateTransformValidation()
        for name in names:
            check = self._validate_one(
                client,
                name,
                markers,
                position_tolerance_m,
                orientation_tolerance_rad,
                velocity_tolerance_mps,
            )
            result.vehicles[name] = check
            if check.ok:
                result.warnings.extend(
                    f"{name}: {w}" for w in self._warnings_for(check)
                )
            else:
                result.errors.append(f"{name}: {check.message}")

        result.validated = not bool(result.errors)
        self._validated = result.validated
        self._last_validation = result
        return result

    def _validate_one(
        self,
        client,
        name: str,
        markers: Optional[Mapping[str, str]],
        pos_tol: float,
        ori_tol: float,
        vel_tol: float,
    ) -> VehicleTransformCheck:
        if name not in self._spawn_offsets:
            return VehicleTransformCheck(name, ok=False, message="unknown vehicle")

        # 1. Rotation matrix sanity.
        rotation = self._rotations[name]
        if not is_proper_rotation(rotation):
            return VehicleTransformCheck(name, ok=False, message="invalid rotation matrix")

        try:
            state = client.getMultirotorState(name)
            pose = client.simGetObjectPose(name)
            local_pos = _read_vec3(state.kinematics_estimated.position)
            local_vel = _read_vec3(state.kinematics_estimated.linear_velocity)
            actual_pos = _read_vec3(pose.position)
        except Exception as exc:  # pragma: no cover - defensive
            return VehicleTransformCheck(name, ok=False, message=f"pose read failed: {exc}")

        # 2. Position check.
        predicted = self._to_common_ned_unchecked(name, local_pos)
        pos_err = _norm(_sub(predicted, actual_pos))

        # 3. Velocity check: proper rotation preserves magnitude; round-trips.
        common_vel = _matvec(rotation, local_vel)
        round_trip = _sub(_matvec(_transpose(rotation), common_vel), local_vel)
        vel_err = max(
            abs(_norm(common_vel) - _norm(local_vel)),
            _norm(round_trip),
        )

        # 4. Orientation (diagnostic + marker-based rotation detection).
        yaw: Optional[float] = None
        try:
            q = _read_quat(state.kinematics_estimated.orientation)
            yaw = quaternion_yaw(*q)
        except Exception:
            yaw = None

        message = ""
        ok = pos_err <= pos_tol and vel_err <= vel_tol

        if markers and markers.get(name):
            try:
                mpose = client.simGetObjectPose(markers[name])
                mq = _read_quat(mpose.orientation)
                marker_yaw = quaternion_yaw(*mq)
                rotation_applied = not _is_identity(rotation)
                if abs(marker_yaw) > ori_tol:
                    if rotation_applied:
                        # A rotation was supplied; the position check decides.
                        message = (
                            f"marker yaw {marker_yaw:.3f} rad detected; rotation supplied "
                            f"(position_error={pos_err:.3f} m)"
                        )
                    else:
                        # Identity transform + rotated marker => translation-only
                        # is unconfirmed.  Fail closed so the caller must supply
                        # the rotation and re-validate.
                        message = (
                            f"marker yaw {marker_yaw:.3f} rad exceeds "
                            f"{ori_tol:.3f} rad and no rotation supplied — a rotation "
                            "matrix is required"
                        )
                        ok = False
            except Exception as exc:  # pragma: no cover - defensive
                message = f"marker pose read failed: {exc}"

        if not ok and not message:
            message = (
                f"position_error={pos_err:.3f} m (tol {pos_tol:.3f}), "
                f"velocity_error={vel_err:.6f} m/s (tol {vel_tol:.6f})"
            )

        return VehicleTransformCheck(
            vehicle_name=name,
            position_error_m=pos_err,
            velocity_error_mps=vel_err,
            yaw_rad=yaw,
            rotation_applied=not _is_identity(rotation),
            ok=ok,
            message=message,
        )

    @staticmethod
    def _warnings_for(check: VehicleTransformCheck) -> List[str]:
        warnings: List[str] = []
        if check.yaw_rad is not None and abs(check.yaw_rad) > 1e-9:
            warnings.append(f"vehicle yaw={check.yaw_rad:.3f} rad (non-zero initial heading)")
        if check.message:
            warnings.append(check.message)
        return warnings

    # ------------------------------------------------------------------
    # Guards / properties
    # ------------------------------------------------------------------

    def _require_known(self, vehicle_name: str) -> None:
        if vehicle_name not in self._spawn_offsets:
            raise KeyError(f"unknown vehicle {vehicle_name!r}")

    def _require_3d(self, value: Sequence[float], label: str) -> None:
        if len(value) != 3:
            raise ValueError(f"{label} must be 3D, got length {len(value)}")
        if not all(math.isfinite(float(c)) for c in value):
            raise ValueError(f"{label} must be finite, got {value!r}")

    def _assert_ready(self, vehicle_name: str) -> None:
        self._require_known(vehicle_name)
        if self._require_validation and not self._validated:
            raise CoordinateTransformUnvalidated(
                "CoordinateTransformer has not passed validate(); refusing to "
                "return unconfirmed coordinate transforms"
            )

    @property
    def validated(self) -> bool:
        return self._validated

    @property
    def last_validation(self) -> Optional[CoordinateTransformValidation]:
        return self._last_validation

    @property
    def vehicle_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._spawn_offsets))


# ---------------------------------------------------------------------------
# AirSim value readers (accept Vector3r / Quaternionr or plain sequences)
# ---------------------------------------------------------------------------


def _read_vec3(obj) -> Vec3:
    if hasattr(obj, "x_val"):
        return (float(obj.x_val), float(obj.y_val), float(obj.z_val))
    seq = tuple(obj)
    if len(seq) != 3:
        raise ValueError(f"expected 3 components, got {len(seq)}")
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _read_quat(obj) -> Tuple[float, float, float, float]:
    if hasattr(obj, "x_val"):
        return (float(obj.x_val), float(obj.y_val), float(obj.z_val), float(obj.w_val))
    seq = tuple(obj)
    if len(seq) != 4:
        raise ValueError(f"expected 4 components, got {len(seq)}")
    return (float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))


def _is_identity(m: RotationMatrix, tol: float = 1e-9) -> bool:
    return all(
        abs(m[i][j] - (1.0 if i == j else 0.0)) <= tol
        for i in range(3) for j in range(3)
    )
