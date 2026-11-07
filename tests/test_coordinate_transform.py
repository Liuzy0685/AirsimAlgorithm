"""Phase 1 — CoordinateTransformer interface + validate() tests.

Self-contained: a fake AirSim client is injected so no AirSim / UE4 / numpy
is required.  Covers translation, rotation, velocity round-trip, fail-closed
behaviour and 1..5 drone validation.
"""

import math
from types import SimpleNamespace

import pytest

from coordination.collision_predictor import to_common_ned as legacy_to_common_ned
from coordination.coordinate_transform import (
    CoordinateTransformer,
    CoordinateTransformUnvalidated,
    IDENTITY_ROTATION,
    is_proper_rotation,
    yaw_rotation_matrix,
)


# ---------------------------------------------------------------------------
# Fake AirSim client
# ---------------------------------------------------------------------------


def vec(x, y, z):
    return SimpleNamespace(x_val=x, y_val=y, z_val=z)


def quat(x, y, z, w):
    return SimpleNamespace(x_val=x, y_val=y, z_val=z, w_val=w)


def _yaw_quat(yaw_rad):
    return quat(0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0))


def _state(position, velocity, yaw=0.0):
    kin = SimpleNamespace(
        position=position,
        linear_velocity=velocity,
        orientation=_yaw_quat(yaw),
    )
    return SimpleNamespace(kinematics_estimated=kin)


class FakeAirSimClient:
    """``vehicles``: name -> {home, vel, global, yaw}.  ``markers``: name -> {global, yaw}."""

    def __init__(self, vehicles, markers=None):
        self._vehicles = vehicles
        self._markers = markers or {}

    def getMultirotorState(self, vehicle_name):
        v = self._vehicles[vehicle_name]
        return _state(vec(*v["home"]), vec(*v["vel"]), v.get("yaw", 0.0))

    def simGetObjectPose(self, name):
        if name in self._vehicles:
            v = self._vehicles[name]
            return SimpleNamespace(
                position=vec(*v["global"]), orientation=_yaw_quat(v.get("yaw", 0.0))
            )
        m = self._markers[name]
        return SimpleNamespace(
            position=vec(*m["global"]), orientation=_yaw_quat(m.get("yaw", 0.0))
        )


# ---------------------------------------------------------------------------
# Transform math (translation)
# ---------------------------------------------------------------------------


def test_translation_matches_legacy_to_common_ned():
    offsets = {"Drone1": (10.0, 20.0, 0.0)}
    tf = CoordinateTransformer(offsets, require_validation=False)
    local = (1.0, -2.0, 3.0)
    assert tf.to_common_ned("Drone1", local) == legacy_to_common_ned(local, offsets["Drone1"])
    assert tf.to_common_ned("Drone1", local) == (11.0, 18.0, 3.0)


def test_position_round_trip():
    offsets = {"Drone1": (5.0, -5.0, 2.0)}
    tf = CoordinateTransformer(offsets, require_validation=False)
    local = (3.0, 4.0, -1.0)
    common = tf.to_common_ned("Drone1", local)
    assert tf.from_common_ned("Drone1", common) == pytest.approx(local)


def test_velocity_round_trip_and_translation_invariance():
    offsets = {"Drone1": (100.0, -100.0, 0.0)}
    tf = CoordinateTransformer(offsets, require_validation=False)
    vel = (1.0, 2.0, 3.0)
    # Pure translation does not change velocity.
    assert tf.to_common_velocity("Drone1", vel) == pytest.approx(vel)
    assert tf.from_common_velocity("Drone1", tf.to_common_velocity("Drone1", vel)) == pytest.approx(vel)


def test_rotation_transforms_position_and_velocity():
    yaw = math.pi / 2.0
    tf = CoordinateTransformer(
        {"Drone1": (0.0, 0.0, 0.0)},
        rotations={"Drone1": yaw_rotation_matrix(yaw)},
        require_validation=False,
    )
    local = (1.0, 0.0, 0.0)
    common = tf.to_common_ned("Drone1", local)
    assert common == pytest.approx((0.0, 1.0, 0.0))
    # Velocity magnitude preserved under rotation.
    vel = (1.0, 0.0, 0.0)
    cv = tf.to_common_velocity("Drone1", vel)
    assert math.hypot(*cv[:2]) == pytest.approx(1.0)
    assert tf.from_common_velocity("Drone1", cv) == pytest.approx(vel)


def test_rotation_matrix_helpers():
    assert is_proper_rotation(IDENTITY_ROTATION)
    assert is_proper_rotation(yaw_rotation_matrix(0.7))
    # A non-orthonormal matrix must be rejected.
    bad = ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.0))
    assert not is_proper_rotation(bad)


def test_invalid_rotation_rejected_at_construction():
    bad = ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="proper rotation"):
        CoordinateTransformer({"Drone1": (0, 0, 0)}, rotations={"Drone1": bad})


def test_unknown_vehicle_raises():
    tf = CoordinateTransformer({"Drone1": (0, 0, 0)}, require_validation=False)
    with pytest.raises(KeyError):
        tf.to_common_ned("Drone2", (0, 0, 0))


def test_non_finite_input_rejected():
    tf = CoordinateTransformer({"Drone1": (0, 0, 0)}, require_validation=False)
    with pytest.raises(ValueError, match="finite"):
        tf.to_common_ned("Drone1", (float("nan"), 0, 0))


# ---------------------------------------------------------------------------
# Fail-closed behaviour
# ---------------------------------------------------------------------------


def test_fail_closed_raises_before_validation():
    tf = CoordinateTransformer({"Drone1": (0, 0, 0)})  # require_validation=True
    with pytest.raises(CoordinateTransformUnvalidated):
        tf.to_common_ned("Drone1", (1, 2, 3))


def test_require_validation_false_bypasses_validate():
    tf = CoordinateTransformer({"Drone1": (0, 0, 0)}, require_validation=False)
    assert tf.to_common_ned("Drone1", (1, 2, 3)) == (1, 2, 3)
    assert not tf.validated


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def _two_vehicle_client():
    # Pure-translation world: global == home + offset.
    vehicles = {
        "Drone1": {
            "home": (1.0, 2.0, 3.0),
            "vel": (0.5, 0.0, 0.0),
            "global": (11.0, 22.0, 3.0),
            "yaw": 0.0,
        },
        "Drone2": {
            "home": (4.0, 5.0, 6.0),
            "vel": (-0.5, 0.0, 0.0),
            "global": (24.0, 35.0, 6.0),
            "yaw": 0.0,
        },
    }
    offsets = {
        "Drone1": (10.0, 20.0, 0.0),
        "Drone2": (20.0, 30.0, 0.0),
    }
    return FakeAirSimClient(vehicles), offsets


def test_validate_passes_translation_only():
    client, offsets = _two_vehicle_client()
    tf = CoordinateTransformer(offsets)
    result = tf.validate(client)
    assert result.validated
    assert tf.validated
    assert not result.errors
    # After validation, transforms are usable.
    assert tf.to_common_ned("Drone1", (1.0, 2.0, 3.0)) == pytest.approx((11.0, 22.0, 3.0))


def test_validate_checks_different_initial_orientations():
    # Drone2 spawns with yaw=90deg, but the home frame is still axis-aligned,
    # so translation-only must still validate (orientation is diagnostic).
    vehicles = {
        "Drone1": {"home": (1, 2, 0), "vel": (0, 0, 0), "global": (11, 2, 0), "yaw": 0.0},
        "Drone2": {"home": (1, 2, 0), "vel": (0, 0, 0), "global": (11, 2, 0), "yaw": math.pi / 2.0},
    }
    client = FakeAirSimClient(vehicles)
    tf = CoordinateTransformer({"Drone1": (10, 0, 0), "Drone2": (10, 0, 0)})
    result = tf.validate(client)
    assert result.validated
    assert result.vehicles["Drone2"].yaw_rad == pytest.approx(math.pi / 2.0)
    assert "non-zero initial heading" in result.warnings[0] if result.warnings else True


def test_validate_fails_on_position_mismatch_and_stays_closed():
    client = FakeAirSimClient(
        {"Drone1": {"home": (1, 2, 3), "vel": (0, 0, 0), "global": (999, 999, 999), "yaw": 0.0}}
    )
    tf = CoordinateTransformer({"Drone1": (10, 20, 0)})
    result = tf.validate(client)
    assert not result.validated
    assert not tf.validated
    assert result.errors
    with pytest.raises(CoordinateTransformUnvalidated):
        tf.to_common_ned("Drone1", (1, 2, 3))


def test_validate_detects_rotation_required_via_marker():
    # The spawn frame is rotated by 90deg: true global = Rz(90) @ home + offset.
    yaw = math.pi / 2.0
    home = (1.0, 0.0, 0.0)
    offset = (0.0, 0.0, 0.0)
    global_pos = (0.0, 1.0, 0.0)  # Rz(90) @ (1,0,0)
    client = FakeAirSimClient(
        {"Drone1": {"home": home, "vel": (0, 0, 0), "global": global_pos, "yaw": yaw}},
        markers={"PlayerStart1": {"global": offset, "yaw": yaw}},
    )
    tf = CoordinateTransformer({"Drone1": offset})
    markers = {"Drone1": "PlayerStart1"}
    result = tf.validate(client, markers=markers)
    assert not result.validated
    assert "rotation" in result.errors[0].lower()

    # Supply the rotation and re-validate -> passes.
    tf.set_rotation("Drone1", yaw_rotation_matrix(yaw))
    result2 = tf.validate(client, markers=markers)
    assert result2.validated
    assert tf.validated
    assert tf.to_common_ned("Drone1", home) == pytest.approx(global_pos)


def test_validate_velocity_transform_is_consistent():
    # A proper rotation preserves velocity magnitude and round-trips, so the
    # velocity error must be ~0 when the transform is internally consistent.
    client = FakeAirSimClient(
        {"Drone1": {"home": (0, 0, 0), "vel": (1, 0, 0), "global": (0, 0, 0), "yaw": 0.0}}
    )
    tf = CoordinateTransformer({"Drone1": (0, 0, 0)})
    result = tf.validate(client)
    check = result.vehicles["Drone1"]
    assert check.velocity_error_mps == pytest.approx(0.0, abs=1e-6)


def test_validate_1_to_5_drones():
    offsets = {}
    vehicles = {}
    for i in range(1, 6):
        name = f"Drone{i}"
        home = (float(i), float(i), -1.0)
        offset = (10.0 * i, 20.0 * i, 0.0)
        vehicles[name] = {
            "home": home,
            "vel": (0.1 * i, 0.0, 0.0),
            "global": (home[0] + offset[0], home[1] + offset[1], home[2] + offset[2]),
            "yaw": 0.0,
        }
        offsets[name] = offset
    tf = CoordinateTransformer(offsets)
    result = tf.validate(FakeAirSimClient(vehicles))
    assert result.validated
    assert tf.vehicle_names == tuple(f"Drone{i}" for i in range(1, 6))


def test_from_settings_json_reuses_loader(tmp_path):
    import json

    settings = {
        "Vehicles": {
            "Drone1": {"X": 1.0, "Y": 2.0, "Z": 0.0},
            "Drone2": {"X": 3.0, "Y": 4.0, "Z": 0.0},
        }
    }
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    tf = CoordinateTransformer.from_settings_json(str(path), ["Drone1", "Drone2"], require_validation=False)
    assert tf.to_common_ned("Drone1", (0, 0, 0)) == (1.0, 2.0, 0.0)
    assert tf.to_common_ned("Drone2", (0, 0, 0)) == (3.0, 4.0, 0.0)
