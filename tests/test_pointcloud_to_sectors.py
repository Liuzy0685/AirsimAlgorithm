"""ROUND 3.3 — sector conversion tests: SectorDef objects, float32/float64
boundary snapping, FOV observability, exact boundary classification.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np, pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from perception.pointcloud_to_sectors import (
    pointcloud_to_directional_distances,
    _in_azimuth_half_open,
    _normalize_angle_rad,
    _snap_to_boundaries,
    _collect_boundaries_rad,
)
from perception.perception_config import SectorDef


def _sdef(name, a_min, a_max, e_min=-22.5, e_max=22.5,
          max_range=40.0, min_pts=3, strategy="nearest_k_median"):
    return SectorDef(
        name=name, legacy_name=name,
        azimuth_min_deg=a_min, azimuth_max_deg=a_max,
        elevation_min_deg=e_min, elevation_max_deg=e_max,
        max_range_m=max_range, min_points=min_pts,
        distance_strategy=strategy,
    )


_SECTORS = [
    _sdef("front", -22.5, 22.5), _sdef("back", 157.5, -157.5),
    _sdef("left", -112.5, -67.5), _sdef("right", 67.5, 112.5),
    _sdef("up", -180, 180, 18, 30),
    _sdef("down", -180, 180, -30, -18),
    _sdef("frontLeft", -67.5, -22.5), _sdef("frontRight", 22.5, 67.5),
    _sdef("backLeft", -157.5, -112.5), _sdef("backRight", 112.5, 157.5),
    _sdef("frontUp", -45, 45, 18, 30), _sdef("frontDown", -45, 45, -30, -18),
    _sdef("leftUp", -135, -45, 18, 30), _sdef("rightUp", 45, 135, 18, 30),
    _sdef("leftDown", -135, -45, -30, -18), _sdef("rightDown", 45, 135, -30, -18),
]


# Sectors with min_points=1 for single-point tests (boundary tests use 1 point)
def _sdef1(name, a_min, a_max, e_min=-22.5, e_max=22.5):
    return _sdef(name, a_min, a_max, e_min, e_max, min_pts=1)

_SECTORS_1 = [
    _sdef1("front", -22.5, 22.5), _sdef1("back", 157.5, -157.5),
    _sdef1("left", -112.5, -67.5), _sdef1("right", 67.5, 112.5),
    _sdef1("up", -180, 180, 18, 30),
    _sdef1("down", -180, 180, -30, -18),
    _sdef1("frontLeft", -67.5, -22.5), _sdef1("frontRight", 22.5, 67.5),
    _sdef1("backLeft", -157.5, -112.5), _sdef1("backRight", 112.5, 157.5),
    _sdef1("frontUp", -45, 45, 18, 30), _sdef1("frontDown", -45, 45, -30, -18),
    _sdef1("leftUp", -135, -45, 18, 30), _sdef1("rightUp", 45, 135, 18, 30),
    _sdef1("leftDown", -135, -45, -30, -18), _sdef1("rightDown", 45, 135, -30, -18),
]


def _make_fov_obs(sectors, all_obs=True):
    """Build fov_observability dict for all sectors."""
    return {s.name: (all_obs, 1.0) for s in sectors}


def _triple(pt):
    return np.array([pt, pt, pt], dtype=np.float32)


def _tp64(pt):
    return np.array([pt, pt, pt], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════
# Diagonal direction tests
# ═══════════════════════════════════════════════════════════════════


class TestDiagonalDirections:
    # ROUND 3.3: Vertical sectors now use [18,30] / [-30,-18] elevation.
    # Test points with elevation ~24° (up) or ~-24° (down).
    # For r_xy=3, z = -r_xy * tan(24°) ≈ -3 * 0.4452 ≈ -1.336 for up,
    #                z =  r_xy * tan(24°) ≈  3 * 0.4452 ≈  1.336 for down.
    def test_fl(self):
        assert pointcloud_to_directional_distances(
            _triple([3, -3, 0]), _SECTORS, fov_compatible=True,
            fov_observability=_make_fov_obs(_SECTORS),
        ).sectors["frontLeft"].has_return
    def test_fr(self):
        assert pointcloud_to_directional_distances(
            _triple([3, 3, 0]), _SECTORS, fov_compatible=True,
            fov_observability=_make_fov_obs(_SECTORS),
        ).sectors["frontRight"].has_return
    def test_bl(self):
        assert pointcloud_to_directional_distances(
            _triple([-3, -3, 0]), _SECTORS, fov_compatible=True,
            fov_observability=_make_fov_obs(_SECTORS),
        ).sectors["backLeft"].has_return
    def test_br(self):
        assert pointcloud_to_directional_distances(
            _triple([-3, 3, 0]), _SECTORS, fov_compatible=True,
            fov_observability=_make_fov_obs(_SECTORS),
        ).sectors["backRight"].has_return
    def test_lu(self):
        # left (-90° azimuth), up (~24° elevation)
        assert pointcloud_to_directional_distances(
            _triple([0, -3, -1.336]), _SECTORS, fov_compatible=True,
            fov_observability=_make_fov_obs(_SECTORS),
        ).sectors["leftUp"].has_return
    def test_ru(self):
        # right (+90° azimuth), up (~24° elevation)
        assert pointcloud_to_directional_distances(
            _triple([0, 3, -1.336]), _SECTORS, fov_compatible=True,
            fov_observability=_make_fov_obs(_SECTORS),
        ).sectors["rightUp"].has_return
    def test_ld(self):
        # left (-90° azimuth), down (~-24° elevation)
        assert pointcloud_to_directional_distances(
            _triple([0, -3, 1.336]), _SECTORS, fov_compatible=True,
            fov_observability=_make_fov_obs(_SECTORS),
        ).sectors["leftDown"].has_return
    def test_rd(self):
        # right (+90° azimuth), down (~-24° elevation)
        assert pointcloud_to_directional_distances(
            _triple([0, 3, 1.336]), _SECTORS, fov_compatible=True,
            fov_observability=_make_fov_obs(_SECTORS),
        ).sectors["rightDown"].has_return


# ═══════════════════════════════════════════════════════════════════
# Mid-sector boundary tests (float64 — well inside each sector)
# ═══════════════════════════════════════════════════════════════════


class TestBoundariesStrict:
    """Mid-sector angles hit exactly one horizontal sector."""

    def _check(self, deg, expected):
        r = math.radians(deg)
        pts = np.array([[math.cos(r) * 5, math.sin(r) * 5, 0]], dtype=np.float64)
        dd = pointcloud_to_directional_distances(
            pts, _SECTORS_1, fov_compatible=True,
            fov_observability=_make_fov_obs(_SECTORS_1),
        )
        h_names = [
            "front", "frontRight", "right", "backRight",
            "back", "backLeft", "left", "frontLeft",
        ]
        hits = [n for n in h_names if dd.sectors[n].has_return]
        assert hits == [expected], f"{deg}° → expected [{expected}], got {hits}"

    def test_0_front(self): self._check(0.0, "front")
    def test_45_frontRight(self): self._check(45.0, "frontRight")
    def test_90_right(self): self._check(90.0, "right")
    def test_135_backRight(self): self._check(135.0, "backRight")
    def test_180_back(self): self._check(180.0, "back")
    def test_m180_back(self): self._check(-180.0, "back")
    def test_m135_backLeft(self): self._check(-135.0, "backLeft")
    def test_m90_left(self): self._check(-90.0, "left")
    def test_m45_frontLeft(self): self._check(-45.0, "frontLeft")


# ═══════════════════════════════════════════════════════════════════
# Exact boundary tests: float32 and float64 MUST hit exactly one sector
# ROUND 3.3: With snapping enabled, boundary points are snapped to
# the exact boundary and then half-open [min, max) ensures exactly
# one hit (upper boundary → next sector)
# ═══════════════════════════════════════════════════════════════════

_EXACT_BOUNDARIES = [
    # (angle_deg, expected_sector_for_upper_boundary)
    # +22.5° is the upper boundary of front, lower boundary of frontRight
    # Half-open [min, max): angle == upper bound → NOT in this sector → next sector
    (22.5, "frontRight"),         # upper boundary of front
    (-22.5, "front"),             # upper boundary of frontLeft? No -22.5 is upper of frontLeft epoch, lower of front
    (67.5, "right"),              # upper boundary of frontRight
    (-67.5, "frontLeft"),         # upper boundary of left? left is [-112.5, -67.5)
    (112.5, "backRight"),         # upper boundary of right
    (-112.5, "left"),             # upper boundary of backLeft
    (157.5, "back"),              # upper boundary of backRight
    (-157.5, "backLeft"),         # upper boundary of back... wait, back is [157.5, -157.5)
    (180.0, "back"),              # +180° normalises to -180°, back sector wraps
    (-180.0, "back"),             # -180° → back sector
]

# The sectors used:
# front:      [-22.5, 22.5)
# frontRight: [22.5, 67.5)
# right:      [67.5, 112.5)
# backRight:  [112.5, 157.5)
# back:       [157.5, -157.5)  (wrap)
# backLeft:   [-157.5, -112.5)
# left:       [-112.5, -67.5)
# frontLeft:  [-67.5, -22.5)


class TestExactBoundaryFloat64:
    """float64 exact boundary: each point hits exactly ONE horizontal sector."""

    def _check(self, deg, expected):
        r = math.radians(deg)
        pts = np.array([[math.cos(r) * 5, math.sin(r) * 5, 0]], dtype=np.float64)
        dd = pointcloud_to_directional_distances(
            pts, _SECTORS_1, fov_compatible=True,
            fov_observability=_make_fov_obs(_SECTORS_1),
        )
        h_names = [
            "front", "frontRight", "right", "backRight",
            "back", "backLeft", "left", "frontLeft",
        ]
        hits = [n for n in h_names if dd.sectors[n].has_return]
        assert len(hits) == 1, f"{deg}° f64 → expected exactly 1 hit, got {hits}"
        assert hits[0] == expected, f"{deg}° f64 → expected [{expected}], got {hits}"

    def test_22_5_f64(self): self._check(22.5, "frontRight")
    def test_m22_5_f64(self): self._check(-22.5, "front")
    def test_67_5_f64(self): self._check(67.5, "right")
    def test_m67_5_f64(self): self._check(-67.5, "frontLeft")
    def test_112_5_f64(self): self._check(112.5, "backRight")
    def test_m112_5_f64(self): self._check(-112.5, "left")
    def test_157_5_f64(self): self._check(157.5, "back")
    def test_m157_5_f64(self): self._check(-157.5, "backLeft")
    def test_180_f64(self): self._check(180.0, "back")
    def test_m180_f64(self): self._check(-180.0, "back")


class TestExactBoundaryFloat32:
    """float32 exact boundary: snapping ensures exactly ONE hit per point.
    ROUND 3.3: With snapping, these must always find exactly one sector."""

    def _check(self, deg, expected):
        r = math.radians(deg)
        pts = np.array([[math.cos(r) * 5, math.sin(r) * 5, 0]], dtype=np.float32)
        dd = pointcloud_to_directional_distances(
            pts, _SECTORS_1, fov_compatible=True,
            fov_observability=_make_fov_obs(_SECTORS_1),
        )
        h_names = [
            "front", "frontRight", "right", "backRight",
            "back", "backLeft", "left", "frontLeft",
        ]
        hits = [n for n in h_names if dd.sectors[n].has_return]
        assert len(hits) == 1, f"{deg}° f32 → expected exactly 1 hit, got {hits}"
        assert hits[0] == expected, f"{deg}° f32 → expected [{expected}], got {hits}"

    def test_22_5_f32(self): self._check(22.5, "frontRight")
    def test_m22_5_f32(self): self._check(-22.5, "front")
    def test_67_5_f32(self): self._check(67.5, "right")
    def test_m67_5_f32(self): self._check(-67.5, "frontLeft")
    def test_112_5_f32(self): self._check(112.5, "backRight")
    def test_m112_5_f32(self): self._check(-112.5, "left")
    def test_157_5_f32(self): self._check(157.5, "back")
    def test_m157_5_f32(self): self._check(-157.5, "backLeft")
    def test_180_f32(self): self._check(180.0, "back")
    def test_m180_f32(self): self._check(-180.0, "back")


class TestExactBoundaryAngleArray:
    """Direct angle array tests — no point cloud conversion."""
    def test_snap_22_5(self):
        """+22.5° float32 should snap to exactly +22.5° and be classified."""
        deg = 22.5
        r = math.radians(deg)
        # Create a point at exactly 22.5° azimuth, float32
        pts = np.array([[math.cos(r) * 5, math.sin(r) * 5, 0]], dtype=np.float32)
        dd = pointcloud_to_directional_distances(
            pts, _SECTORS_1, fov_compatible=True,
            fov_observability=_make_fov_obs(_SECTORS_1),
        )
        # front: [-22.5, 22.5) → 22.5 is NOT in front (half-open upper bound)
        # frontRight: [22.5, 67.5) → 22.5 IS in frontRight
        assert dd.sectors["frontRight"].has_return
        assert not dd.sectors["front"].has_return


# ═══════════════════════════════════════════════════════════════════
# FOV observability propagation tests
# ═══════════════════════════════════════════════════════════════════


class TestFovObservability:
    """ROUND 3.3: FOV metadata must be propagated into SectorMeasurement."""

    def test_observable_by_fov_set(self):
        """With all sectors observable in fov_observability."""
        dd = pointcloud_to_directional_distances(
            _triple([5, 0, 0]), _SECTORS,
            fov_compatible=True,
            fov_observability=_make_fov_obs(_SECTORS, all_obs=True),
        )
        assert dd.sectors["front"].observable_by_fov is True
        assert dd.sectors["front"].fov_coverage_fraction == 1.0

    def test_unobservable_by_fov_set(self):
        """With all sectors unobservable."""
        obs = {s.name: (False, 0.0) for s in _SECTORS}
        dd = pointcloud_to_directional_distances(
            _triple([5, 0, 0]), _SECTORS,
            fov_compatible=False,
            fov_observability=obs,
        )
        assert dd.sectors["front"].observable_by_fov is False
        assert dd.sectors["front"].fov_coverage_fraction == 0.0

    def test_fov_metadata_missing_defaults_to_false(self):
        """When fov_observability is None, observable_by_fov defaults to False."""
        dd = pointcloud_to_directional_distances(
            _triple([5, 0, 0]), _SECTORS,
            fov_compatible=False,
        )
        # Default (no fov_observability provided) → False
        assert dd.sectors["front"].observable_by_fov is False


class TestUnobservableSectorLegacy:
    """ROUND 3.3: unobservable and partially observable sectors cannot
    be converted to legacy distances."""

    def test_unobservable_cannot_convert(self):
        obs = {s.name: (False, 0.0) for s in _SECTORS}
        dd = pointcloud_to_directional_distances(
            _triple([5, 0, 0]), _SECTORS,
            fov_compatible=False,
            fov_observability=obs,
        )
        with pytest.raises(ValueError, match="FOV incompatible"):
            dd.to_legacy_ray_distances()

    def test_partially_observable_cannot_convert(self):
        # partially = not fully, not zero coverage
        obs = {}
        for s in _SECTORS:
            obs[s.name] = (False, 0.5)  # partially observable
        dd = pointcloud_to_directional_distances(
            _triple([5, 0, 0]), _SECTORS,
            fov_compatible=False,
            fov_observability=obs,
        )
        with pytest.raises(ValueError):
            dd.to_legacy_ray_distances()

    def test_fully_observable_no_return_uses_max_range(self):
        """Fully observable sector with no returns: max_range_m is used."""
        # Use a point that gets filtered out by self-exclusion (too close)
        pts = np.array([[0.01, 0.01, 0.01]], dtype=np.float32)
        dd = pointcloud_to_directional_distances(
            pts, _SECTORS, default_min_points=3,  # requires 3 points
            fov_compatible=True,
            fov_observability=_make_fov_obs(_SECTORS, all_obs=True),
        )
        # front sector has < min_points → has_return=False, distance = max_range
        assert dd.sectors["front"].has_return is False
        assert dd.sectors["front"].distance_m == 40.0
        # Legacy conversion should use max_range for this sector
        legacy = dd.to_legacy_ray_distances()
        assert legacy["front"] == 40.0


# ═══════════════════════════════════════════════════════════════════
# Legacy name validation (SectorDef objects)
# ═══════════════════════════════════════════════════════════════════


class TestLegacyNameRequired:
    def test_missing_legacy_name(self):
        """A sector without matching legacy_name in the map raises ValueError."""
        from models.directional_distances import DirectionalDistances
        from models.sector_measurement import SectorMeasurement

        dd = DirectionalDistances(
            frame_valid=True,
            fov_compatible=True,
            sectors={
                "a": SectorMeasurement(
                    name="a", distance_m=5, point_count=3, has_return=True,
                    azimuth_min_rad=0, azimuth_max_rad=1,
                    elevation_min_rad=0, elevation_max_rad=1,
                    max_range_m=40, min_points=3,
                    observable_by_fov=True, fov_coverage_fraction=1.0,
                )
            },
            legacy_map={},
        )
        with pytest.raises(ValueError, match="legacy_name"):
            dd.to_legacy_ray_distances()


# ═══════════════════════════════════════════════════════════════════
# Empty filtered point cloud
# ═══════════════════════════════════════════════════════════════════


class TestEmptyFiltered:
    def test_empty(self):
        dd = pointcloud_to_directional_distances(
            np.empty((0, 3)), _SECTORS, fov_compatible=True,
        )
        assert dd.frame_valid is False
        assert dd.invalid_reason == "empty_filtered_pointcloud"
