"""ROUND 3.3 legacy compatibility tests — FOV-gated to_legacy_ray_distances."""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np, pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from perception.pointcloud_to_sectors import pointcloud_to_directional_distances
from perception.perception_config import SectorDef
from models.directional_distances import DirectionalDistances
from models.sector_measurement import SectorMeasurement

_LEGACY = [
    "front", "back", "left", "right", "up", "down",
    "frontLeft", "frontRight", "backLeft", "backRight",
    "frontUp", "frontDown", "leftUp", "rightUp", "leftDown", "rightDown",
]


def _sdef(n, amin, amax, emin=-22.5, emax=22.5):
    return SectorDef(
        name=n, legacy_name=n,
        azimuth_min_deg=amin, azimuth_max_deg=amax,
        elevation_min_deg=emin, elevation_max_deg=emax,
        max_range_m=40.0, min_points=3, distance_strategy="nearest_k_median",
    )


_SECTORS = [
    _sdef("front", -22.5, 22.5), _sdef("back", 157.5, -157.5),
    _sdef("left", -112.5, -67.5), _sdef("right", 67.5, 112.5),
    _sdef("up", -180, 180, 18, 30), _sdef("down", -180, 180, -30, -18),
    _sdef("frontLeft", -67.5, -22.5), _sdef("frontRight", 22.5, 67.5),
    _sdef("backLeft", -157.5, -112.5), _sdef("backRight", 112.5, 157.5),
    _sdef("frontUp", -45, 45, 18, 30), _sdef("frontDown", -45, 45, -30, -18),
    _sdef("leftUp", -135, -45, 18, 30), _sdef("rightUp", 45, 135, 18, 30),
    _sdef("leftDown", -135, -45, -30, -18), _sdef("rightDown", 45, 135, -30, -18),
]


def _make_fov_obs(sectors, all_obs=True):
    return {s.name: (all_obs, 1.0) for s in sectors}


def _triple(pt):
    return np.array([pt, pt, pt], dtype=np.float32)


class TestAllKeys:
    def test_16_keys_present(self):
        dd = pointcloud_to_directional_distances(
            _triple([5, 0, 0]), _SECTORS,
            fov_compatible=True,
            fov_observability=_make_fov_obs(_SECTORS),
        )
        legacy = dd.to_legacy_ray_distances()
        assert set(legacy.keys()) == set(_LEGACY)

    def test_invalid_frame_raises(self):
        """ROUND 3.3: invalid frame raises ValueError (not returns {})."""
        dd = DirectionalDistances(frame_valid=False, invalid_reason="x")
        with pytest.raises(ValueError, match="Cannot convert invalid frame"):
            dd.to_legacy_ray_distances()

    def test_missing_legacy_raises(self):
        partial = [s for s in _SECTORS if s.name != "front"]
        dd = pointcloud_to_directional_distances(
            _triple([5, 0, 0]), partial,
            fov_compatible=True,
            fov_observability=_make_fov_obs(partial),
        )
        with pytest.raises(ValueError, match="Missing required"):
            dd.to_legacy_ray_distances()

    def test_internal_name_diff_from_legacy(self):
        """Internal name 'front_sector' with legacy_name='front' → output key is 'front'."""
        sdefs = [
            SectorDef(
                name="front_sector", legacy_name="front",
                azimuth_min_deg=-22.5, azimuth_max_deg=22.5,
                elevation_min_deg=-22.5, elevation_max_deg=22.5,
                max_range_m=40.0, min_points=3, distance_strategy="nearest_k_median",
            )
        ]
        sdefs += [
            SectorDef(
                name=n, legacy_name=n,
                azimuth_min_deg=0, azimuth_max_deg=1,
                elevation_min_deg=0, elevation_max_deg=1,
                max_range_m=40.0, min_points=3, distance_strategy="nearest_k_median",
            )
            for n in _LEGACY[1:]
        ]
        dd = pointcloud_to_directional_distances(
            _triple([5, 0, 0]), sdefs,
            fov_compatible=True,
            fov_observability=_make_fov_obs(sdefs),
        )
        legacy = dd.to_legacy_ray_distances()
        assert "front" in legacy
        assert "front_sector" not in legacy

    def test_fov_incompatible_raises(self):
        """ROUND 3.3: FOV incompatible raises ValueError."""
        dd = pointcloud_to_directional_distances(
            _triple([5, 0, 0]), _SECTORS,
            fov_compatible=False,
            fov_invalid_sectors=("up", "down"),
        )
        with pytest.raises(ValueError, match="FOV incompatible"):
            dd.to_legacy_ray_distances()

    def test_unobservable_sector_in_legacy_map_raises(self):
        """ROUND 3.3: Even if fov_compatible=True on the DD object,
        individual unobservable sectors cause failure."""
        obs = {}
        for s in _SECTORS:
            obs[s.name] = (s.legacy_name not in ("up", "down"), 1.0 if s.legacy_name not in ("up", "down") else 0.0)
        dd = pointcloud_to_directional_distances(
            _triple([5, 0, 0]), _SECTORS,
            fov_compatible=False,
            fov_observability=obs,
        )
        with pytest.raises(ValueError):
            dd.to_legacy_ray_distances()
