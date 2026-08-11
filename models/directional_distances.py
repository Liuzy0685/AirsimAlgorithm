"""
Directional distances data model.

Aggregates sector measurements into a frame-level result with a legacy
compatibility layer that uses **config-supplied** ``legacy_name`` values
(not hardcoded mappings).

ROUND 3.3: Added FOV compatibility fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models.sector_measurement import SectorMeasurement

# The 16 required legacy names from the old web project (ROUND 3.1).
_REQUIRED_LEGACY_NAMES: List[str] = [
    "front", "back", "left", "right", "up", "down",
    "frontLeft", "frontRight", "backLeft", "backRight",
    "frontUp", "frontDown", "leftUp", "rightUp", "leftDown", "rightDown",
]


@dataclass
class DirectionalDistances:
    """Frame-level directional distance result.

    Attributes:
        frame_valid: ``True`` if the LiDAR frame passed validation and filtering.
        invalid_reason: Human-readable reason when ``frame_valid`` is ``False``.
        raw_timestamp_ns: AirSim raw timestamp from the input frame.
        received_monotonic_seconds: Monotonic receive time from the input frame.
        minimum_distance_m: Minimum 3-D Euclidean distance across ALL valid
            sector points (or ``inf`` if no points).
        sectors: ``dict[str, SectorMeasurement]`` keyed by internal sector name.
        max_range_m: Configured maximum range (used for empty sectors).
        legacy_map: ``dict[str, str]`` mapping internal name → legacy name
            (populated from config by the conversion function).
        fov_compatible: ``True`` if all 16 required legacy sectors are fully
            observable by the LiDAR FOV.
        fov_invalid_sectors: Tuple of legacy sector names that are NOT fully
            observable by the LiDAR FOV.
    """

    frame_valid: bool = False
    invalid_reason: Optional[str] = None
    raw_timestamp_ns: int = 0
    received_monotonic_seconds: float = 0.0
    minimum_distance_m: float = float("inf")
    sectors: Dict[str, SectorMeasurement] = field(default_factory=dict)
    max_range_m: float = 40.0
    legacy_map: Dict[str, str] = field(default_factory=dict)
    fov_compatible: bool = False
    fov_invalid_sectors: Tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Legacy compatibility (ROUND 3.1 — config-driven, ROUND 3.3 — FOV-gated)
    # ------------------------------------------------------------------

    def to_legacy_ray_distances(self) -> Dict[str, float]:
        """Produce a ``rayDistances`` dict with the exact 16 camelCase keys.

        Returns an empty dict when ``frame_valid`` is ``False`` — an
        invalid frame is NEVER translated to "all max_range".

        ROUND 3.3: Additionally requires ``fov_compatible=True`` and all
        16 required legacy sectors to be fully observable.  An unobservable
        or partially observable sector must NOT be silently converted to
        a safe-looking ``max_range`` — the caller must handle the FOV
        incompatibility before relying on legacy distances.

        Raises ``ValueError`` if:
        - ``frame_valid`` is ``False``
        - ``fov_compatible`` is ``False``
        - Any required legacy sector is missing from the mapping
        """
        if not self.frame_valid:
            raise ValueError(
                "Cannot convert invalid frame to legacy ray distances. "
                "Check frame_valid before calling to_legacy_ray_distances()."
            )

        if not self.fov_compatible:
            raise ValueError(
                f"Cannot convert to legacy ray distances: FOV incompatible. "
                f"Invalid sectors: {list(self.fov_invalid_sectors)}. "
                f"Unobservable/partially-observable sectors must not be "
                f"silently converted to max_range."
            )

        # Build reverse mapping: legacy_name → SectorMeasurement
        result: Dict[str, float] = {}
        seen: set = set()

        for internal_name, sector in self.sectors.items():
            if internal_name not in self.legacy_map:
                raise ValueError(
                    f"Internal sector {internal_name!r} has no legacy_name mapping. "
                    f"All sectors must have an explicit legacy_name in perception.yaml."
                )
            legacy_name = self.legacy_map[internal_name]
            if legacy_name in seen:
                raise ValueError(
                    f"Duplicate legacy_name {legacy_name!r} "
                    f"(internal names: …).  Check perception.yaml."
                )
            seen.add(legacy_name)

            # ROUND 3.3: Verify every sector is fully observable
            if not sector.observable_by_fov:
                raise ValueError(
                    f"Sector {legacy_name!r} is not fully observable by FOV "
                    f"(coverage fraction: {sector.fov_coverage_fraction:.3f}). "
                    f"Cannot produce legacy ray distance."
                )

            result[legacy_name] = (
                sector.distance_m if sector.has_return else sector.max_range_m
            )

        # Verify all 16 required names are present.
        missing = [n for n in _REQUIRED_LEGACY_NAMES if n not in seen]
        if missing:
            raise ValueError(
                f"Missing required legacy names in sector config: {missing}. "
                f"All 16 legacy names must be present in perception.yaml."
            )

        return result
