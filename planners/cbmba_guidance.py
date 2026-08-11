"""
CBMBA → Horizontal Guidance Adapter — pure computation, no external RPC.

Selects an actionable horizontal guidance target from a CBMBA 3D path by
finding the earliest path segment that crosses a forward lookahead plane
and linearly interpolating along that segment.

This module does **not** modify the CBMBA path, does **not** replan,
and does **not** produce velocity commands.

NED convention (matching the rest of the project):
    +X = North / forward at yaw=0
    +Y = East  / right   at yaw=0
    +Z = Down
    -Z = Up

Ground-rule note (for future phases):
    Ground plane is roughly ``z >= 0`` in NED (drone **cannot** descend
    below z=0).  If a clearance margin is needed the constraint becomes
    ``z >= ground_z - clearance_m``.  Do **not** write ``z <= ground_z``
    — that would allow underground flight.

Strategy (v2 — segment crossing)
---------------------------------
1. Convert all path waypoints to body-frame XY relative to the current
   drone pose.
2. Walk consecutive segments (A→B) in original path order.
3. Find the **earliest** segment that crosses the forward lookahead plane
   ``body_x = guidance_lookahead_x``, i.e. ``A.x < lookahead <= B.x``.
4. Linearly interpolate along A→B at the crossing point:

       t = (lookahead - A.x) / (B.x - A.x)
       target_body_x = lookahead
       target_body_y = A.y + t * (B.y - A.y)

5. Interpolate world XY and Z using the same *t* (the body transform is
   linear, so parameter *t* is identical in world and body space).
6. Z is preserved in ``source_waypoint`` for diagnostics only — it
   does **not** affect the horizontal guidance geometry.

This allows a 3D CBMBA path whose first few waypoints go backward/up to
still produce a forward horizontal guidance target while preserving the
path's lateral detour information.

Fallback
--------
If no segment crosses the lookahead plane the adapter falls back to the
original first-forward-waypoint rule, but rejects the selection if the
candidate is the **last** waypoint in the path (i.e. effectively the
goal).  If the fallback also fails the adapter returns
``valid=False, reason="no_forward_path_intersection"``.

All behaviour is deterministic — no randomness, no retry, no replan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ── data classes ──


@dataclass
class CbmbaGuidanceParams:
    """Configurable parameters for horizontal guidance target selection."""

    guidance_lookahead_x: float = 1.0
    """Body-frame forward distance (m) at which the lookahead plane sits.
    The adapter finds the earliest path segment crossing ``body_x = lookahead_x``
    and interpolates the guidance target there."""

    min_forward_progress: float = 0.25
    """Body-frame forward distance (m) used **only** in the fallback path
    (first-forward-waypoint rule).  Not used in the primary segment-crossing
    strategy."""

    min_waypoint_distance: float = 0.5
    """Minimum world XY distance (m) — used **only** in the fallback path."""


@dataclass
class CbmbaGuidanceResult:
    """Output of one guidance selection call.

    All nullable fields are ``None`` when ``valid`` is False.
    """

    valid: bool = False
    """True when a guidance target was successfully computed."""

    source_segment: Optional[Tuple[int, int]] = None
    """Indices ``(from_idx, to_idx)`` of the path segment that was interpolated.
    ``None`` when the fallback first-forward-waypoint rule was used instead."""

    interpolated: bool = False
    """True when the target was produced by linear interpolation along a segment.
    False when a raw waypoint was used (fallback only)."""

    source_waypoint: Optional[Tuple[float, float, float]] = None
    """The interpolated 3D point (world NED) at the lookahead-plane crossing,
    or the raw waypoint in fallback mode.  Z is diagnostic only."""

    target_world_xy: Optional[Tuple[float, float]] = None
    """World XY (NED) of the guidance target."""

    target_body_xy: Optional[Tuple[float, float]] = None
    """Body-frame XY offset (m) from drone to guidance target.
    +X = forward, +Y = right."""

    direction_body_xy: Optional[Tuple[float, float]] = None
    """Unit direction vector in body frame pointing to the guidance target."""

    forward_progress_m: float = 0.0
    """Body-frame forward component (body_x) of the guidance target offset."""

    lateral_offset_m: float = 0.0
    """Body-frame lateral component (body_y) of the guidance target offset.
    Positive = right, negative = left."""

    reason: str = ""
    """Human-readable reason for the selection outcome."""


# ── helpers ──


def _body_xy(
    world_x: float,
    world_y: float,
    px: float,
    py: float,
    cos_yaw: float,
    sin_yaw: float,
) -> Tuple[float, float]:
    """Convert a world XY point to body-frame XY relative to drone at (px, py)."""
    dx = world_x - px
    dy = world_y - py
    body_x = dx * cos_yaw + dy * sin_yaw
    body_y = -dx * sin_yaw + dy * cos_yaw
    return body_x, body_y


# ── guidance adapter ──


class CbmbaGuidance:
    """Select an actionable horizontal guidance target from a CBMBA 3D path.

    Primary strategy (segment crossing)
    -----------------------------------
    Converts the full path to body-frame XY, then walks consecutive
    segments.  The **earliest** segment whose body-x span crosses
    ``guidance_lookahead_x`` is selected, and the exact crossing point
    is linearly interpolated.

    This preserves the CBMBA path's lateral detour shape — the guidance
    target is NOT simply the first forward waypoint, but rather the
    point where the path first reaches the lookahead distance.

    Fallback (first-forward-waypoint)
    ---------------------------------
    If no segment crosses the lookahead plane the adapter tries the
    original rule: first waypoint with ``body_x >= min_forward_progress``
    and ``world distance >= min_waypoint_distance``.  If that candidate
    is the **last** waypoint in the path it is rejected (it is just the
    goal, not a meaningful intermediate waypoint).

    Usage::

        guidance = CbmbaGuidance(params)
        result = guidance.select_waypoint(
            drone_position_ned=(x, y, z),
            yaw_rad=yaw,
            path_world=cbmba_result.path_world,
        )
    """

    def __init__(self, params: Optional[CbmbaGuidanceParams] = None) -> None:
        self.params = params if params is not None else CbmbaGuidanceParams()

    # ── public API ──

    def select_waypoint(
        self,
        drone_position_ned: Tuple[float, float, float],
        yaw_rad: float,
        path_world: List[List[float]],
    ) -> CbmbaGuidanceResult:
        """Select a horizontal guidance target from *path_world*.

        Args:
            drone_position_ned: Current drone position ``(x, y, z)`` in NED.
            yaw_rad: Current yaw in radians (0 = North, π/2 = East).
            path_world: Ordered waypoints from ``CbmbaPlanResult.path_world``.
                Each element is ``[x, y, z]`` in world NED.

        Returns:
            ``CbmbaGuidanceResult``.
        """
        # ── input guards ──
        if not path_world or len(path_world) < 2:
            return CbmbaGuidanceResult(valid=False, reason="empty_path")

        if len(drone_position_ned) < 2:
            return CbmbaGuidanceResult(valid=False, reason="invalid_drone_position")

        if any(not math.isfinite(v) for v in drone_position_ned[:2]):
            return CbmbaGuidanceResult(valid=False, reason="nonfinite_drone_position")

        if not math.isfinite(yaw_rad):
            return CbmbaGuidanceResult(valid=False, reason="nonfinite_yaw")

        px, py = drone_position_ned[0], drone_position_ned[1]
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        lookahead = self.params.guidance_lookahead_x

        # ── validate + convert all waypoints to body-frame XY ──
        # body_pts[i] = (bx, by) or None if the waypoint is invalid
        body_pts: List[Optional[Tuple[float, float]]] = []
        for wp in path_world:
            if wp is None or len(wp) < 2:
                body_pts.append(None)
                continue
            wx, wy = wp[0], wp[1]
            if any(not math.isfinite(v) for v in (wx, wy)):
                body_pts.append(None)
                continue
            body_pts.append(_body_xy(wx, wy, px, py, cos_yaw, sin_yaw))

        # ── primary: segment crossing ──
        for i in range(len(body_pts) - 1):
            A = body_pts[i]
            B = body_pts[i + 1]
            if A is None or B is None:
                continue

            # Check for forward crossing: A.x < lookahead <= B.x
            if not (A[0] < lookahead <= B[0]):
                continue

            seg_dx = B[0] - A[0]
            if seg_dx < 1e-9:
                continue  # degenerate vertical-in-X segment — skip

            t = (lookahead - A[0]) / seg_dx
            # Clamp for numerical safety
            t = max(0.0, min(1.0, t))

            # ── interpolate body XY ──
            target_body_x = A[0] + t * (B[0] - A[0])  # ≈ lookahead
            target_body_y = A[1] + t * (B[1] - A[1])

            # ── interpolate world XYZ (same t — body transform is linear) ──
            wA = path_world[i]
            wB = path_world[i + 1]
            twx = wA[0] + t * (wB[0] - wA[0])
            twy = wA[1] + t * (wB[1] - wA[1])
            twz = (wA[2] if len(wA) > 2 else 0.0) + t * (
                (wB[2] if len(wB) > 2 else 0.0) - (wA[2] if len(wA) > 2 else 0.0)
            )

            # ── direction ──
            dist = math.hypot(target_body_x, target_body_y)
            if dist > 1e-12:
                dir_x = target_body_x / dist
                dir_y = target_body_y / dist
            else:
                dir_x, dir_y = 1.0, 0.0

            return CbmbaGuidanceResult(
                valid=True,
                source_segment=(i, i + 1),
                interpolated=True,
                source_waypoint=(twx, twy, twz),
                target_world_xy=(twx, twy),
                target_body_xy=(target_body_x, target_body_y),
                direction_body_xy=(dir_x, dir_y),
                forward_progress_m=target_body_x,
                lateral_offset_m=target_body_y,
                reason="segment_crosses_lookahead",
            )

        # ── fallback: no segment crosses the lookahead plane ──
        # Try the original first-forward-waypoint rule,
        # but reject if the candidate is the last waypoint (goal).
        min_dist = self.params.min_waypoint_distance
        min_fwd = self.params.min_forward_progress
        last_idx = len(path_world) - 1

        for i, waypoint in enumerate(path_world):
            if waypoint is None or len(waypoint) < 2:
                continue
            wx, wy = waypoint[0], waypoint[1]
            if any(not math.isfinite(v) for v in (wx, wy)):
                continue

            dx = wx - px
            dy = wy - py
            dist = math.hypot(dx, dy)
            if dist < min_dist:
                continue

            body_x, body_y = _body_xy(wx, wy, px, py, cos_yaw, sin_yaw)
            if body_x < min_fwd:
                continue

            # ── reject if this is the final goal waypoint ──
            if i == last_idx:
                return CbmbaGuidanceResult(
                    valid=False,
                    reason="no_forward_path_intersection",
                )

            # ── acceptable intermediate waypoint ──
            wz = waypoint[2] if len(waypoint) > 2 else 0.0
            if dist > 1e-12:
                dir_x = body_x / dist
                dir_y = body_y / dist
            else:
                dir_x, dir_y = 1.0, 0.0

            return CbmbaGuidanceResult(
                valid=True,
                source_segment=None,
                interpolated=False,
                source_waypoint=(wx, wy, wz),
                target_world_xy=(wx, wy),
                target_body_xy=(body_x, body_y),
                direction_body_xy=(dir_x, dir_y),
                forward_progress_m=body_x,
                lateral_offset_m=body_y,
                reason="fallback_first_forward",
            )

        # ── complete failure ──
        return CbmbaGuidanceResult(
            valid=False,
            reason="no_forward_path_intersection",
        )
