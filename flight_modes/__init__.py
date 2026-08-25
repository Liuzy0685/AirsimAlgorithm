"""Flight modes package — manual and automatic drone operation modes.

Provides:
- SharedFlightSession: AirSim lifecycle (connect, arm, takeoff, land, cleanup)
- ManualMode: keyboard-controlled flight via velocity or attitude commands
- ManualGamepadMode: gamepad (Xbox/XInput) manual flight
- AutomaticMode: LiDAR-based autonomous obstacle avoidance
"""

from flight_modes.shared_flight_session import SharedFlightSession
from flight_modes.manual_mode import ManualMode, ManualControlType
from flight_modes.manual_gamepad_mode import ManualGamepadMode, ManualGamepadController
from flight_modes.automatic_mode import AutomaticMode, AutomaticFlightResult

__all__ = [
    "SharedFlightSession",
    "ManualMode",
    "ManualControlType",
    "ManualGamepadMode",
    "ManualGamepadController",
    "AutomaticMode",
    "AutomaticFlightResult",
]
