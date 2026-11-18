"""Shared, validated deployment configuration for A and its subscribers."""
from dataclasses import dataclass
import math
from pathlib import Path
import yaml


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    snapshot_max_age_s: float = 0.6
    heartbeat_s: float = 1.0
    disconnect_timeout_s: float = 2.0
    startup_timeout_s: float = 15.0
    lease_ttl_s: float = 2.0
    lifecycle_lease_s: float = 90.0
    task_id: str = "swarm-mission"
    leader: str = "Drone1"
    enabled: bool = False
    allow_legacy_readiness: bool = False
    conflict_clear_frames: int = 3
    conflict_clear_duration_s: float = 0.5
    formation_tolerance_m: float = 0.25

    def __post_init__(self):
        if self.host not in ("127.0.0.1", "::1"):
            raise ValueError("state service must bind loopback until authentication is implemented")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("state service port must be 1..65535")
        for key in ("snapshot_max_age_s", "heartbeat_s", "disconnect_timeout_s",
                    "startup_timeout_s", "lease_ttl_s", "lifecycle_lease_s", "conflict_clear_duration_s", "formation_tolerance_m"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.heartbeat_s >= self.disconnect_timeout_s:
            raise ValueError("heartbeat_s must be less than disconnect_timeout_s")
        if type(self.enabled) is not bool or not self.task_id.strip() or not self.leader.strip():
            raise ValueError("invalid enabled/task_id/leader")
        if type(self.allow_legacy_readiness) is not bool:
            raise ValueError("allow_legacy_readiness must be boolean")
        if type(self.conflict_clear_frames) is not int or self.conflict_clear_frames < 1:
            raise ValueError("conflict_clear_frames must be a positive integer")

    @property
    def url(self):
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"ws://{host}:{self.port}/ws/swarm"


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    members = config["drones"]
    names = [m["vehicle_name"] for m in members]
    if not 1 <= len(names) <= 5 or len(set(names)) != len(names) or any(not n.strip() for n in names):
        raise ValueError("swarm must contain 1..5 distinct vehicles")
    service = ServiceConfig(**config.get("state_service", {}))
    if service.leader not in names:
        raise ValueError("state_service.leader must be a configured vehicle")
    return config, service
