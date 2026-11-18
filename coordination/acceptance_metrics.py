"""Measured snapshot metrics; no control and no invented obstacle data."""
import itertools
import math


def segment_min_distance(a0, a1, b0, b1):
    relative = [a-b for a,b in zip(a0,b0)]
    delta = [(a-b)-r for a,b,r in zip(a1,b1,relative)]
    denominator = sum(v*v for v in delta)
    t = max(0.,min(1.,-sum(r*v for r,v in zip(relative,delta))/denominator)) if denominator > 1e-12 else 0.
    return math.sqrt(sum((r+t*v)**2 for r,v in zip(relative,delta)))


class AcceptanceMetrics:
    def __init__(self, names, *, safety_distance_m, formation_tolerance_m=.25):
        self.names = tuple(names)
        self.safety_distance_m = safety_distance_m
        self.formation_tolerance_m = formation_tolerance_m
        self.formation_errors = []
        self.formation_settled_frames = 0
        self.frames = self.complete_frames = self.incomplete_frames = 0
        self.min_sampled = self.min_linear = math.inf
        self.path_length = {name:0. for name in self.names}
        self.stopped_s = {name:0. for name in self.names}
        self.collection_durations = []
        self.receive_intervals = []
        self._last = None
        self._last_received = None
        self._last_identity = None
        self.status_events = []
        self._last_status = {}

    def observe(self, snapshot, *, received_at, stream_id, formation_targets=None):
        identity=(stream_id,snapshot.sequence_id)
        if identity == self._last_identity:
            return False
        if self._last_identity is not None and stream_id != self._last_identity[0]:
            self._last = self._last_received = None
            self._last_status.clear()
        self._last_identity=identity
        self.frames += 1
        self.collection_durations.append(snapshot.collection_duration)
        complete=(snapshot.collection_complete and not snapshot.errors and set(snapshot.states)==set(self.names))
        if not complete:
            self.incomplete_frames += 1
            self._last = self._last_received = None
            return True
        self.complete_frames += 1
        positions={n:tuple(snapshot.states[n].position_ned_m) for n in self.names}
        if formation_targets:
            errors={n:math.dist(positions[n],target) for n,target in formation_targets.items()}
            self.formation_errors.append(errors)
            if all(e<=self.formation_tolerance_m for e in errors.values()): self.formation_settled_frames+=1
        dt=received_at-self._last_received if self._last_received is not None else None
        if dt is not None:
            self.receive_intervals.append(dt)
        for a,b in itertools.combinations(self.names,2):
            self.min_sampled=min(self.min_sampled,math.dist(positions[a],positions[b]))
            if self._last is not None:
                self.min_linear=min(self.min_linear,segment_min_distance(self._last[a],positions[a],self._last[b],positions[b]))
        if self._last is not None:
            for n in self.names:
                self.path_length[n] += math.dist(self._last[n],positions[n])
                if math.sqrt(sum(v*v for v in snapshot.states[n].linear_velocity_ned_mps)) < .05:
                    self.stopped_s[n] += max(0.,dt)
        self._last, self._last_received = positions, received_at
        return True

    def status(self, payload, received_at):
        item=dict(payload, received_monotonic_s=received_at)
        name=payload.get('vehicle_name')
        signature=tuple(item.get(k) for k in ('state','reason'))
        if self._last_status.get(name)==signature:
            return
        self._last_status[name]=signature
        self.status_events.append(item)

    def summary(self):
        def finite(v): return v if math.isfinite(v) else None
        def timing(values):
            values=sorted(values)
            return {"count":len(values),"mean_s":sum(values)/len(values) if values else None,
                "p95_s":values[min(len(values)-1,math.ceil(.95*len(values))-1)] if values else None,
                "max_s":max(values) if values else None}
        minimum=min(self.min_sampled,self.min_linear)
        return dict(frames=self.frames,complete_frames=self.complete_frames,incomplete_frames=self.incomplete_frames,
            minimum_sampled_separation_m=finite(self.min_sampled),
            minimum_piecewise_linear_separation_m=finite(self.min_linear),
            separation_observed_ok=(minimum>=self.safety_distance_m) if math.isfinite(minimum) else None,
            configured_safety_distance_m=self.safety_distance_m,
            path_length_m=self.path_length,low_speed_time_s=self.stopped_s,
            formation_observed_frames=len(self.formation_errors),formation_settled_frames=self.formation_settled_frames,
            formation_error_rmse_m=(math.sqrt(sum(v*v for row in self.formation_errors for v in row.values())/
                sum(len(row) for row in self.formation_errors)) if self.formation_errors else None),
            maximum_formation_error_m=max((v for row in self.formation_errors for v in row.values()),default=None),
            collection_timing=timing(self.collection_durations),receive_intervals=timing(self.receive_intervals),
            status_events=self.status_events,
            limitation="Linear interpolation between measured samples is not a continuous-time safety proof; low speed includes deliberate holds and ground time.")
