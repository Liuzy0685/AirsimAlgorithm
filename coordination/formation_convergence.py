"""Display evidence of physical formation convergence, never a flight permit."""
import math


class FormationConvergence:
    def __init__(self, *, tolerance_m=.25, relative_speed_mps=.2, dwell_s=1., minimum_frames=3):
        self.tolerance = tolerance_m
        self.relative_speed = relative_speed_mps
        self.dwell = dwell_s
        self.minimum_frames = minimum_frames
        self.reset()

    def reset(self):
        self.identity = self.revision = self.since = self.last = None
        self.frames = 0

    def observe(self, mission, runtime, *, now, now_ms, max_age_s=.6):
        result = dict(physical_settled=False, reason='unavailable', revision=mission.get('applied_revision'),
                      tracking_error_m={}, dwell_s=0., distinct_frames=0)
        feedback = mission.get('formation_runtime')
        def invalid(reason):
            self.reset()
            return dict(result, reason=reason)
        if mission.get('state') != 'RUNNING':
            return invalid('mission_not_running')
        if (runtime.get('stale', True) or not runtime.get('collection_complete')
                or not 0 <= now_ms-runtime.get('timestamp_unix_ms', 0) < max_age_s*1000):
            return invalid('authoritative_state_stale')
        if (feedback is None or feedback.get('revision') != mission.get('applied_revision')
                or not 0 <= now_ms-feedback.get('received_at_unix_ms', 0) < max_age_s*1000):
            return invalid('formation_feedback_stale')
        if feedback.get('effective_mode') != 'nominal':
            return invalid('temporary_or_recovering_formation')
        try:
            names = set(mission['config']['vehicle_names'])
            leader = mission['config']['leader']
            states = runtime['vehicles']
            targets = feedback['slot_targets_ned_m']
            if set(states) != names or set(targets) != names-{leader}:
                return invalid('incomplete_fleet')
            if abs(max(s['timestamp'] for s in states.values())-feedback['sim_timestamp_ns']) > max_age_s*1e9:
                return invalid('simulation_time_skew')
            leader_velocity = states[leader]['linear_velocity_ned_mps']
            for name, target in targets.items():
                values = [*states[name]['position_ned_m'], *target,
                          *states[name]['linear_velocity_ned_mps'], *leader_velocity]
                if len(values) != 12 or not all(math.isfinite(v) for v in values):
                    return invalid('invalid_vehicle_state')
                error = math.dist(states[name]['position_ned_m'], target)
                result['tracking_error_m'][name] = error
                speed = math.dist(states[name]['linear_velocity_ned_mps'], leader_velocity)
                if error > self.tolerance or speed > self.relative_speed:
                    return invalid('tracking_not_converged')
            identity = (runtime['stream_id'], runtime['snapshot_id'])
            if not identity[0] or type(identity[1]) is not int:
                return invalid('invalid_snapshot_identity')
        except (KeyError, TypeError, ValueError, OverflowError):
            return invalid('invalid_runtime_evidence')
        revision = mission['applied_revision']
        if (self.revision != revision or self.identity is not None and
                (identity[0] != self.identity[0] or identity[1] < self.identity[1])
                or self.last is not None and not 0 <= now-self.last < max_age_s):
            self.reset()
        if identity != self.identity:
            if self.since is None:
                self.since = now
            self.frames += 1
            self.identity, self.revision, self.last = identity, revision, now
        elapsed = now-self.since
        settled = self.frames >= self.minimum_frames and elapsed >= self.dwell
        return dict(result, physical_settled=settled, reason='settled' if settled else 'verifying_dwell',
                    dwell_s=elapsed, distinct_frames=self.frames)
