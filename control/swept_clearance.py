"""LiDAR distance to the motion actually issued, with bounded observation age.

This uses the same piecewise constant velocity model as the local controller;
it is not an acceleration-limited hardware braking guarantee.
"""
import math
import numpy as np


def measured_command_clearance(points, measured_velocity_body, command_velocity_body,
                               duration_s, observation_age_s):
    cloud=np.asarray(points,dtype=float).reshape(-1,3)
    measured=np.asarray(measured_velocity_body,dtype=float)
    command=np.asarray(command_velocity_body,dtype=float)
    if (not len(cloud) or measured.shape!=(3,) or command.shape!=(3,)
            or not all(math.isfinite(t) and t>=0 for t in (duration_s,observation_age_s))
            or not all(np.all(np.isfinite(x)) for x in (cloud,measured,command))):
        raise ValueError('invalid swept-clearance inputs')
    a=np.zeros(3)
    b=measured*observation_age_s
    c=b+command*duration_s
    def distance(start,end):
        segment=end-start
        denom=float(segment@segment)
        t=np.clip((cloud-start)@segment/denom,0,1) if denom>1e-12 else np.zeros(len(cloud))
        return float(np.linalg.norm(cloud-start-t[:,None]*segment,axis=1).min())
    return min(distance(a,b),distance(b,c))
