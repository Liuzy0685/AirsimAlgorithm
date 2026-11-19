"""LiDAR distance to the motion actually issued, with bounded observation age.

This uses the same piecewise constant velocity model as the local controller;
it is not an acceleration-limited hardware braking guarantee.
"""
import math
import numpy as np


def observed_takeoff_column_clearance(points_local_ned, position_local_ned, target_z_ned):
    """Horizontal clearance to observed returns along the takeoff column.

    Exclude the ground below a landed body, retain raw near-wall returns.
    No returns in the height band means no *observed* obstruction, not proof
    that unobserved overhead space is free.
    """
    cloud=np.asarray(points_local_ned,dtype=float)
    own=np.asarray(position_local_ned,dtype=float)
    if (cloud.ndim!=2 or cloud.shape[1]!=3 or not len(cloud) or own.shape!=(3,)
            or not np.all(np.isfinite(cloud)) or not np.all(np.isfinite(own))
            or not math.isfinite(target_z_ned)):
        raise ValueError('invalid takeoff-column observation')
    lo=min(own[2],target_z_ned)-.1
    hi=own[2]+.1
    in_column_height=cloud[(cloud[:,2]>=lo)&(cloud[:,2]<=hi)]
    return (float(np.linalg.norm(in_column_height[:,:2]-own[:2],axis=1).min())
            if len(in_column_height) else math.inf)


def raw_frame_takeoff_column_clearance(frame, position_local_ned, target_z_ned):
    from transforms.lidar_to_local_ned import sensor_to_local_ned
    if not frame.frame_valid:raise ValueError(frame.invalid_reason)
    pose=frame.sensor_pose
    points=sensor_to_local_ned(frame.point_cloud_sensor,
        [pose['position'][k] for k in ('x','y','z')],
        [pose['orientation'][k] for k in ('x','y','z','w')])
    return observed_takeoff_column_clearance(points,position_local_ned,target_z_ned)


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
