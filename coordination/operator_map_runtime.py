"""Read-only map identity checks shared by launch and explicit recovery."""
import hashlib
import math
from coordination.operator_common import require


def verify_map(profile, client):
    from scripts.validate_blocks_maps import active_map
    require(profile.blocks_project is not None, 'map_verification_unavailable', 'Need --blocks-project', 503)
    log = getattr(profile, 'engine_log', None)
    loaded = (active_map(profile.blocks_project, log_path=log) if log
              else active_map(profile.blocks_project))
    require(loaded == profile.map.asset_path, 'runtime_map_mismatch', 'Active Blocks map differs', 409)
    asset = profile.map.asset_file(profile.blocks_project)
    require(hashlib.sha256(asset.read_bytes()).hexdigest() == profile.map.value.get('asset_sha256'),
            'map_asset_changed', 'Re-export map after asset changes', 409)
    for item in profile.map.value.get('obstacles', []) + profile.map.value.get('reference_objects', []):
        pose = client.simGetObjectPose(item['id'])
        scale = client.simGetObjectScale(item['id'])
        actual = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
        require(all(math.isfinite(v) for v in actual)
                and math.dist(actual, item['actor_position_ned_m']) < .03,
                'map_geometry_changed', 'Map object moved; re-export map', 409)
        require(math.dist([scale.x_val, scale.y_val, scale.z_val], item['actor_scale']) < .001,
                'map_geometry_changed', 'Map object scale changed; re-export map', 409)
        q = [pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val]
        require(all(math.isfinite(v) for v in q)
                and abs(sum(a*b for a, b in zip(q, item['actor_orientation_xyzw']))) > .99999,
                'map_geometry_changed', 'Map object rotated; re-export map', 409)
