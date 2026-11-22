"""Export measured cylinder render-mesh AABBs from a landed Blocks instance.

This is a partial display/endpoint map, not a collision-complete navigation map.
No scene asset, vehicle pose or user settings are changed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.airsim_client import AirSimClientAdapter
from coordination.operator_common import OperatorMap, atomic_json, require
from scripts.validate_blocks_maps import active_map


def export(client, project, map_id, bounds, *, asset_path=None, allow_no_cylinders=False,
           reference_objects=(), engine_log=None):
    project=Path(project)
    asset_path=asset_path or '/Game/'+map_id
    descriptor=OperatorMap(dict(map_id=map_id,asset_path=asset_path,frame='world_ned',units='m',
                                bounds_xy_m=bounds,obstacles=[]))
    loaded=active_map(project,log_path=engine_log) if engine_log else active_map(project)
    require(loaded==asset_path,'runtime_map_mismatch','Active map does not match export request')
    for name in client.listVehicles():
        require(client.getMultirotorState(vehicle_name=name).landed_state==0
                and not client.isApiControlEnabled(vehicle_name=name),
                'vehicle_not_released','Export only with landed, API-released vehicles')
    names=[n for n in client.simListSceneObjects() if n.lower().startswith('cylinder')]
    require(bool(names) or allow_no_cylinders,'no_cylinders','No cylinder actors found; explicit partial-map opt-in required')
    references=[]
    for name in reference_objects:
        pose=client.simGetObjectPose(name);scale=client.simGetObjectScale(name)
        pos=[pose.position.x_val,pose.position.y_val,pose.position.z_val]
        require(all(math.isfinite(v) for v in pos),'invalid_reference','Reference object is missing/nonfinite')
        references.append(dict(id=name,actor_position_ned_m=pos,
            actor_scale=[scale.x_val,scale.y_val,scale.z_val],actor_orientation_xyzw=[
                pose.orientation.x_val,pose.orientation.y_val,pose.orientation.z_val,pose.orientation.w_val]))
    require(bool(names) or bool(references),'missing_references','An empty-cylinder map requires measured identity reference objects')
    meshes=client.simGetMeshPositionVertexBuffers()
    rows=[]
    for name in names:
        matching=[m for m in meshes if m.name.lower()==name.lower()]
        require(len(matching)==1,'ambiguous_mesh','Need owner-named, single-component cylinder meshes')
        mesh=matching[0]
        pose=client.simGetObjectPose(name)
        scale=client.simGetObjectScale(name)
        position=np.array([pose.position.x_val,pose.position.y_val,pose.position.z_val])
        unreal=np.array([mesh.position.x_val,mesh.position.y_val,mesh.position.z_val])
        offset=unreal-position*np.array([100.,100.,-100.])
        vertices=np.asarray(mesh.vertices,float).reshape(-1,3)
        indices=np.asarray(mesh.indices,int)
        require(indices.ndim==1 and len(indices)>=3 and len(indices)%3==0
                and indices.min()>=0 and indices.max()<len(vertices),
                'invalid_mesh','Invalid triangle index buffer')
        # The local AirSim exporter includes extra, unreferenced vertex data.
        # Only triangle-referenced vertices describe the rendered surface.
        vertices=vertices[np.unique(indices)]
        require(len(vertices)>=3 and np.all(np.isfinite(vertices)) and np.all(np.isfinite(offset)),
                'invalid_mesh','Invalid vertex buffer')
        rows.append((name,pose,scale,position,offset,vertices))
    # The local AirSim source returns world Unreal centimeters. Verify a common
    # translation against every actor, rather than assuming PlayerStart is zero.
    origin=np.median(np.array([row[4] for row in rows]),axis=0) if rows else None
    require(all(np.linalg.norm(row[4]-origin)<2. for row in rows),
            'mesh_frame_mismatch','Mesh-to-NED frame could not be calibrated consistently')
    obstacles=[]
    for name,pose,scale,position,_,vertices in rows:
        ned=(vertices-origin)*np.array([.01,.01,-.01])
        lo,hi=ned.min(axis=0),ned.max(axis=0)
        require(np.all(lo<=position+.03) and np.all(hi>=position-.03),
                'mesh_frame_mismatch','Actor is outside measured mesh bounds')
        obstacles.append(dict(id=name,shape='aabb',min_ned_m=lo.tolist(),max_ned_m=hi.tolist(),
            geometry_source='measured_render_mesh_aabb',actor_position_ned_m=position.tolist(),
            actor_scale=[scale.x_val,scale.y_val,scale.z_val],
            actor_orientation_xyzw=[pose.orientation.x_val,pose.orientation.y_val,pose.orientation.z_val,pose.orientation.w_val]))
    value=dict(map_id=map_id,asset_path=asset_path,frame='world_ned',units='m',origin='AirSim common PlayerStart NED',
        bounds_xy_m=bounds,coverage='partial_cylinder_render_meshes_only' if rows else 'no_cylinder_geometry_exported_not_free_space',
        obstacles=sorted(obstacles,key=lambda x:x['id']),reference_objects=references,
        unreal_origin_cm=origin.tolist() if origin is not None else None,
        asset_sha256=hashlib.sha256(descriptor.asset_file(project).read_bytes()).hexdigest())
    OperatorMap(value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks-project',required=True)
    parser.add_argument('--map-id',required=True)
    parser.add_argument('--asset-path',help='Nested project asset, e.g. /Game/FlyingCPP/Maps/FlyingExampleMap')
    parser.add_argument('--allow-no-cylinders',action='store_true',help='Explicit partial-map export; never certifies free space')
    parser.add_argument('--reference-objects',nargs='*',default=[])
    parser.add_argument('--engine-log')
    parser.add_argument('--bounds',type=float,nargs=4,required=True,metavar=('XMIN','XMAX','YMIN','YMAX'))
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    AirSimClientAdapter._ensure_airsim_importable()
    import airsim
    client=airsim.MultirotorClient(timeout_value=10)
    try:
        value=export(client,args.blocks_project,args.map_id,args.bounds,asset_path=args.asset_path,
            allow_no_cylinders=args.allow_no_cylinders,reference_objects=args.reference_objects,engine_log=args.engine_log)
        atomic_json(args.output,value)
        print(f'Exported {len(value["obstacles"])} measured AABBs; coverage is partial',flush=True)
    finally:
        client.client.close()


if __name__=='__main__':main()
