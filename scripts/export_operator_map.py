"""Export measured cylinder render-mesh AABBs from a landed Blocks instance.

This is a partial display/endpoint map, not a collision-complete navigation map.
No scene asset, vehicle pose or user settings are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.airsim_client import AirSimClientAdapter
from coordination.operator_common import OperatorMap, atomic_json, require
from scripts.validate_blocks_maps import active_map


def export(client, project, map_id, bounds):
    project=Path(project)
    require(active_map(project)=='/Game/'+map_id,'runtime_map_mismatch','Active map does not match export request')
    for name in client.listVehicles():
        require(client.getMultirotorState(vehicle_name=name).landed_state==0
                and not client.isApiControlEnabled(vehicle_name=name),
                'vehicle_not_released','Export only with landed, API-released vehicles')
    names=[n for n in client.simListSceneObjects() if n.lower().startswith('cylinder')]
    require(bool(names),'no_cylinders','No cylinder actors found')
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
    origin=np.median(np.array([row[4] for row in rows]),axis=0)
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
    value=dict(map_id=map_id,frame='world_ned',units='m',origin='AirSim common PlayerStart NED',
        bounds_xy_m=bounds,coverage='partial_cylinder_render_meshes_only',obstacles=sorted(obstacles,key=lambda x:x['id']),
        unreal_origin_cm=origin.tolist(),asset_sha256=hashlib.sha256((project.parent/'Content'/f'{map_id}.umap').read_bytes()).hexdigest())
    OperatorMap(value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks-project',required=True)
    parser.add_argument('--map-id',required=True)
    parser.add_argument('--bounds',type=float,nargs=4,required=True,metavar=('XMIN','XMAX','YMIN','YMAX'))
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    AirSimClientAdapter._ensure_airsim_importable()
    import airsim
    client=airsim.MultirotorClient(timeout_value=10)
    try:
        value=export(client,args.blocks_project,args.map_id,args.bounds)
        atomic_json(args.output,value)
        print(f'Exported {len(value["obstacles"])} measured AABBs; coverage is partial',flush=True)
    finally:
        client.client.close()


if __name__=='__main__':main()
