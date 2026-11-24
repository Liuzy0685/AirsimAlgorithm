"""Landed/API-released runtime geometry verification; no flight writes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coordination.operator_common import require
from coordination.operator_runtime import OperatorProfile
from coordination.operator_map_runtime import verify_map
from scripts.check_mission_endpoints import check


def containment(vertices,indices,point):
    vertices=np.asarray(vertices,float).reshape(-1,3);indices=np.asarray(indices,int)
    require(indices.ndim==1 and len(indices)>=3 and len(indices)%3==0
        and indices.min()>=0 and indices.max()<len(vertices),'invalid_mesh','Invalid triangle indices')
    triangles=vertices[indices.reshape(-1,3)]
    require(np.all(np.isfinite(triangles)),'invalid_mesh','Nonfinite referenced vertices')
    # Weld render seams before checking closed two-manifold edge incidence.
    keys=np.rint(triangles/1e-5).astype(np.int64)
    edges=Counter()
    for face in keys:
        for a,b in ((0,1),(1,2),(2,0)):
            va,vb=tuple(face[a]),tuple(face[b])
            if va!=vb:edges[tuple(sorted((va,vb)))]+=1
    closed=bool(edges) and all(n==2 for n in edges.values())
    e1=triangles[:,1]-triangles[:,0];e2=triangles[:,2]-triangles[:,0]
    rows=[];point=np.asarray(point,float)
    require(point.shape==(3,) and np.all(np.isfinite(point)),'invalid_goal','Nonfinite goal')
    for heading in ((1.,.173,.117),(-1.,.231,.137),(.157,1.,.213),(.113,-1.,.217),(.193,.227,1.),(.173,.137,-1.)):
        direction=np.asarray(heading);direction/=np.linalg.norm(direction)
        pv=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,pv)
        valid=np.abs(det)>1e-9;inv=np.zeros_like(det);inv[valid]=1./det[valid]
        tv=point-triangles[:,0];u=np.einsum('ij,ij->i',tv,pv)*inv
        qv=np.cross(tv,e1);w=qv@direction*inv;t=np.einsum('ij,ij->i',e2,qv)*inv
        hits=sorted(t[valid&(u>=-1e-8)&(w>=-1e-8)&(u+w<=1.+1e-8)&(t>1e-8)])
        unique=[]
        for hit in hits:
            if not unique or abs(hit-unique[-1])>1e-5:unique.append(float(hit))
        rows.append(dict(direction=direction.tolist(),intersections_m=unique,odd=len(unique)%2==1))
    consensus=len({row['odd'] for row in rows})==1
    return dict(triangle_count=len(triangles),closed_after_seam_welding=closed,ray_checks=rows,
        inside_render_mesh=rows[0]['odd'] if closed and consensus else None,
        limitation='Render geometry only; not exported collision geometry or flight authorization')


def inspect(profile,client,goal,actor):
    def ground():
        require(set(client.listVehicles())==set(profile.names),'runtime_fleet_mismatch','Unexpected vehicles')
        require(json.loads(client.getSettingsString())['Vehicles']==profile.settings['Vehicles'],
            'runtime_settings_mismatch','Settings vehicles differ')
        for name in profile.names:
            state=client.getMultirotorState(vehicle_name=name);v=state.kinematics_estimated.linear_velocity
            require(state.landed_state==0 and client.isApiControlEnabled(vehicle_name=name) is False
                and np.linalg.norm([v.x_val,v.y_val,v.z_val])<=.05,
                'vehicle_not_released','Geometry check requires landed, stationary, API-released fleet')
    ground();verify_map(profile,client)
    meshes=client.simGetMeshPositionVertexBuffers()
    offsets=[]
    for name in ('Ground_5','Ground_6'):
        matches=[m for m in meshes if m.name.lower()==name.lower()]
        require(len(matches)==1,'ambiguous_mesh','Need owner-named ground mesh '+name)
        m=matches[0];p=client.simGetObjectPose(name).position
        offsets.append(np.array([m.position.x_val,m.position.y_val,m.position.z_val])-
            np.array([p.x_val,p.y_val,p.z_val])*np.array([100.,100.,-100.]))
    origin=np.mean(offsets,axis=0)
    require(all(np.linalg.norm(v-origin)<2. for v in offsets),'mesh_frame_mismatch','Ground NED calibration differs')
    matches=[m for m in meshes if m.name.lower()==actor.lower()]
    require(len(matches)==1,'ambiguous_mesh','Need a single owner-named mesh '+actor)
    m=matches[0];vertices=(np.asarray(m.vertices,float).reshape(-1,3)-origin)*np.array([.01,.01,-.01])
    referenced=vertices[np.unique(np.asarray(m.indices,int))]
    lo,hi=referenced.min(axis=0),referenced.max(axis=0)
    expected=next((v for v in profile.map.value['obstacles'] if v['id']==actor),None)
    require(expected is not None,'unknown_actor','Actor missing from supplied map manifest')
    result=dict(actor=actor,goal_ned_m=goal,active_map_verified=True,
        asset_sha256=profile.map.value['asset_sha256'],flight_authorized=False,
        measured_min_ned_m=lo.tolist(),measured_max_ned_m=hi.tolist(),
        render_aabb_matches_manifest=bool(np.allclose(lo,expected['min_ned_m'],atol=.03,rtol=0)
            and np.allclose(hi,expected['max_ned_m'],atol=.03,rtol=0)),**containment(vertices,m.indices,goal))
    ground();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('settings-json','swarm-config','formation-config','map-manifest','blocks-project','engine-log'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--actor',default='TemplateCube_Rounded_75')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists')
    client=None
    try:
        endpoint=check(args.settings_json,args.swarm_config,args.formation_config,args.map_manifest)
        profile=OperatorProfile(args.settings_json,args.swarm_config,args.formation_config,args.map_manifest,
            blocks_project=args.blocks_project,engine_log=args.engine_log)
        from adapters.airsim_client import AirSimClientAdapter
        AirSimClientAdapter._ensure_airsim_importable();import airsim
        client=airsim.MultirotorClient(timeout_value=20)
        report=inspect(profile,client,endpoint['leader_goal_ned_m'],args.actor)
    except Exception as exc:
        report=dict(active_map_verified=False,flight_authorized=False,
            reason=getattr(exc,'code',type(exc).__name__),message=str(exc))
    finally:
        if client is not None:client.client.close()
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    return 0 if (report.get('active_map_verified') and report.get('render_aabb_matches_manifest')
        and report.get('inside_render_mesh') is False) else 2


if __name__=='__main__':raise SystemExit(main())
