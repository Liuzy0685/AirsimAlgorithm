import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from coordination.navigation import NavigationConfig,CommittedGridRouter

OUT=Path(__file__).resolve().parent
data=np.load(OUT/'ground_geometry_03/target_cube_surface.npz')
v=data['vertices'];tri=v[data['indices'].reshape(-1,3)]
goal=np.array([30.5,-6.,-2.]);e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
rows=[]
for heading in ((1.,.173,.117),(-1.,.231,.137),(.157,1.,.213),(.113,-1.,.217),(.193,.227,1.),(.173,.137,-1.)):
    direction=np.array(heading);direction/=np.linalg.norm(direction)
    pv=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,pv)
    valid=np.abs(det)>1e-9;inv=np.zeros_like(det);inv[valid]=1./det[valid]
    tv=goal-tri[:,0];u=np.einsum('ij,ij->i',tv,pv)*inv
    qv=np.cross(tv,e1);w=qv@direction*inv;t=np.einsum('ij,ij->i',e2,qv)*inv
    hits=sorted(t[valid&(u>=-1e-8)&(w>=-1e-8)&(u+w<=1.+1e-8)&(t>1e-8)])
    unique=[]
    for hit in hits:
        if not unique or abs(hit-unique[-1])>1e-5:unique.append(float(hit))
    rows.append(dict(direction=direction.tolist(),positive_ray_intersections=unique,inside_by_parity=len(unique)%2==1))
assert all(row['inside_by_parity'] for row in rows)
# Reproduce a plausible visible-front-face observation, not B's missing cloud.
points=np.array([(23.1,y) for y in np.arange(-11.5,-1.5+.01,.25)])
router=CommittedGridRouter(NavigationConfig(enabled=True,resolution_m=.25,clearance_m=1.1),clock=lambda:10.,minimum_clearance_m=.9)
router.observe(points);router.capture_next=True
waypoint=router.waypoint((.5,-6.,-2.),tuple(goal))
report=dict(goal=goal.tolist(),render_mesh_triangle_count=len(tri),ray_checks=rows,
    interpretation='Inside the measured closed render mesh in six independent directions; collision mesh is not exported',
    front_face_fixture=dict(source='Synthetic surface samples from measured front-face bounds, not original B LiDAR',
        path_available=router.path_available,goal_blocked=router.goal_blocked,reason=router.last_route_reason,
        waypoint=waypoint,searches=router.search_diagnostics,escape=router.escape_diagnostic))
(OUT/'goal_geometry_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
