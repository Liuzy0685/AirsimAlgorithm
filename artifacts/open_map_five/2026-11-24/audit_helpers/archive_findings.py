import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[2]
RUN=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/open_map_five/2026-11-24';OUT.mkdir(parents=True,exist_ok=False)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def copy(src,dst):
    target=OUT/dst;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)

(OUT/'.gitattributes').write_text('* -text\n',encoding='utf-8',newline='\n')
archive=Path(json.loads((RUN/'imported/archive_sha256.json').read_text(encoding='utf-8'))['source'])
copy(archive,'b_received_logs.zip')
for src,dst in [(RUN/'b_log_analysis.json','b_log_analysis.json'),
    (RUN/'goal_geometry_verification.json','measured_goal_and_synthetic_surface.json'),
    (RUN/'endpoint_preflight.json','offline_endpoint_rejection.json'),
    (RUN/'rejected_before_flight/endpoint_preflight.json','launcher_endpoint_rejection.json'),
    (RUN/'ground_geometry_03/geometry.json','landed_scene_geometry.json'),
    (RUN/'ground_geometry_03/target_cube_surface.npz','target_cube_surface.npz'),
    (RUN/'ground_runtime_cli/runtime_goal_geometry.json','runtime_goal_geometry.json'),
    (RUN/'ground_runtime_cli/validation.json','runtime_cli_validation.json'),
    (RUN/'regression.xml','regression.xml'),
    (ROOT/'configs/operator/flying_example.json','flying_example_manifest.json'),
    (ROOT/'docs/flying_example_five_no_motion_20261124.md','README.md')]:copy(src,dst)
local=RUN/'5_baseline_cam0'
for path in [local/'flight/summary.json',local/'preflight_ground.json',local/'postflight_ground.json',
    local/'explicit_recovery.json',local/'explicit_recovery_02.json',local/'explicit_recovery_03.json',local/'final_ground_release.json']:
    if path.exists():copy(path,'local_startup_failure/'+path.name)
with zipfile.ZipFile(OUT/'local_startup_failure/runtime_logs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted((local/'flight/runtime').glob('*')):
        if path.is_file() and path.suffix in ('.log','.json'):z.write(path,path.name)
    for path in (local/'flight/launcher.log',local/'observer.log'):
        z.write(path,path.name)
for name in ('verify_goal_geometry.py','analyze_b_logs.py','release_verified_ground.py','validate_runtime_geometry_cli.py'):
    copy(RUN/name,'audit_helpers/'+name)
copy(Path(__file__),'audit_helpers/archive_findings.py')
provenance=dict(b_archive_basename=archive.name,b_archive_sha256=sha(archive),
    b_reported_map_path=r'D:\blocks\blocks\Content\FlyingCPP\Maps\FlyingExampleMap.umap',
    b_umap_sha256_user_confirmed='a0fa58a356583f8642dba9585315a670e86f719af5358a8fbb1120a067068a84',
    b_runtime_geometry_confirmed=False,b_exact_source_commit_not_recorded_in_zip=True,
    local_flight_probe_source_commit='ae9b376',fix_source_commit='0510092',targeted_regression_tests=294,
    conclusions=['Measured local render mesh contains configured leader goal',
        'Observed-front-face synthetic fixture reproduces unavailable route with goal_blocked=false; not B raw LiDAR replay',
        'Updated known-map endpoint preflight rejects before flight RPC',
        'Local five-UAV flight probe failed during startup with authoritative_snapshot_unavailable; not B scenario reproduction',
        'Explicit recovery eventually verified landed/API release for all local vehicles; original failures preserved',
        'B runtime level/actor/dependency geometry still requires read-only verification'],
    shared_config_sha256={name:sha(ROOT/name) for name in ('configs/acceptance/settings_5.json',
        'configs/acceptance/swarm_5_navigation.yaml','configs/acceptance/formation_5.yaml')})
(OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Standard plot of measured render bounds, not an AI-generated map.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
manifest=json.loads((OUT/'flying_example_manifest.json').read_text(encoding='utf-8'))
fig,ax=plt.subplots(figsize=(10,6.9))
fig.subplots_adjust(left=.08,right=.99,top=.94,bottom=.14)
for item in manifest['obstacles']:
    lo,hi=item['min_ned_m'],item['max_ned_m']
    if not lo[2]<=-2.<=hi[2]:continue
    highlight=item['id']=='TemplateCube_Rounded_75'
    ax.add_patch(Rectangle(lo[:2],hi[0]-lo[0],hi[1]-lo[1],facecolor='#f59e0b' if highlight else '#94a3b8',
        edgecolor='#b45309' if highlight else '#64748b',alpha=.55,linewidth=1.))
settings=json.loads((ROOT/'configs/acceptance/settings_5.json').read_text(encoding='utf-8'))
starts=[(v['X'],v['Y']) for v in settings['Vehicles'].values()]
ax.scatter(*zip(*starts),color='#2563eb',s=50,zorder=4,label='Shared five-UAV starts')
ax.plot([.5,30.5],[-6,-6],'--',color='#dc2626',lw=1.4,label='Configured start-to-goal direction')
ax.scatter([30.5],[-6],marker='x',s=120,color='#dc2626',lw=2.5,zorder=5,label='Configured leader goal')
ax.annotate('Goal (30.5, -6, -2)\ninside measured block',xy=(30.5,-6),xytext=(10,-13.5),
    arrowprops={'arrowstyle':'->','color':'#b91c1c'},color='#991b1b',fontsize=11)
ax.text(24.,-5.,'Cube 75',fontsize=11,color='#78350f')
ax.set(xlim=(-5,45),ylim=(-16,16),xlabel='World NED X (m)',ylabel='World NED Y (m)',
    title='FlyingExampleMap: configured goal intersects a measured block')
ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=9)
fig.text(.5,.01,'Render-mesh bounds at cruise height; partial geometry, not collision certification.',ha='center',fontsize=9)
fig.savefig(OUT/'configured_goal_map.png',dpi=150);plt.close(fig)
hashes={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'evidence_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
for name,value in hashes.items():assert sha(OUT/name)==value,name
print(json.dumps(dict(files=len(hashes),bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),output=str(OUT))))
