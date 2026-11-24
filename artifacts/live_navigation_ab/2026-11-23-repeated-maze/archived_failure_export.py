import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/live_navigation_ab/2026-11-23-repeated-maze'
RUN=Path(__file__).resolve().parent/'pair_03_swept/2_candidate_cam0'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

original=json.loads((OUT/'evidence_sha256.json').read_text(encoding='utf-8'))
for name,value in original.items():assert sha(OUT/name)==value,name
formation=(RUN/'flight/runtime/formation.log').read_text(encoding='utf-8')
drone=(RUN/'flight/runtime/Drone2.log').read_text(encoding='utf-8')
trace=[json.loads(line) for line in (RUN/'flight/navigation.jsonl').read_text(encoding='utf-8').splitlines()]
rows=[row for row in trace if row['vehicle']=='Drone2']
start=None;longest=0.
for row in rows:
    if row['path_available']:start=None
    else:
        start=row['t'] if start is None else start
        longest=max(longest,row['t']-start)
diagnosis=dict(case='pair_03_swept',observed_final_navigation=rows[-1],
    longest_observed_unavailable_span_s=longest,
    formation_accepted_ack_count=formation.count("'accepted': True"),
    formation_rejected_ack_count=formation.count("'accepted': False"),
    lidar_unavailable_at_gateway_in_drone2_log='lidar_unavailable_at_gateway' in drone,
    observed_stop='Route unavailable and no_route_progress; failed to reach final formation slot before 180s',
    not_proven=['Exact cause of missing route','Whether swept geometry caused the failure',
        'Full router memory not persisted: navigation trace retains point count/hash, not complete cached occupancy'],
    next_checks=['Capture bounded full planner obstacle memory, actual pose, rounded start cell and transient peer obstacles on no-route',
        'Replay start-cell occupancy, preferred clearance and escape-path rejection reasons offline',
        'Reproduce and regression-test before further automatic five-UAV/camera flights'],
    raw_file_sha256={str(p.relative_to(RUN)).replace('\\','/'):sha(p) for p in sorted((RUN/'flight/runtime').glob('*.log'))})
with zipfile.ZipFile(OUT/'pair_03_swept/failed_runtime_logs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted((RUN/'flight/runtime').glob('*.log')):z.write(path,path.relative_to(RUN).as_posix())
    for path in (RUN/'flight/environment.json',RUN/'flight/snapshots.jsonl',RUN/'observer.log'):
        z.write(path,path.relative_to(RUN).as_posix())
with zipfile.ZipFile(OUT/'pair_03_swept/failed_runtime_logs.zip') as z:
    assert z.testzip() is None
    for name,expected in diagnosis['raw_file_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==expected
(OUT/'failure_diagnosis.json').write_text(json.dumps(diagnosis,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
shutil.copy2(Path(__file__),OUT/'archived_failure_export.py')
print(json.dumps(diagnosis,ensure_ascii=False))
