"""Export measured acceptance evidence and plots from an existing local run."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path


def export(run, output):
    run,output=Path(run),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    summary.pop('status_events',None)
    env=summary.get('environment',{})
    env.pop('scene_objects',None)
    summary['raw_evidence_sha256']={name:hashlib.sha256((run/name).read_bytes()).hexdigest()
        for name in ('summary.json','snapshots.jsonl','environment.json')}
    summary['export_note']='All complete sampled positions exported; plots show cylinder centers, not obstacle outlines. No continuous-time safety or tight-formation accuracy certification.'
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    rows=[json.loads(line) for line in (run/'snapshots.jsonl').read_text(encoding='utf-8').splitlines()]
    complete=[row for row in rows if row['complete']]
    with (output/'trajectory.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        w.writerow(('wall_elapsed_s','snapshot_id','vehicle','x_ned_m','y_ned_m','z_ned_m',
                    'vx_mps','vy_mps','vz_mps','sim_timestamp_ns','landed_state'))
        for row in complete:
            for name,s in row['states'].items():
                w.writerow((row['t'],row['snapshot_id'],name,*s['position'],*s['velocity'],s['sim_timestamp_ns'],s['landed_state']))
    if not complete:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(ax,sep)=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    names=list(complete[0]['states'])
    for name in names:
        states=[r['states'][name]['position'] for r in complete if name in r['states']]
        line,=ax.plot([p[0] for p in states],[p[1] for p in states],label=name,linewidth=1.4)
        ax.scatter(*states[0][:2],marker='o',s=22,color=line.get_color())
        ax.scatter(*states[-1][:2],marker='s',s=22,color=line.get_color())
    cylinders=env.get('cylinders',[])
    if cylinders:
        ax.scatter([c['position_ned_m'][0] for c in cylinders],[c['position_ned_m'][1] for c in cylinders],
                   marker='x',c='black',s=28,label='Cylinder centers (not outlines)')
    ax.set(xlabel='World NED X (m)',ylabel='World NED Y (m)',title='Measured trajectories; circle=start, square=end')
    ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.25);ax.legend(fontsize=8)
    distances=[min(math.dist(r['states'][a]['position'],r['states'][b]['position'])
        for a,b in itertools.combinations(names,2)) for r in complete]
    sep.plot([r['t'] for r in complete],distances,label='Minimum sampled pair distance')
    sep.axhline(summary['configured_safety_distance_m'],color='red',linestyle='--',label='Configured safety distance')
    sep.set(xlabel='Observer wall elapsed time (s)',ylabel='3D separation (m)',title='Sampled inter-vehicle separation')
    sep.grid(alpha=.25);sep.legend(fontsize=8)
    fig.suptitle(f"{env.get('map_label','')} | {len(names)} UAV | mission acceptance: {summary['flight_acceptance_passed']}")
    fig.savefig(output/'trajectory.png',dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();export(a.run,a.output)


if __name__=='__main__':main()
