"""Export completed runs, check controller provenance, retain sampled decisions."""
import hashlib
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.export_swarm_acceptance import export

spec=importlib.util.spec_from_file_location('motion_analysis',ROOT/'runs/live_navigation_ab_20261123_06/analyze_isolation.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
spec=importlib.util.spec_from_file_location('runtime_analysis',ROOT/'runs/live_navigation_ab_20261123_04/analyze_results.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)


def digest(data):return hashlib.sha256(data).hexdigest()


def sampled_acceleration(csv_path):
    previous={};values={};skipped={}
    for row in csv.DictReader(csv_path.open(encoding='utf-8',newline='')):
        name=row['vehicle']
        current=(int(row['sim_timestamp_ns']),float(row['z_ned_m']),float(row['vx_mps']),float(row['vy_mps']))
        old=previous.get(name);previous[name]=current
        if old is None:continue
        dt=(current[0]-old[0])/1e9
        if not (.05<=dt<=.5 and abs(current[1]+2.)<=.2 and abs(old[1]+2.)<=.2):
            skipped[name]=skipped.get(name,0)+1;continue
        values.setdefault(name,[]).append(math.hypot(current[2]-old[2],current[3]-old[3])/dt)
    return {name:dict(samples=len(v),median_mps2=sorted(v)[(len(v)-1)//2],
        p95_mps2=sorted(v)[math.ceil(.95*len(v))-1],maximum_mps2=max(v),skipped_pairs=skipped.get(name,0))
        for name,v in values.items() if v}


def source_checks(ref,manifest):
    files=manifest['files']
    archive=subprocess.check_output(['git','archive',ref,'--',*files],cwd=ROOT)
    results={}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for name,recorded in files.items():
            blob=tar.extractfile(name).read()
            lf=blob.replace(b'\r\n',b'\n')
            results[name]=recorded in {digest(blob),digest(lf),digest(lf.replace(b'\n',b'\r\n'))}
    if not all(results.values()):raise RuntimeError('Source does not match '+ref)
    return results


def main():
    output=ROOT/'artifacts/live_navigation_ab/2026-11-23-motion-policy'
    output.mkdir(parents=True,exist_ok=False)
    (output/'.gitattributes').write_text('* -text\n',encoding='utf-8',newline='\n')
    cases=[('smoothing_only','06','1dcd791'),('latency_only','07','1dcd791'),
           ('combined_radial','08','172f017'),('combined_swept','09','172f017')]
    rows={};configs={};environments={};owners={}
    for label,suffix,ref in cases:
        folder=ROOT/f'runs/live_navigation_ab_20261123_{suffix}/2_candidate_cam0'
        row=analysis.analyze(folder)
        runtime.ROOT=folder.parent
        full=runtime.case('candidate')
        row.update(source_commit=ref,performance=full['public']['performance_full_run_summaries'],
            observed_window_max_p95=full['public']['observed_window_max_p95'])
        config=yaml.safe_load((folder/'swarm.yaml').read_text(encoding='utf-8'))
        env=full['summary']['environment'];owner=full['ownership']
        row['source_file_checks']=source_checks(ref,env['source_manifest'])
        row['latency_clearance_mode']=config['local_navigation_hardening'].get('latency_clearance_mode','radial')
        row['total_path_m']=sum(full['summary']['path_length_m'].values())
        row['source_manifest_sha256']=env['source_manifest']['sha256']
        destination=output/label
        export(folder/'flight',destination)
        row['sampled_horizontal_acceleration_at_cruise_altitude']=sampled_acceleration(destination/'trajectory.csv')
        for src,dst in [('flight/summary.json','raw_summary.json'),('preflight_ground.json','preflight_ground.json'),
                ('postflight_ground.json','postflight_ground.json'),('swarm.yaml','swarm.yaml'),
                ('swarm.manifest.json','swarm.manifest.json'),('ownership.json','ownership.json')]:
            shutil.copy2(folder/src,destination/dst)
        (destination/'decision_analysis.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        rows[label]=row;configs[label]=config;environments[label]=env;owners[label]=owner
    checks={}
    for pair,names in [('isolated_policies',('smoothing_only','latency_only')),('clearance_geometry',('combined_radial','combined_swept'))]:
        a,b=names;ea,eb=environments[a],environments[b]
        ca,cb=json.loads(json.dumps(configs[a])),json.loads(json.dumps(configs[b]))
        pa,pb=ca.pop('local_navigation_hardening'),cb.pop('local_navigation_hardening')
        if pair=='clearance_geometry':pa.pop('latency_clearance_mode');pb.pop('latency_clearance_mode')
        else:
            for key in ('smoothing','latency_limit'):pa.pop(key);pb.pop(key)
        def formation(env):return next(v for k,v in env['config_sha256'].items() if k.replace('\\','/').endswith('configs/acceptance/formation_2.yaml'))
        checks[pair]=dict(map=ea['map_asset_sha256']==eb['map_asset_sha256'],settings=ea['settings_sha256']==eb['settings_sha256'],
            formation=formation(ea)==formation(eb),source=ea['source_manifest']==eb['source_manifest'],
            non_treatment_config=ca==cb and pa==pb,extra_cameras=owners[a]['extra_camera_count']==owners[b]['extra_camera_count']==0)
        if not all(checks[pair].values()):raise RuntimeError('Unmatched comparison: '+pair)
    # Exact helpers, verified against the file hash recorded before launch.
    helper=(ROOT/'runs/live_navigation_ab_20261123_01/run_live_ab.py').read_text(encoding='utf-8')
    old=helper.replace('memory,recovery,smoothing,latency_limit,passage','memory,recovery,smoothing,passage')
    old=old.replace("    geometry=os.environ.get('NAV_AB_LATENCY_CLEARANCE_MODE','radial')\n",'')
    old=old.replace('prepare(base,config,features,latency_clearance_mode=geometry)','prepare(base,config,features)')
    old=old.replace("    meta['latency_clearance_mode']=geometry\n",'')
    for name,text,owner in [('archived_isolation_harness.py',old,owners['smoothing_only']),('archived_geometry_harness.py',helper,owners['combined_radial'])]:
        if digest(text.encode())!=owner['harness_sha256']:raise RuntimeError('Harness hash mismatch: '+name)
        (output/name).write_text(text,encoding='utf-8',newline='\n')
    summary={label:{k:v for k,v in row.items() if k not in ('motion','source_file_checks','performance','observed_window_max_p95')} for label,row in rows.items()}
    report=dict(evidence_type='serial_dual_uav_exploratory_policy_isolation',cases=summary,matched_conditions=checks,
        all_landed_released=all(row['actual_ground_verified'] and row['reported_cleanup_verified'] for row in rows.values()),
        geometry_duration_relative_change=rows['combined_swept']['flight_duration_s']/rows['combined_radial']['flight_duration_s']-1,
        limitations=['One trial per policy; no randomized repeated comparison or statistical superiority claim',
            'Isolated-policy and geometry pairs use different source commits; compare within each matched pair',
            'Navigation decisions are sampled about once per second, not duration fractions or full frame counts',
            'No added camera streams; built-in UE subwindows unchanged; only this dual-UAV maze case',
            'Nominal braking heuristic does not certify observed free space or actual vehicle braking',
            'Window P95 maxima and legacy hover counters are not global percentiles or actual hovering'])
    report['limitations'].append('Acceleration is finite-difference of sampled measured velocities at z=-2 +/-0.2m using simulator time; not command acceleration, jerk, or a dynamics bound')
    (output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 双机限速与平滑拆分试验','',
        '2026-11-23，本机 AirSim/Blocks maze，固定 settings_2.json、formation_2.yaml、0.35米/秒速度上限、0.6秒时效和2米机间距离要求；不增加前端相机流。每轮独立启动且串行运行，所有安全门控保留。','',
        '| 试验 | 控制代码 | 任务验收 | 飞行时间（秒） | 最小机间距离（米） | 记录碰撞数 |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for name,row in rows.items():
        lines.append(f"| {name} | {row['source_commit']} | {'通过' if row['passed'] else '未通过'} | {row['flight_duration_s']:.2f} | {row['minimum_pair_distance_m']:.3f} | {row['collision_count']} |")
    lines+=['','## 结果与边界','',
        '前两轮只改变 smoothing/latency_limit 开关；后两轮四项策略和延迟限速均开启，只改变 radial/swept 几何模式。两对使用不同提交，应分别比较，不能将四轮当成同一提交的完整因子试验。',
        f"几何模式这一对，swept 相对 radial 的飞行时间变化为 {report['geometry_duration_relative_change']*100:+.1f}%。每项只有一轮、未随机化顺序，不能证明统计显著改进或普遍优越性。",
        '最近距离算法会用侧方/后方障碍的距离作为前向制动预算；swept 候选改为当前行进方向进入缓冲障碍的距离，并在平滑改变方向后再检查一次。它是名义几何限速，不是完整动力学制动控制器，也不证明未观测空间可通行。',
        '新模式仍需显式选择；默认保留 radial，新导航策略主开关仍默认关闭。未更改共享验收配置、硬净空、时效、机间距离、速度上限或碰撞门控。',
        'comparison.json 保存配对条件一致性检查；每轮 decision_analysis.json 保存源码逐文件核验、每秒抽样的速度决策和运行性能汇总。抽样计数不是实际限速时长、比例或全程逐帧计数，轻量模式旧 hover 来源计数也不是物理悬停次数。',
        '原始结果保留在 raw_summary.json，派生汇总、完整位置轨迹、导航反馈及图由既有导出器生成；图中仅标出柱体中心。preflight_ground.json/postflight_ground.json 是实际 RPC 地面与控制权检查。源码核验接受 LF/CRLF 检出差异。',
        'archived_* helper 保留审计流程与当时本机路径，不能直接在 artifacts 中运行；启动前必须配置正确的 SDK、UE 和工程路径，使用新的 runs 输出目录，并遵守既有预检及清理流程。',
        '不覆盖五机、前端相机负载、所有地图、交叉会车或其他电脑。上轮失败记录保留在 ../2026-11-23/pre_fix_timeout。','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    shutil.copy2(Path(__file__),output/'archived_export.py')
    shutil.copy2(ROOT/'runs/live_navigation_ab_20261123_06/analyze_isolation.py',output/'archived_motion_analysis.py')
    shutil.copy2(ROOT/'runs/live_navigation_ab_20261123_04/analyze_results.py',output/'archived_runtime_analysis.py')
    (output/'evidence_sha256.json').write_text(json.dumps({str(p.relative_to(output)).replace('\\','/'):digest(p.read_bytes())
        for p in output.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=summary,matched_conditions=checks,geometry_duration_relative_change=report['geometry_duration_relative_change']),ensure_ascii=False))


if __name__=='__main__':main()
