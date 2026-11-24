"""Descriptive report for repeated trials; no universal superiority claim."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import shutil
import statistics
import sys

import yaml

ROOT=Path(__file__).resolve().parents[2]
RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.export_swarm_acceptance import export

spec=importlib.util.spec_from_file_location('geometry_archive',ROOT/'runs/live_navigation_ab_20261123_08/archive_geometry.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def reported_goal_dwell(folder):
    result={}
    for path in (folder/'flight/runtime').glob('Drone*.log'):
        values=[]
        for line in path.read_text(encoding='utf-8').splitlines():
            if 'goal_termination ' not in line or 'reached=true' not in line:continue
            match=re.search(r'\bdwell=([0-9.]+)',line)
            if match:values.append(float(match.group(1)))
        result[path.stem]=max(values,default=None)
    return result


def main():
    plan=load(RUN/'plan.json');run_report=load(RUN/'report.json')
    output=ROOT/'artifacts/live_navigation_ab/2026-11-23-repeated-maze'
    output.mkdir(parents=True,exist_ok=False)
    (output/'.gitattributes').write_text('* -text\n',encoding='utf-8',newline='\n')
    cases=[];configs={};envs={}
    for result in run_report['cases']:
        folder=Path(result['summary']).parents[1]
        name=f"pair_{result['pair']:02d}_{result['geometry']}"
        destination=output/name
        export(folder/'flight',destination)
        row=helpers.analysis.analyze(folder)
        helpers.runtime.ROOT=folder.parent
        full=helpers.runtime.case('candidate')
        environment=full['summary']['environment']
        row.update(pair=result['pair'],geometry=result['geometry'],source_commit=plan['source_commit'],
            source_file_checks=helpers.source_checks(plan['source_commit'],environment['source_manifest']),
            performance=full['public']['performance_full_run_summaries'],
            observed_window_max_p95=full['public']['observed_window_max_p95'],
            total_path_m=sum(full['summary']['path_length_m'].values()),
            maximum_reported_goal_dwell_s=reported_goal_dwell(folder),
            sampled_horizontal_acceleration=helpers.sampled_acceleration(destination/'trajectory.csv'))
        for src,dst in [('flight/summary.json','raw_summary.json'),('preflight_ground.json','preflight_ground.json'),
                ('postflight_ground.json','postflight_ground.json'),('swarm.yaml','swarm.yaml'),
                ('swarm.manifest.json','swarm.manifest.json'),('ownership.json','ownership.json')]:
            shutil.copy2(folder/src,destination/dst)
        (destination/'decision_analysis.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        if environment['source_manifest']!=plan['source_manifest']:raise RuntimeError('Source differs from frozen plan')
        config_hashes={k.replace('\\','/'):v for k,v in environment['config_sha256'].items()}
        if environment['map_asset_sha256']!=plan['map_sha256'] or config_hashes.get('configs/acceptance/settings_2.json')!=plan['settings_sha256']:
            raise RuntimeError('Map/settings differs from frozen plan')
        owner=load(folder/'ownership.json')
        if owner['harness_sha256']!=plan['harness_sha256']:raise RuntimeError('Harness changed')
        config=yaml.safe_load((folder/'swarm.yaml').read_text(encoding='utf-8'))
        cases.append({k:v for k,v in row.items() if k not in ('motion','performance','observed_window_max_p95','source_file_checks')})
        configs[name]=config;envs[name]=environment
    pairs=[]
    for number in range(1,4):
        members={r['geometry']:r for r in cases if r['pair']==number}
        if set(members)!={'radial','swept'}:continue
        a=f'pair_{number:02d}_radial';b=f'pair_{number:02d}_swept'
        ca=json.loads(json.dumps(configs[a]));cb=json.loads(json.dumps(configs[b]))
        ga=ca['local_navigation_hardening'].pop('latency_clearance_mode')
        gb=cb['local_navigation_hardening'].pop('latency_clearance_mode')
        def formation(env):return next(v for k,v in env['config_sha256'].items() if k.replace('\\','/').endswith('configs/acceptance/formation_2.yaml'))
        checks=dict(non_treatment_config=ca==cb,geometry=ga=='radial' and gb=='swept',
            source=envs[a]['source_manifest']==envs[b]['source_manifest'],
            runtime_settings=envs[a]['settings_sha256']==envs[b]['settings_sha256'],
            formation=formation(envs[a])==formation(envs[b])==plan['formation_sha256'])
        if not all(checks.values()):raise RuntimeError('Pair mismatch: '+str(number))
        radial=members['radial']['flight_duration_s'];swept=members['swept']['flight_duration_s']
        both_passed=members['radial']['passed'] and members['swept']['passed']
        pairs.append(dict(pair=number,order=plan['pair_orders'][number-1],radial_s=radial,swept_s=swept,
            radial_passed=members['radial']['passed'],swept_passed=members['swept']['passed'],
            relative_change=swept/radial-1 if both_passed else None,conditions_verified=checks))
    report=dict(evidence_type='three_additional_serial_matched_pairs_dual_maze',source_commit=plan['source_commit'],
        planned_pairs=3,completed_cases=len(cases),completed_pairs=len(pairs),cases=cases,pairs=pairs,
        all_passed=bool(cases) and all(r['passed'] for r in cases),
        all_cleanup_verified=bool(cases) and all(r['actual_ground_verified'] and r['reported_cleanup_verified'] for r in cases),
        collision_count=sum(r['collision_count'] for r in cases),
        minimum_pair_distance_m=min((r['minimum_pair_distance_m'] for r in cases),default=None),
        suite_failure=run_report.get('failure'),all_planned_executed=len(cases)==6,
        driver_completed_without_failure=run_report.get('all_planned_cases_completed',False),
        limitations=['Only this two-UAV maze scenario; no added frontend camera streams',
            'Run-order seed only, no AirSim physics randomization seed',
            'Three new pairs, earlier exploratory pilot excluded from aggregate; descriptive evidence only',
            'No statistical or universal safety/performance claim; no comparison against original baseline',
            'Sampled acceleration and pairwise separation are not continuous-time dynamics/safety bounds'])
    report['limitations'].append('Maximum logged goal dwell is not total physical hover or exact formation-wait duration')
    report['time_comparison_conclusion']='blocked_by_task_failure' if not report['all_passed'] else 'descriptive_only'
    report['success_counts']={g:dict(passed=sum(r['passed'] for r in cases if r['geometry']==g),
        attempted=sum(r['geometry']==g for r in cases)) for g in ('radial','swept')}
    # A timeout is censored runtime, not a completed task's duration. Do not
    # remove the failed candidate and advertise a survivor-only speed gain.
    if pairs and report['all_passed']:
        report['median_paired_relative_change']=statistics.median(p['relative_change'] for p in pairs)
    (output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    shutil.copy2(RUN/'plan.json',output/'plan.json');shutil.copy2(RUN/'report.json',output/'raw_runner_report.json')
    for source,name,expected in [(RUN/'run_repeated.py','archived_runner.py',plan['runner_sha256']),
            (ROOT/'runs/live_navigation_ab_20261123_01/run_live_ab.py','archived_harness.py',plan['harness_sha256'])]:
        if sha(source)!=expected:raise RuntimeError('Runner/helper hash mismatch')
        shutil.copy2(source,output/name)
    shutil.copy2(Path(__file__),output/'archived_export.py')
    for source,name in [(ROOT/'runs/live_navigation_ab_20261123_08/archive_geometry.py','archived_geometry_analysis.py'),
            (ROOT/'runs/live_navigation_ab_20261123_06/analyze_isolation.py','archived_motion_analysis.py'),
            (ROOT/'runs/live_navigation_ab_20261123_04/analyze_results.py','archived_runtime_analysis.py')]:
        shutil.copy2(source,output/name)
    lines=['# maze 双机方向限速重复对照','',
        '控制代码固定为 '+plan['source_commit']+'。新增三组成对试验，先后顺序记录在 plan.json；顺序随机种子不等于 AirSim 物理随机种子。此前探索性单次结果不计入本次汇总。',
        '所有轮次开启记忆、脱困、平滑、延迟限速和通行建议，只改变 radial/swept 几何模式。使用同一 maze、settings_2.json、formation_2.yaml、0.35米/秒速度上限、0.6秒时效、2米机间距离要求，无额外前端相机流。','',
        '| 配对 | 顺序 | radial 飞行记录秒数 | swept 飞行记录秒数 | 成功任务耗时变化 |',
        '| --- | --- | ---: | ---: | ---: |']
    for p in pairs:
        radial_text=f"{p['radial_s']:.2f}"+('（未通过）' if not p['radial_passed'] else '')
        swept_text=f"{p['swept_s']:.2f}"+('（未通过）' if not p['swept_passed'] else '')
        change=f"{p['relative_change']*100:+.1f}%" if p['relative_change'] is not None else '不比较：任务未通过'
        lines.append(f"| {p['pair']} | {' → '.join(p['order'])} | {radial_text} | {swept_text} | {change} |")
    lines+=['','## 验收与解释','',
        f"计划轮次全部执行：{report['all_planned_executed']}；执行{len(cases)}轮/{len(pairs)}对；任务全部通过：{report['all_passed']}；实际落地及控制权释放全部确认：{report['all_cleanup_verified']}；记录碰撞合计：{report['collision_count']}。",
        f"观测到的最小机间距离：{report['minimum_pair_distance_m']}米。逐轮任务终止原因和清理结果保留在原始汇总中。"]
    if 'median_paired_relative_change' in report:
        lines.append(f"配对时间变化的中位数：{report['median_paired_relative_change']*100:+.1f}%。这些是描述性结果，不能声称统计显著、普遍更优或已优于原始基线。")
    else:
        lines.append('本轮出现任务失败，阻止正向性能结论。超时记录不是完成任务耗时，不能剔除失败轮后计算速度提升。radial 通过3/3，swept 通过2/3，仅代表本次小样本；不能据此保证 radial 总是成功。')
    lines+=['',
        '每轮独立启动 UE，确认落地、解除武装和 API 释放后退出实例，再启动下一轮。发生验收/安全失败或清理/退出未确认时，驱动会停止后续自动起飞。未改共享验收配置，也未放宽点云过期、净空或碰撞门控。',
        'comparison.json 保存逐对条件检查与任务指标；decision_analysis.json 保存逐源码核验、抽样速度决策及运行时延。抽样速度决策计数不是实际限速时长；滑动窗口 P95 最大值不是全程 P95；采样速度差分也不是指令加速度或连续时间安全证明。',
        'raw_summary.json 保留原始汇总，trajectory.csv/trajectory.png 为既有导出器输出的观测轨迹，图中仅标出柱体中心。地图、settings、编队和完整控制源码清单均核对；源码允许 LF/CRLF 检出差异。',
        '仍只覆盖本机这一双机 maze 场景，未覆盖五机、相机负载、窄通道会车和其他电脑。默认策略和模式保持原样；进一步扩大测试前应同时评价任务时间、实际运动变化、队形恢复与安全。',
        'archived_* 保留审计流程和当时本机路径，不要直接在 artifacts 中启动飞行；复现应配置本机 SDK/UE 路径，复制启动流程到新的 runs 目录，遵守预检及清理规则。','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    (output/'evidence_sha256.json').write_text(json.dumps({p.relative_to(output).as_posix():sha(p)
        for p in sorted(output.rglob('*')) if p.is_file()},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False))


if __name__=='__main__':main()
