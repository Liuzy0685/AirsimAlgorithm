"""Prepare an instrumented baseline or opt-in trial; never edits shared settings."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import yaml
from coordination.navigation_hardening import HardeningConfig
from coordination.runtime_config import load_config


def prepare(base, output, features):
    base,output=Path(base).resolve(),Path(output).resolve()
    if base==output:
        raise ValueError('trial must not overwrite the shared baseline')
    if output.exists():
        raise ValueError('trial output already exists; use a new run directory')
    config,_=load_config(base)
    if not config.get('navigation',{}).get('enabled'):
        raise ValueError('trial requires an existing navigation.enabled baseline')
    if set(features)-{'memory','recovery','smoothing','passage'}:
        raise ValueError('unknown trial feature')
    policy=asdict(HardeningConfig(enabled=True))
    for name in ('memory','recovery','smoothing','passage'):
        policy[name]=name in features
    config['local_navigation_hardening']=policy
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf-8')
    manifest=dict(base_path=str(base),base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        trial_path=str(output),trial_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        features=sorted(features),note='Only local_navigation_hardening differs; settings, formation and safety thresholds are unchanged')
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--features',nargs='*',choices=('memory','recovery','smoothing','passage'),default=[])
    args=parser.parse_args()
    try:print(json.dumps(prepare(args.base,args.output,args.features),ensure_ascii=False))
    except ValueError as exc:parser.error(str(exc))


if __name__=='__main__':main()
