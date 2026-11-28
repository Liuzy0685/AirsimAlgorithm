"""Serve A's local operator tasks API; existing flight controllers keep all authority."""
import argparse
from pathlib import Path
import secrets
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from coordination.operator_common import RuntimeDirectoryLock
from coordination.operator_http import OperatorHTTPServer
from coordination.operator_manager import MissionManager
from coordination.operator_runtime import AirSimPreflight, OperatorProfile, SubprocessMissionBackend


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('maze_2','maze_5','flying_2','flying_5'),
                        help='Select the shared settings/swarm/formation/map bundle')
    parser.add_argument('--settings-json')
    parser.add_argument('--swarm-config')
    parser.add_argument('--formation-config')
    parser.add_argument('--map-manifest')
    parser.add_argument('--blocks-project',required=True)
    parser.add_argument('--engine-log',help='Log of the selected UE instance, including -abslog paths')
    parser.add_argument('--runtime-dir',default=str(ROOT/'runs/operator'))
    parser.add_argument('--port',type=int,default=8790)
    parser.add_argument('--allow-origin',action='append',default=[])
    parser.add_argument('--allow-flight',action='store_true',help='Enable explicitly requested task launches')
    parser.add_argument('--stop-file',help='Cooperative API shutdown; active jobs still clean up normally')
    args=parser.parse_args()
    from coordination.operator_common import read_json
    fields=('settings_json','swarm_config','formation_config','map_manifest')
    if args.profile:
        if any(getattr(args,key) is not None for key in fields):
            parser.error('--profile cannot be mixed with individual configuration overrides')
        bundle=read_json(ROOT/'configs/operator/profiles.json')[args.profile]
        for key in fields:setattr(args,key,str(ROOT/bundle[key]))
    elif not all(getattr(args,key) for key in fields):
        parser.error('Provide --profile, or all four explicit configuration paths')
    directory=Path(args.runtime_dir).resolve()
    lock=RuntimeDirectoryLock(directory)
    server=None
    manager=None
    try:
        token_path=directory/'operator.token'
        if not token_path.exists():
            token_path.write_text(secrets.token_urlsafe(32),encoding='utf-8')
        token=token_path.read_text(encoding='utf-8').strip()
        profile=OperatorProfile(args.settings_json,args.swarm_config,args.formation_config,args.map_manifest,
                                blocks_project=args.blocks_project,allow_flight=args.allow_flight,
                                engine_log=args.engine_log)
        manager=MissionManager(profile,directory,AirSimPreflight(profile),SubprocessMissionBackend(profile))
        server=OperatorHTTPServer(('127.0.0.1',args.port),manager,token,args.allow_origin)
        threading.Thread(target=manager.monitor,daemon=True).start()
        if args.stop_file:
            def stop_when_requested():
                while not manager.closed.wait(.2):
                    if Path(args.stop_file).exists():
                        server.shutdown()
                        return
            threading.Thread(target=stop_when_requested,daemon=True).start()
        print(f'A operator API http://127.0.0.1:{server.server_port}/api/v1 ; credential file: {token_path}',flush=True)
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        if manager and not manager.shutdown():
            print('Controllers are still cleaning up. They were left running; inspect runtime before any new flight.',flush=True)
        if server:
            server.server_close()
        lock.close()


if __name__=='__main__':
    import signal
    if hasattr(signal,'SIGBREAK'):
        signal.signal(signal.SIGBREAK,signal.default_int_handler)
    main()
