"""Fresh frozen old-side cases through the original bounded rollout/oracle owners.

Run serially within the team's worker lease. This is reference execution only;
no candidate build, checkpoint selection, holdout, optimizer or service writes.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import qualify

SOURCES=('experiments/water-residual-learning-v1/audit.py',
         'experiments/water-residual-learning-v1/rollout.py',
         'experiments/water-stability-learning-v1/learned_iteration.py',
         'experiments/water-pressure-accuracy-v1/diagnose.py',
         'experiments/water-stopping-floor-v1/episode.py',
         'experiments/global-pressure-v1/verify_capture.py')


def memory_available():
    return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--bindings',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scene',type=int,action='append',required=True)
    p.add_argument('--cpu',type=int,required=True);p.add_argument('--runtime-receipt',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve();b=qualify.read(a.bindings);reg=b['regression']
    qualify.require(a.cpu in os.sched_getaffinity(0),'CPU outside affinity')
    qualify.require(a.scene==sorted(set(a.scene)) and all(0<=n<74 for n in a.scene),'ordered unique bound cases only')
    qualify.admission(root,b)
    rr=qualify.read(a.runtime_receipt)
    qualify.require(rr['status']=='verified_fixed_installed_runtime' and rr['fresh_inventory'] and rr['full_second_verification']
                    and rr['current_manifest_sha256']==rr['baseline_manifest_sha256']==reg['installed_runtime_manifest']['sha256'],'verified fixed runtime required')
    runtime=qualify.read(rr['current_manifest'])
    out.mkdir(parents=True,exist_ok=False);source=out/'frozen-source';source.mkdir()
    (source/'runs').symlink_to(root/'runs',target_is_directory=True)
    sources=[]
    for relative in SOURCES:
        value=subprocess.check_output(['git','-C',str(root),'show',b['source_commit']+':'+relative])
        path=source/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
        sources.append({'path':relative,'sha256':qualify.sha(value)})
    scenes=json.loads(qualify.bound_bytes(root,reg['membership_source']))['scenes']
    config={'binary':str(root/reg['native_binary']['path']),'scenes':scenes,
            'models':[{'name':'control','project':str(root/reg['project']),'artifacts':str(root/reg['artifacts']),
                       'minimum_network_evaluations':0,'stopping_profile':str((out/'stopping.json').resolve())}],
            'profile':str((out/'stopping.json').resolve()),'native_timeout_seconds':120,
            'oracle_address_space_growth_bytes':1024**3}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');(out/'stopping.json').write_bytes(qualify.canonical(reg['stopping_values']))
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','TULDOK_TORCH_DEVICE':'cpu',
         'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','RAYON_NUM_THREADS':'1','PYTHONPATH':runtime['site_packages']}
    report={'schema':'rheon-migration-old-reproduction-v1','command':sys.argv,'source_sha256':qualify.digest(__file__),
            'source_commit':b['source_commit'],'sources':sources,'bindings_sha256':qualify.digest(a.bindings),
            'runtime_receipt_sha256':qualify.digest(a.runtime_receipt),'runtime_manifest_sha256':qualify.digest(rr['current_manifest']),
            'cpu':a.cpu,'torch_threads':1,'workers':1,'available_memory_reserve_bytes':8*1024**3,
            'native_deadline_seconds':120,'raw_phase_limit_bytes':1024**3,'optimizer_updates':0,
            'records':[],'failures':[],'scope':'fresh old-side reproduction; no migrated-side acceptance'}
    started=time.perf_counter()
    try:
        for index in a.scene:
            qualify.require(memory_available()>=8*1024**3,'available memory reserve')
            qualify.require(sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())<1024**3,'raw phase budget')
            # Preserve the owner's lexical LAB-relative capture paths through this
            # symlink; source/runs resolves to the same primary evidence tree.
            case=out/f'scene-{index:02d}';lexical=source/'runs'/case.relative_to(root/'runs')
            command=[runtime['python'],'-B',str(source/'experiments/water-stopping-floor-v1/episode.py'),
                     '--config',str(out/'config.json'),'--scene',str(index),'--model','control','--output',str(lexical)]
            stamp=time.perf_counter()
            with (out/f'scene-{index:02d}.log').open('xb') as log:
                child=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                                     preexec_fn=lambda:os.sched_setaffinity(0,{a.cpu}),timeout=180)
            entry={'scene':index,'command':command,'exit_status':child.returncode,'elapsed_seconds':time.perf_counter()-stamp,'output':str(case)}
            try:
                qualify.require(child.returncode==0,'old case child failed')
                entry.update(qualify.case_metrics(case,scenes[index],reg['cases'][index],reg['stopping_values'],reg['native_binary']['sha256']))
                qualify.accept_case(reg['cases'][index],entry)
                entry['archived_reproduction_passed']=True
            except (ValueError,KeyError,FileNotFoundError) as exc:
                entry['archived_reproduction_passed']=False;entry['error']=str(exc);report['failures'].append({'scene':index,'error':str(exc)})
            report['records'].append(entry)
            (out/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            if report['failures']:break
    except BaseException as exc:
        report['failures'].append({'error':type(exc).__name__+': '+str(exc)})
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
                      status='failed' if report['failures'] else 'passed',measured_scenes=len(report['records']))
        with (out/'receipt.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({'status':report['status'],'measured_scenes':report['measured_scenes'],'failures':report['failures'],'elapsed_seconds':report['elapsed_seconds']}))
    return 1 if report['failures'] else 0


if __name__=='__main__':raise SystemExit(main())
