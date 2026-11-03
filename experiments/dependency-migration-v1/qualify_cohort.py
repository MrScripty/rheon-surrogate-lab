"""Serial candidate worker; two workers may share one bounded qualification phase."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import qualify


def admit_runtime_receipt(receipt):
    qualify.require(receipt['all_original_fields_and_asset_bytes_identical'] is True
                    and receipt['full_second_verification'] is True,
                    'unchanged numerical installation must be independently verified')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','bindings','binary','manifest','q4','old-cohort','old-cases','runtime-receipt','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--scene',type=int,action='append',required=True);p.add_argument('--cpu',type=int,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    sources={str(Path(__file__).resolve()):qualify.digest(__file__),
             str(Path(qualify.__file__).resolve()):qualify.digest(qualify.__file__)}
    runner=Path(__file__).with_name('qualify_candidate.py').resolve();sources[str(runner)]=qualify.digest(runner)
    b=qualify.read(a.bindings);old=qualify.read(a.old_cases)
    qualify.require(a.scene==sorted(set(a.scene)) and all(0<=i<74 for i in a.scene),'ordered bound consumed cases required')
    qualify.require(a.cpu in os.sched_getaffinity(0),'CPU outside affinity')
    qualify.require(qualify.read(a.old_cohort)['status']=='passed' and qualify.read(a.old_cohort)['result']['cases']==74,'fresh old74 admission required')
    qualify.compare_cohorts(b,old)
    runtime_receipt=qualify.read(a.runtime_receipt);runtime=qualify.read(runtime_receipt['current_manifest'])
    admit_runtime_receipt(runtime_receipt)
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','TULDOK_TORCH_DEVICE':'cpu',
         'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','RAYON_NUM_THREADS':'1','PYTHONPATH':runtime['site_packages']}
    report={'schema':'rheon-migration-current-cohort-worker-v1','command':sys.argv,'sources':sources,
            'bindings_sha256':qualify.digest(a.bindings),'old_cases_sha256':qualify.digest(a.old_cases),
            'old_cohort_sha256':qualify.digest(a.old_cohort),'runtime_receipt_sha256':qualify.digest(a.runtime_receipt),
            'binary_sha256':qualify.digest(a.binary),'manifest_sha256':qualify.digest(a.manifest),
            'cpu':a.cpu,'threads':1,'workers':1,'native_deadline_seconds':120,'raw_phase_budget_bytes':1024**3,
            'memory_reserve_bytes':8*1024**3,'optimizer_updates':0,'records':[],'failures':[],
            'cache_status':'uncontrolled warm OS cache following installed-runtime verification; correctness only',
            'environment':{k:env[k] for k in ('PYTHONDONTWRITEBYTECODE','CUDA_VISIBLE_DEVICES','TULDOK_TORCH_DEVICE','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS')}}
    stop=a.output.parent/'candidate-stop.json'
    try:
        for index in a.scene:
            if stop.exists():
                report['stopped_by_peer']=qualify.read(stop);break
            available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
            qualify.require(available>=8*1024**3,'8 GiB available memory reserve')
            qualify.require(sum(p.stat().st_size for p in a.output.parent.rglob('*') if p.is_file())<1024**3,'1 GiB raw phase budget')
            qualify.require(all(qualify.digest(p)==h for p,h in sources.items()),'qualification source changed during campaign')
            out=a.output/f'scene-{index:02d}'
            command=[runtime['python'],'-B',str(runner),*sum(([f'--{n}',str(getattr(a,n.replace('-','_')).resolve())] for n in ('root','bindings','binary','manifest','q4','old-cohort')),[]),
                     '--scene',str(index),'--output',str(out.resolve())]
            stamp=time.perf_counter()
            with (a.output/f'scene-{index:02d}.log').open('xb') as log:
                child=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=lambda:os.sched_setaffinity(0,{a.cpu}),timeout=300)
            entry={'scene':index,'command':command,'exit_status':child.returncode,'elapsed_seconds':time.perf_counter()-stamp,'output':str(out.resolve()),'available_memory_before_bytes':available}
            try:
                qualify.require(child.returncode==0,'candidate case child failed')
                measured=qualify.read(out/'receipt.json');qualify.require(measured['status']=='passed','candidate scientific case failed')
                entry.update(measured['metrics'])
                entry['native_control']={'scene':index,**qualify.native_metrics(out/'native-control',qualify.read(out/'native/receipt.json')['scene'],report['binary_sha256'])}
                entry['comparison']={'archived_pressure_max':b['regression']['cases'][index]['worst_paired_pressure_relative_l2'],
                                     'reproduced_old_pressure_max':old[index]['worst_paired_pressure_relative_l2'],
                                     'old_mean_calls':old[index]['mean_calls'],'old_maximum_calls':old[index]['maximum_calls']}
                qualify.accept_case(b['regression']['cases'][index],old[index],entry)
                entry['migration_case_passed']=True
            except (ValueError,KeyError,FileNotFoundError) as exc:
                entry['migration_case_passed']=False;entry['error']=str(exc);report['failures'].append({'scene':index,'error':str(exc)})
                try:
                    with stop.open('x') as stream:json.dump({'scene':index,'reason':str(exc),'worker_output':str(a.output)},stream,indent=2)
                except FileExistsError:pass
            report['records'].append(entry)
            (a.output/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            if report['failures']:break
    except BaseException as exc:
        report['failures'].append({'error':type(exc).__name__+': '+str(exc)});raise
    finally:
        report.update(status='failed' if report['failures'] or report.get('stopped_by_peer') else 'passed',
                      elapsed_seconds=time.perf_counter()-started,peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024)
        with (a.output/'receipt.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({'status':report['status'],'cases':len(report['records']),'failures':report['failures'],'elapsed_seconds':report['elapsed_seconds']}))
    return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
