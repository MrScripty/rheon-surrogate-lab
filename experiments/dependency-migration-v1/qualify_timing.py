"""Separate advisory timing through frozen rollout owners; no performance gate.

Run each side in a fresh process. Startup-only Python function stamps are removed
before neural inference. Their measured scopes are explicit and may overlap;
no sum is represented as an uninstrumented startup decomposition.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time
import qualify


def describe(values):
    return {'samples':len(values),'median_seconds':statistics.median(values),
            'minimum_seconds':min(values),'maximum_seconds':max(values),
            'mean_seconds':statistics.mean(values),'pstdev_seconds':statistics.pstdev(values)} if values else {'samples':0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--side',choices=('old','new'),required=True)
    for n in ('root','bindings','binary','manifest','old-source','candidate-cohort','old-cases','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--scene',choices=(0,13,16),type=int,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    receipt={'command':sys.argv,'source_sha256':qualify.digest(__file__),'side':a.side,'scene':a.scene,
             'binary_sha256':qualify.digest(a.binary),'comparison_manifest_sha256':qualify.digest(a.manifest),
             'cpu_affinity':sorted(os.sched_getaffinity(0)),'optimizer_updates':0,'performance_acceptance':False,
             'python':sys.executable,'python_version':sys.version,
             'environment':{key:os.environ.get(key) for key in ('PYTHONDONTWRITEBYTECODE','CUDA_VISIBLE_DEVICES','TULDOK_TORCH_DEVICE','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS')},
             'native_deadline_seconds':120,'memory_reserve_bytes':8*1024**3}
    clocks={};counts={};active={}
    try:
        b=qualify.read(a.bindings);reg=b['regression'];old=qualify.read(a.old_cases)
        accepted=qualify.read(a.candidate_cohort)
        qualify.require(accepted['status']=='passed' and accepted['result']['passed'] and accepted['result']['cases']==74,'full current74 correctness required before timing')
        scene=json.loads(qualify.bound_bytes(a.root,reg['membership_source']))['scenes'][a.scene]
        qualify.require(len(os.sched_getaffinity(0))==1,'one explicit CPU/core for serial timings')
        if a.side=='old':
            source=a.old_source.resolve();binary=a.root/reg['native_binary']['path']
            qualify.require(qualify.digest(binary)==reg['native_binary']['sha256'],'old timing binary mismatch')
            original=qualify.read(source.parent/'receipt.json')
            qualify.require(original['status']=='passed' and original['source_commit']==b['source_commit'],'frozen baseline source witness required')
            source_hashes={str(source/item['path']):item['sha256'] for item in original['sources']}
        else:
            source=Path(__file__).resolve().parents[2];binary=a.binary.resolve()
            build=qualify.read(binary.parent/'build-receipt.json')
            qualify.require(build['status']=='qualified_build' and build['binary_sha256']==qualify.digest(binary),'candidate qualified build required')
            source_hashes=accepted['active_sources']
        qualify.require(all(qualify.digest(path)==value for path,value in source_hashes.items()),'timing owner source changed')
        receipt['source_witnesses']=source_hashes
        available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
        qualify.require(available>=8*1024**3,'8 GiB memory reserve required')
        receipt['available_memory_before_bytes']=available
        receipt.update(binary_sha256=qualify.digest(binary),adapter_manifest_sha256=qualify.digest(a.manifest) if a.side=='new' else None,weights_sha256=reg['weights']['sha256'])
        sys.path.insert(0,str(source/'experiments/water-residual-learning-v1'))
        import rollout
        def label(frame):
            name=frame.f_code.co_name;filename=frame.f_code.co_filename
            if name=='verify' and filename.endswith('/torch_runtime.py'):return 'installed_runtime_verify'
            if name=='load' and filename.endswith('/torch_runtime.py'):return 'runtime_manifest_load'
            if name=='measure' and filename.endswith('/model_adapter.py'):return 'package_or_interpreter_file_measure'
            if name=='sha' and str(a.root/reg['project']) in filename:return 'old_model_material_sha_compute'
            if name=='load_state_dict':return 'torch_weight_state_restore'
            if name=='load_weights' and str(a.root/reg['project']) in filename:return 'old_weight_load_and_restore'
            if name=='initialize' and filename.endswith('/audit.py'):return 'old_total_model_startup'
            if name=='load' and filename.endswith('/model_adapter.py'):return 'new_total_model_startup'
            return None
        def profile(frame,event,arg):
            key=label(frame)
            if key is None:return
            if event=='call':active[id(frame)]=(key,time.perf_counter())
            elif event=='return' and id(frame) in active:
                measured,t=active.pop(id(frame));clocks[measured]=clocks.get(measured,0.)+time.perf_counter()-t;counts[measured]=counts.get(measured,0)+1
                if measured in ('old_total_model_startup','new_total_model_startup'):sys.setprofile(None)
        sys.setprofile(profile)
        try:
            if a.side=='old':
                result=rollout.run(binary,scene,a.output/'native',a.root/reg['project'],a.root/reg['artifacts'],
                                   timeout=120,proposals=32,stopping=reg['stopping_values'],single_evaluation=True,threads=1,minimum_network_evaluations=0)
            else:
                result=rollout.run(binary,scene,a.output/'native',adapter_manifest=a.manifest,timeout=120,threads=1)
        finally:sys.setprofile(None)
        qualify.require(result['status']=='completed' and result['native_process']['reaped'] and not result['result']['hard_stop'],'timing child/physics failed')
        frames=[json.loads(s) for s in (a.output/'native/frames.jsonl').read_text().splitlines()]
        frames=[f for f in frames if f.get('ai_quality')]
        callbacks=[json.loads(s) for s in (a.output/'native/callbacks.jsonl').read_text().splitlines()]
        calls=[c['network_passes'] for c in callbacks]
        qualify.require(len(frames)==len(callbacks)==scene['steps']-15 and all(f['ai_quality']['qualified'] and f['reference_quality']['qualified'] for f in frames),'incomplete timing physical horizon')
        metrics={'calls_sha256':qualify.sha(qualify.canonical(calls)),'mean_calls':statistics.mean(calls),'maximum_calls':max(calls),
                 'worst_paired_pressure_relative_l2':max(f['relative_pressure_difference'] for f in frames)}
        if a.side=='old':qualify.accept_case(reg['cases'][a.scene],metrics)
        else:qualify.accept_case(reg['cases'][a.scene],old[a.scene],metrics)
        qualify.require(all(qualify.digest(path)==value for path,value in source_hashes.items())
                        and qualify.digest(binary)==receipt['binary_sha256'],'timing owner source/binary drift')
        receipt.update(status='passed',scene_configuration=scene,model_calls=calls,quality_metrics=metrics,
                       first_ai={'step':frames[0]['step'],'native_full_step_seconds':frames[0]['reference_seconds'],'ai_full_step_seconds':frames[0]['ai_seconds'],'model_callback_seconds':callbacks[0]['model_correction_seconds']},
                       steady={'native_full_step':describe([f['reference_seconds'] for f in frames[1:]]),
                               'ai_full_step':describe([f['ai_seconds'] for f in frames[1:]]),
                               'model_callback':describe([c['model_correction_seconds'] for c in callbacks[1:]])},
                       startup_function_seconds=clocks,startup_function_counts=counts,torch_threads=1,cuda_initialized=result['cuda_initialized'],
                       phase_elapsed_seconds=result['elapsed_s'],adapter_loading_seconds=result.get('adapter_loading_seconds'),
                       cache_status='uncontrolled warm OS cache; matched serial order reverses on second repetition',
                       instrumentation='Python function stamps only during startup; disabled at model startup return, before all network forwards. Function scopes may overlap.',
                       clock_scopes={'installed_runtime_verify':'current owner topology and complete installed-asset verification; absent from old local audit owner startup',
                                     'package_or_interpreter_file_measure':'current adapter package/interpreter byte-hash calls, excluding installed runtime owner checks',
                                     'old_model_material_sha_compute':'captured source/artifact SHA computation only; file I/O and interpreter imports are in total startup',
                                     'torch_weight_state_restore':'load_state_dict; excludes NPZ read and network construction',
                                     'old_weight_load_and_restore':'captured weight helper including NPZ read and tensor restore',
                                     'full_step':'Rheon engine advancement with projection/preparation and model IPC where applicable; not isolated pressure-kernel timing'})
    except BaseException as exc:
        receipt.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        sys.setprofile(None)
        receipt.update(elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       peak_native_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024)
        with (a.output/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps({'status':receipt['status'],'side':a.side,'scene':a.scene,'elapsed_seconds':receipt['elapsed_seconds']}))


if __name__=='__main__':main()
