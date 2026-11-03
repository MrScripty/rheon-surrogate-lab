"""Own a bounded public diagnostic trajectory through the current CPU adapter.

The package owns numerical preprocessing, reconstruction and stopping. Rejected
attempts remain raw diagnostics and never become accepted simulation history.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import audit

LIMIT=8*1024*1024


def messages(child,deadline):
    buffer=b''
    with selectors.DefaultSelector() as selector:
        selector.register(child.stdout,selectors.EVENT_READ)
        while time.monotonic()<deadline:
            if not selector.select(min(1.,max(0.,deadline-time.monotonic()))):continue
            raw=os.read(child.stdout.fileno(),65536)
            if not raw:
                if buffer:raise ValueError('incomplete native message')
                return
            buffer+=raw
            if len(buffer)>LIMIT:raise ValueError('native message bound')
            while b'\n' in buffer:
                line,buffer=buffer.split(b'\n',1)
                yield json.loads(line)
    raise TimeoutError('native trajectory deadline')


def run(binary,scene,output,adapter_manifest=None,control=False,timeout=120,threads=1,observe=None):
    affinity=sorted(os.sched_getaffinity(0))
    if type(threads) is not int or threads not in (1,4,8) or threads>len(affinity):
        raise ValueError('one, four or eight CPU threads within explicit affinity required')
    if not control and (adapter_manifest is None or threads!=1):
        raise ValueError('current adapter manifest and one CPU thread required')
    if control and adapter_manifest is not None:
        raise ValueError('native algorithm control does not use a model package')
    adapter=None;state=None;loading_seconds=0.
    os.environ.update(CUDA_VISIBLE_DEVICES='',TULDOK_TORCH_DEVICE='cpu')
    if not control:
        adapter_manifest=Path(adapter_manifest).resolve(strict=True)
        spec=importlib.util.spec_from_file_location('rheon_current_rollout_adapter',adapter_manifest.parent/'model_adapter.py')
        adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter
        spec.loader.exec_module(adapter)
        started_loading=time.perf_counter();state=adapter.load(adapter_manifest)
        loading_seconds=time.perf_counter()-started_loading
        # Existing independent oracle helpers consume these numerical owners only.
        audit.TORCH=state.torch;audit.MODEL=state.model;audit.NETWORK=state.network;audit.NORM=state.normalization
        os.sched_setaffinity(0,set(affinity))
    output.mkdir(parents=True,exist_ok=False)
    (output/'scene.json').write_text(json.dumps(scene,indent=2)+'\n')
    environment={**os.environ,'OMP_NUM_THREADS':str(threads),'MKL_NUM_THREADS':str(threads),'RAYON_NUM_THREADS':str(threads),'CUDA_VISIBLE_DEVICES':'','TULDOK_TORCH_DEVICE':'cpu'}
    command=[str(binary),*(['--reference-control'] if control else ['--adapter-inputs',json.dumps(state.manifest['inputs'],sort_keys=True,separators=(',',':')),'--adapter-output-role',state.manifest['output']['role']])]
    started=time.perf_counter();complete=None;frames=[];attempts=[];error=None;terminated=False;inference_error=None
    with (output/'native-stderr.log').open('wb') as stderr:
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,env=environment,start_new_session=True)
        try:
            child.stdin.write(json.dumps(scene).encode()+b'\n');child.stdin.flush()
            for value in messages(child,time.monotonic()+timeout):
                if value['kind']=='predict':
                    if adapter is None:raise ValueError('native control unexpectedly requested model inference')
                    trace=[];network_calls=[];inference_start=time.perf_counter()
                    hook=state.network.register_forward_hook(lambda *ignored:network_calls.append(1))
                    try:
                        candidate=adapter.predict(value,state,trace)
                        expected=trace[-1]['proposals'] if trace else len(network_calls)
                        if len(network_calls)!=expected or len(network_calls)>state.policy['max_network_calls']:
                            raise ValueError('actual model calls disagree with frozen trace or budget')
                    except Exception as exc:
                        inference_error=f'{type(exc).__name__}: {exc}'
                        candidate=adapter.failure(exc,value['id'])
                    finally:hook.remove()
                    elapsed=time.perf_counter()-inference_start
                    calls=trace[-1]['proposals'] if trace else len(network_calls)
                    if observe is not None and candidate['kind']=='candidate':observe(value,state,candidate,trace)
                    with (output/'callbacks.jsonl').open('a') as stream:
                        stream.write(json.dumps({'step':value['id'],'model_correction_seconds':elapsed,
                            'proposals':calls,'network_passes':len(network_calls),'trace':trace,'outcome':candidate['kind'],
                            'error':inference_error if candidate['kind']=='error' else None},allow_nan=False)+'\n')
                    child.stdin.write(json.dumps(candidate,allow_nan=False).encode()+b'\n');child.stdin.flush()
                elif value['kind']=='frame':
                    frames.append(value)
                    with (output/'frames.jsonl').open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')
                elif value['kind']=='diagnostic_attempt':
                    attempts.append(value)
                    with (output/'rejected-attempts.jsonl').open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')
                elif value['kind']=='complete':complete=value
                else:raise ValueError('unknown native message')
            child.stdin.close();child.wait(timeout=5)
            if child.returncode!=0 or complete is None:raise RuntimeError('native trajectory did not complete')
            if inference_error is not None:raise RuntimeError('current adapter failed: '+inference_error)
        except BaseException as exc:
            error=f'{type(exc).__name__}: {exc}'
            raise
        finally:
            if child.poll() is None:
                terminated=True;os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
            if not child.stdin.closed:
                try:child.stdin.close()
                except BrokenPipeError:pass
            child.stdout.close()
            policy=state.policy if state is not None else None
            report={'schema':'rheon-residual-trajectory-v1','status':'failed' if error else 'completed','error':error,
                'scene':scene,'result':complete,'frames':len(frames),'command':command,
                'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'elapsed_s':time.perf_counter()-started,
                'native_process':{'pid':child.pid,'returncode':child.returncode,'reaped':child.poll() is not None,'terminated':terminated},
                'adapter_manifest':str(adapter_manifest) if adapter_manifest else None,
                'adapter_manifest_sha256':hashlib.sha256(adapter_manifest.read_bytes()).hexdigest() if adapter_manifest else None,
                'adapter_loading_seconds':loading_seconds,
                'kind':'native-algorithm-control' if control else 'raw-learned-independent-trajectory-after-native-warmup',
                'deployment_changed':False,'cpu_threads':threads,'cpu_affinity':sorted(os.sched_getaffinity(0)),
                'cuda_initialized':False if control else state.torch.cuda.is_initialized(),
                'learned_proposals_per_pressure_solve':None if control else policy['max_network_calls'],
                'native_pressure_refinement':False,'native_fallback':False,
                'execution_policy':'independent_stop_on_reject',
                'rejected_attempts':len(attempts),
                'same_request_pressure_comparisons':[f['diagnostic_attempt']['same_request_pressure_comparison'] for f in frames if f.get('diagnostic_attempt')],
                'adaptive_stopping':None if control else policy.get('stopping'),
                'proposal_network_evaluations':None if control else 1,
                'minimum_network_evaluations':0 if control else policy.get('minimum_network_evaluations',0),
                'qualified_native_frames':sum(bool((f.get('reference_quality') or {}).get('qualified')) for f in frames),
                'worst_ai_relative_residual':max([f['ai_quality']['relative_residual'] or 0 for f in frames if f.get('ai_quality')] or [0]),
                'worst_ai_actual_divergence':max([f['ai_quality']['actual_divergence'] for f in frames if f.get('ai_quality')] or [0])}
            (output/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','scene','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--adapter-manifest',type=Path)
    parser.add_argument('--reference-control',action='store_true')
    parser.add_argument('--threads',type=int,choices=(1,4,8),default=1)
    args=parser.parse_args()
    if not args.reference_control and args.adapter_manifest is None:parser.error('current adapter manifest required')
    result=run(args.binary.absolute(),json.loads(args.scene.read_text()),args.output.absolute(),
        adapter_manifest=args.adapter_manifest,control=args.reference_control,threads=args.threads)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
