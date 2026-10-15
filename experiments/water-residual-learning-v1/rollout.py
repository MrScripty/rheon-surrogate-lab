"""Own one bounded native paired trajectory and serve a recorded CPU checkpoint.

Reuses the existing Rheon comparison engine through a separate laboratory probe;
no application deployment contract is changed. Failures remain raw diagnostics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
import audit
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/water-stability-learning-v1'))
from learned_iteration import correct,stopping_policy,validate_budget

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


def run(binary,scene,output,project=None,artifacts=None,control=False,timeout=120,proposals=1,stopping=None):
    if type(proposals) is not int or not 1<=proposals<=32 or control and proposals!=1:
        raise ValueError('at most 32 learned proposals; native control requires one')
    if stopping is not None:
        stopping_policy(stopping)
        if control:raise ValueError('native control does not use learned stopping')
    if not control:
        audit.initialize(project,artifacts,sorted(os.sched_getaffinity(0)))
        validate_budget(proposals,audit.MODEL.CORRECTION['steps'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'scene.json').write_text(json.dumps(scene,indent=2)+'\n')
    environment={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','RAYON_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','TULDOK_TORCH_DEVICE':'cpu'}
    command=[str(binary),*(['--reference-control'] if control else [])]
    started=time.perf_counter();complete=None;frames=[];error=None;terminated=False
    with (output/'native-stderr.log').open('wb') as stderr:
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,env=environment,start_new_session=True)
        try:
            child.stdin.write(json.dumps(scene).encode()+b'\n');child.stdin.flush()
            for value in messages(child,time.monotonic()+timeout):
                if value['kind']=='predict':
                    torch=audit.TORCH;shape=(1,1,*value['counts_xyz'][::-1])
                    sample={key:torch.tensor(value[source],dtype=torch.float64).reshape(shape) for key,source in [('base','base_pressure'),('native_input_residual','native_input_residual')]}
                    sample.update(cell_size_m=torch.tensor([value['cell_size_m']],dtype=torch.float64),rho=torch.tensor([value['density_kg_m3']],dtype=torch.float64),dt=torch.tensor([value['dt_s']],dtype=torch.float64))
                    trace=[];inference_start=time.perf_counter()
                    with torch.no_grad():pressure=correct(audit.MODEL,audit.NETWORK,audit.NORM,sample,proposals,stopping,trace).flatten().tolist()
                    elapsed=time.perf_counter()-inference_start
                    with (output/'callbacks.jsonl').open('a') as stream:
                        stream.write(json.dumps({'step':value['id'],'model_correction_seconds':elapsed,
                            'proposals':trace[-1]['proposals'],'network_passes':trace[-1]['proposals']*audit.MODEL.CORRECTION['steps'],
                            'trace':trace},allow_nan=False)+'\n')
                    child.stdin.write(json.dumps({'id':value['id'],'pressure':pressure},allow_nan=False).encode()+b'\n');child.stdin.flush()
                elif value['kind']=='frame':
                    frames.append(value)
                    with (output/'frames.jsonl').open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')
                elif value['kind']=='complete':complete=value
                else:raise ValueError('unknown native message')
            child.stdin.close();child.wait(timeout=5)
            if child.returncode!=0 or complete is None:raise RuntimeError('native trajectory did not complete')
        except BaseException as exc:
            error=f'{type(exc).__name__}: {exc}'
            raise
        finally:
            if child.poll() is None:
                terminated=True;os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
            if not child.stdin.closed:child.stdin.close()
            child.stdout.close()
            report={'schema':'rheon-residual-trajectory-v1','status':'failed' if error else 'completed','error':error,
                'scene':scene,'result':complete,'frames':len(frames),'command':command,
                'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'elapsed_s':time.perf_counter()-started,
                'native_process':{'pid':child.pid,'returncode':child.returncode,'reaped':child.poll() is not None,'terminated':terminated},
                'source_project':str(project) if project else None,'cpu_restore':str(artifacts) if artifacts else None,
                'kind':'native-algorithm-control' if control else 'raw-learned-independent-trajectory-after-native-warmup',
                'deployment_changed':False,'cuda_initialized':False if control else audit.TORCH.cuda.is_initialized(),
                'learned_proposals_per_pressure_solve':None if control else proposals,
                'native_pressure_refinement':False,
                'adaptive_stopping':stopping,
                'qualified_native_frames':sum(bool((f.get('reference_quality') or {}).get('qualified')) for f in frames),
                'worst_ai_relative_residual':max([f['ai_quality']['relative_residual'] or 0 for f in frames if f.get('ai_quality')] or [0]),
                'worst_ai_actual_divergence':max([f['ai_quality']['actual_divergence'] for f in frames if f.get('ai_quality')] or [0])}
            (output/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','scene','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--project',type=Path);parser.add_argument('--artifacts',type=Path);parser.add_argument('--reference-control',action='store_true')
    parser.add_argument('--proposals',type=int,choices=range(1,33),default=1)
    parser.add_argument('--adaptive-profile',type=Path)
    args=parser.parse_args()
    if not args.reference_control and not (args.project and args.artifacts):parser.error('CPU restore project and artifacts required')
    stopping=stopping_policy(json.loads(args.adaptive_profile.read_text())) if args.adaptive_profile else None
    result=run(args.binary.absolute(),json.loads(args.scene.read_text()),args.output.absolute(),args.project.absolute() if args.project else None,args.artifacts.absolute() if args.artifacts else None,args.reference_control,proposals=args.proposals,stopping=stopping)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
