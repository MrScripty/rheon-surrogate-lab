"""Bounded accuracy-only native/GPU qualification, one owned case per CPU slot."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)

def run(args):
    if args.output.exists():raise ValueError('Fresh bounded qualification batch required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('GPU model inference only')
    for name,h in reg['source_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Frozen batch source changed: '+name)
    population={c['identifier']:c for c in reg['cases']}
    if len(population)!=19 or any(c['split']!='development' for c in population.values()):raise ValueError('Exactly19remaining whole development scenes required')
    slots=reg['batch_cpu_slots']
    if len(slots)!=2 or len(set(slots))!=2:raise ValueError('Two distinct registered CPU slots required')
    pending={slot:[name for name in population if reg['host_cpu_by_identifier'][name]==slot] for slot in slots}
    args.output.mkdir(parents=True);started=time.monotonic();active={};completed=[];stopping=False
    result=dict(status='running',active=[],completed=completed,remaining=list(population),inference_device='cuda',display_label='GPU (CUDA)',
        numerical_references='Owned CPU native teachers',cpu_model_forward=False,optimizer_updates=0,latency_claim=False,
        expanded_reserved52_references_opened=0,new_independent_confirmation=False,protocol_sha256=digest(args.protocol))
    def save():
        result.update(active=[dict(identifier=row['identifier'],pid=row['process'].pid,host_cpu=slot) for slot,row in active.items()],
            remaining=[name for slot in slots for name in pending[slot]],elapsed_diagnostic_seconds=time.monotonic()-started)
        write(args.output/'controller.json',result)
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);save()
    try:
        while active or any(pending.values()):
            if stopping:raise InterruptedError('Stop requested; drain owned evaluation children')
            if time.monotonic()-started>reg['batch_maximum_wall_seconds']:raise TimeoutError('Bounded batch deadline reached')
            for slot in slots:
                if slot in active or not pending[slot]:continue
                name=pending[slot].pop(0);out=args.output/name;log=(args.output/(name+'.log')).open('w')
                command=[sys.executable,'-B',str(ROOT/reg['batch_evaluator']),'--protocol',str(args.protocol),'--identifier',name,'--output',str(out)]
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[slot]=dict(identifier=name,process=child,log=log,path=out/'results.json');save()
            for slot,row in list(active.items()):
                code=row['process'].poll()
                if code is None:continue
                row['log'].close()
                if code!=0 or not row['path'].is_file():raise RuntimeError('Owned evaluation failed operationally: '+row['identifier']+' code '+str(code))
                q=read(row['path']);custody=q['native_process']
                if q['status']!='completed_streamed_expanded_gpu_240step_scene' or len(q['rows'])!=960 or not custody['reaped'] or custody['returncode']!=0 or not custody['graceful_protocol_complete']:
                    raise ValueError('Incomplete case or native custody: '+row['identifier'])
                completed.append(dict(identifier=row['identifier'],path=str(row['path'].relative_to(ROOT)),sha256=digest(row['path']),
                    all_accuracy_pass=q['all_accuracy_pass'],accuracy_failing_outputs=q['accuracy_failing_outputs'],worst_error=q['worst_error'],
                    host_cpu=slot,native_process=custody))
                del active[slot];save()
                print('Full240development',row['identifier'],'accuracy',q['all_accuracy_pass'],'completed',len(completed),'/19',flush=True)
            time.sleep(.2)
        if len(completed)!=19:raise ValueError('Full registered19case population required')
        for name,h in reg['source_sha256'].items():
            if digest(ROOT/name)!=h:raise ValueError('Frozen batch source changed during execution: '+name)
        result.update(status='completed_remaining19development_240histories',all_accuracy_pass=all(c['all_accuracy_pass'] for c in completed),
            complete_outputs=19*960,all_owned_native_children_gracefully_reaped=True)
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        cleanup=[]
        for slot,row in list(active.items()):
            child=row['process']
            if child.poll() is None:child.send_signal(signal.SIGTERM)
        for slot,row in list(active.items()):
            child=row['process']
            try:code=child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=20)
            row['log'].close();cleanup.append(dict(identifier=row['identifier'],evaluator_returncode=code))
        active.clear();result['interrupted_child_cleanup']=cleanup;save()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
