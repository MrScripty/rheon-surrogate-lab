"""Prepare independent late labels and larger native water parents, never train."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
from importlib.util import module_from_spec,spec_from_file_location
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

from diagnose import LAB,digest,read,write
sys.path.insert(0,str(LAB/'experiments/water-residual-learning-v1'))
import audit
import rollout
sys.path.insert(0,str(LAB/'experiments/water-shape-repeatability-v1'))
from execution import utc


def initialize(project,artifacts,cpus):
    audit.initialize(project,artifacts,cpus)
    sys.path.insert(0,str(LAB/'experiments/global-pressure-v1'))
    global ORACLE,PROCESS
    for name,path in [('ORACLE','experiments/global-pressure-v1/verify_capture.py'),('PROCESS','experiments/global-pressure-v1/capture.py')]:
        spec=spec_from_file_location('preparation_'+name,LAB/path);module=module_from_spec(spec);spec.loader.exec_module(module);globals()[name]=module


def independent(delta,sample):
    import numpy as np
    counts=list(delta.shape[2:][::-1]);h=float(sample['cell_size_m'][0]);rho=float(sample['rho'][0])
    values=ORACLE.full_action(delta.flatten().tolist(),counts,[h]*3,rho)
    return audit.TORCH.from_numpy(np.asarray(values,dtype='<f8')).reshape_as(delta)


def late(task):
    import numpy as np
    case,pressure,path=task;torch=audit.TORCH
    _,parents=audit.DATA.read_view(LAB/'runs/water-residual-learning-v1/views/cohort-16')
    parent=parents[case['parent']];assert parent['entry']['split']=='train'
    sample=audit.DATA.sample(parent,case['frame']);pressure=torch.from_numpy(pressure).reshape_as(sample['base'])
    residual=sample['native_input_residual']-audit.MODEL.full_operator(pressure-sample['base'],sample['cell_size_m'],sample['rho'])
    target=audit.spectral_correction(residual,sample)
    compatible=residual-residual.mean();scale=max(audit.vector_norm(compatible),1e-30)
    defect=independent(target,sample)-compatible;relative=audit.vector_norm(defect)/scale
    assert relative<1e-9 and float(target[0,0,0,0,0])==0
    original=sample['reference']-pressure
    teacher_defect=audit.vector_norm(independent(original,sample)-residual)/scale
    rounded=audit.MODEL.frozen.gauge_correction((target/audit.NORM['scales']['pressure']['value']).float(),audit.NORM['scales']['pressure']['value'])
    rhs=sample['rhs'];rounded_error=independent(rounded,sample)-residual
    arrays={'base':pressure.numpy(),'native_input_residual':compatible.numpy(),'reference_correction':target.numpy(),
            **{k:sample[k].numpy() for k in ('rho','dt','cell_size_m')}}
    with Path(path).open('xb') as stream:np.savez(stream,**arrays)
    return {'trajectory_id':case['trajectory_id'],'frame':case['frame'],'counts_xyz':case['counts_xyz'],
        'operator_relative_error':relative,'captured_teacher_defect_over_late':teacher_defect,
        'removed_constant_over_late':audit.vector_norm(residual-compatible)/scale,
        'float32_target_relative_to_late':audit.vector_norm(rounded_error)/scale,
        'float32_target_relative_to_original_rhs':audit.vector_norm(rounded_error)/max(audit.vector_norm(rhs),1e-30),
        'path':str(Path(path).relative_to(LAB)),'sha256':digest(path),'affinity':sorted(os.sched_getaffinity(0))}


def native_parent(task):
    import numpy as np
    torch=audit.TORCH;scene,output,binary=task;output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'scene.json',scene);arrays={};samples=[];frames=[];complete=None;error=None
    start=time.perf_counter();child=None
    try:
        with (output/'native-stderr.log').open('xb') as stderr:
            with PROCESS.bounded_process([str(binary)],120,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,
                    env={**os.environ,'RAYON_NUM_THREADS':'1','OMP_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}) as child:
                child.stdin.write(json.dumps(scene).encode()+b'\n');child.stdin.flush()
                for value in rollout.messages(child,time.monotonic()+120):
                    if value['kind']=='predict':
                        shape=(1,1,*value['counts_xyz'][::-1])
                        sample={k:torch.tensor(value[v],dtype=torch.float64).reshape(shape) for k,v in [('base','base_pressure'),('native_input_residual','native_input_residual')]}
                        sample.update(cell_size_m=torch.tensor([value['cell_size_m']],dtype=torch.float64),rho=torch.tensor([value['density_kg_m3']],dtype=torch.float64),dt=torch.tensor([value['dt_s']],dtype=torch.float64))
                        delta=audit.spectral_correction(sample['native_input_residual'],sample);pressure=sample['base']+delta
                        assert float(pressure[0,0,0,0,0])==0
                        label_error=independent(delta,sample)-sample['native_input_residual']
                        relative=audit.vector_norm(label_error)/max(audit.vector_norm(sample['native_input_residual']),1e-30)
                        assert relative<1e-8,relative
                        index=value['id']
                        for key,field in [('pressure',pressure),('base',sample['base']),('residual',sample['native_input_residual'])]:arrays[f'{index:03d}-{key}']=field[0,0].numpy().copy()
                        samples.append({'step':index,'operator_relative_error':relative})
                        child.stdin.write(json.dumps({'id':index,'pressure':pressure.flatten().tolist()},allow_nan=False).encode()+b'\n');child.stdin.flush()
                    elif value['kind']=='frame':
                        frames.append(value)
                        with (output/'frames.jsonl').open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')
                    elif value['kind']=='complete':complete=value
                    else:raise ValueError('unknown native message')
                child.stdin.close();assert child.wait(timeout=5)==0 and complete is not None
        assert len(frames)==scene['steps'] and len(samples)==scene['steps']-scene['warmup_steps']
        assert all(f['reference_quality']['qualified'] and (not f.get('ai_quality') or f['ai_quality']['qualified']) for f in frames)
        with (output/'fields.npz').open('xb') as stream:np.savez(stream,**arrays)
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';raise
    finally:
        result={'status':'failed' if error else 'qualified_cpu_teacher_parent','error':error,'scene':scene,
            'elapsed_seconds':time.perf_counter()-start,'frames':len(frames),'samples':samples,
            'native_process':{'pid':child.pid if child else None,'returncode':child.returncode if child else None,'reaped':child is not None and child.poll() is not None},
            'binary_sha256':digest(binary),'network_calls':0,'optimizer_updates':0,'cuda_initialized':torch.cuda.is_initialized(),
            'affinity':sorted(os.sched_getaffinity(0)),
            'label_source':'independent float64 sealed-Neumann CPU spectral teacher; native paired physics/reference gates; not an AI solution or the native iterative pressure field'}
        if (output/'fields.npz').exists():result['fields_sha256']=digest(output/'fields.npz')
        write(output/'receipt.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',type=Path,required=True);p.add_argument('--reports',type=Path,required=True);p.add_argument('--output',type=Path)
    args=p.parse_args();sequence=read(args.sequence);root=LAB/sequence['root'];out=args.output or root/'cpu-preparation';out.mkdir(exist_ok=False)
    started=time.perf_counter();cpus=sorted(os.sched_getaffinity(0))[1:]
    ref=sequence['reference'];init=(LAB/ref['project'],LAB/ref['artifacts'],cpus)
    with (root/'cpu-qualification.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        previous=LAB/'runs/water-coverage-feedback-v1/prepared-late-residuals';receipt=read(previous/'receipt.json')
        assert receipt['weights_sha256']==digest(LAB/ref['artifacts']/'weights.npz')
        import numpy as np
        with np.load(previous/'predictions.npz',allow_pickle=False) as arrays:
            tasks=[(case,arrays[f'case_{i:03d}'].copy(),str(out/f'late-{i:03d}.npz')) for i,case in enumerate(receipt['cases'])]
        with ProcessPoolExecutor(max_workers=min(22,len(cpus)),mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=init) as pool:
            late_records=list(pool.map(late,tasks))
        scenes=read(LAB/'experiments/water-targeted-feedback-v1/configs/preparation-scenes.json')['scenes']
        fixture={**scenes[0],'steps':16}
        binary=LAB/'runs/water-residual-learning-v1/native-rollout-build/rheon-residual-rollout'
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=init) as pool:
            fixture_result=pool.submit(native_parent,(fixture,str(out/'fixture'),str(binary))).result()
        binary=LAB/'runs/water-residual-learning-v1/native-rollout-build/rheon-residual-rollout'
        with ProcessPoolExecutor(max_workers=min(8,len(cpus)),mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=init) as pool:
            parents=list(pool.map(native_parent,[(s,str(out/f'parent-{i:02d}'),str(binary)) for i,s in enumerate(scenes)]))
    result={'recorded_at_utc':utc(),'output_root':str(out.relative_to(LAB)),'late_labels':late_records,'fixture':fixture_result,'parents':parents,
        'elapsed_seconds':time.perf_counter()-started,'source_sha256':digest(Path(__file__)),
        'old_predictions_sha256':digest(previous/'predictions.npz'),'optimizer_updates':0,
        'scope':'future training preparation only; no current-fit input, final data or learned inference credit; pressure/base/residual arrays, native physics telemetry, no velocity/tracer export'}
    write(args.reports/'cpu-preparation.json',result)
    print({'late_labels':len(late_records),'new_parents':len(parents),'elapsed_seconds':result['elapsed_seconds']})


if __name__=='__main__':main()
