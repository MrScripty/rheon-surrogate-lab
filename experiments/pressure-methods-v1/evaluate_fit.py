"""Compare restored fits on consumed small grids and fresh domain development scenes."""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'experiments/pressure-four-pass-distillation-v1'),
               str(ROOT/'experiments/pressure-pass-sweep-v1')]
from common import read, write, sha
import inference
import numerics


def evaluate(args):
    if args.output.exists():
        raise ValueError('Fresh evaluation directory required')
    os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[-1]})
    original = read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    receipt = read(args.study/'receipt.json')
    if receipt['status'] != 'completed' or not all(receipt['artifact_parity'].values()):
        raise ValueError('Completed managed CUDA fit and exact fresh CPU restore required')
    artifacts = args.study/'restore'
    entry = dict(id=args.name,project=str(args.project.relative_to(ROOT)),restore=str(artifacts.relative_to(ROOT)),
        weights_sha256=receipt['weights_sha256'],identities={str(p.relative_to(ROOT)):sha(p) for p in
            [args.project/'source-witness.json',args.project/'config.json',artifacts/'weights.npz',
             artifacts/'normalization.json',artifacts/'receipt.json',artifacts/'history.json']})
    material = dict(original,models=[entry])
    state = inference.load(material,args.name)
    torch,np = state.torch,state.np
    inputs = ROOT/original['config']['validation_inputs']
    for name,expected in original['validation_sources'].items():
        if sha(inputs/name) != expected:
            raise ValueError('Consumed validation changed')
    _,parents = state.data.load(tuple((inputs/name).read_bytes() for name in
        ['validation_dataset.npz','validation_dataset_extra.npz']),read(inputs/'validation_splits.json'))
    cases = [(dict(population='consumed128',trajectory_id=p['entry']['trajectory_id'],frame=p['frames'][i]['index'],
                   counts_xyz=p['counts_xyz'],cell_size_m=p['cell_size_m']),state.data.sample(p,i))
             for p in parents for i in range(len(p['frames']))]
    if len(cases) != 128:
        raise ValueError('Exact consumed 128-request population required')
    preparation = HERE/'pack_training.py'
    loader = importlib.util.spec_from_file_location('pressure_methods_evaluation_capture',preparation)
    native = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(native)
    reservation_path = ROOT/'runs/pressure-methods-v1/native-training-02/reservation.json'
    reservation = read(reservation_path)
    source_evidence = {str(reservation_path.relative_to(ROOT)):sha(reservation_path)}
    for row in reservation['scenes']:
        if row['campaign_group'] != 'domain-development':
            continue
        directory,capture = native.captured(row)
        if row['split'] != 'validation':
            raise ValueError('New-domain evaluation must use development validation scenes')
        source_evidence[str((directory/'receipt.json').relative_to(ROOT))] = sha(directory/'receipt.json')
        for frame in capture['frames']:
            sample = {name:torch.from_numpy(native.field(directory,frame,'pressure' if name=='reference' else name).copy())[None,None]
                      for name in ['base','rhs','reference','native_input_residual']}
            sample.update(cell_size_m=torch.tensor([1/row['cells_per_metre']],dtype=torch.float64),
                          rho=torch.tensor([1000.],dtype=torch.float64),dt=torch.tensor([1/60],dtype=torch.float64))
            population = ('domain32' if row['counts_xyz']==[32]*3 else
                          'domain64' if row['counts_xyz']==[64]*3 else 'domain-odd')
            cases.append((dict(population=population,trajectory_id=row['trajectory_id'],frame=frame['index'],
                               counts_xyz=row['counts_xyz'],cell_size_m=1/row['cells_per_metre']),sample))
    if len(cases) != 140:
        raise ValueError('All six development trajectories and both snapshots required')
    args.output.mkdir(parents=True)
    calls = []
    hook = state.network.register_forward_hook(lambda *_:calls.append(1))
    rows = []
    rng = np.random.Generator(np.random.PCG64(2026101812))
    policies = [1,2,4,8,0]
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for index,(identity,sample) in enumerate(cases):
                rhs_norm = max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
                action = lambda value:state.model.full_operator(value,sample['cell_size_m'],sample['rho'])
                closure = float(torch.linalg.vector_norm(sample['rhs']-action(sample['base'])-sample['native_input_residual']))/rhs_norm
                if closure > 1e-10:
                    raise ValueError('Incoming field closure changed')
                for passes in rng.permutation(policies):
                    passes = int(passes)
                    def run():
                        pressure = (inference.fixed(state,sample,passes) if passes else
                                    sample['base']+numerics.correction(sample['native_input_residual'],sample['cell_size_m'],sample['rho'],None))
                        residual = float(torch.linalg.vector_norm(sample['rhs']-action(pressure)))/rhs_norm
                        return pressure,residual
                    calls.clear()
                    tick = time.perf_counter()
                    pressure,error = run()
                    first_seconds = time.perf_counter()-tick
                    count = len(calls)
                    times = []
                    for _ in range(args.repetitions):
                        calls.clear()
                        tick = time.perf_counter()
                        repeated,repeated_error = run()
                        times.append(time.perf_counter()-tick)
                        if len(calls)!=count or error!=repeated_error or not torch.equal(pressure,repeated):
                            raise ValueError('Repeated inference changed output or work')
                    if count > passes or not bool(torch.isfinite(pressure).all()) or float(pressure.flatten()[0]) != 0.:
                        raise ValueError('Bounded finite gauge-fixed inference required')
                    delta = pressure-sample['reference']
                    centered = delta-delta.mean()
                    reference = sample['reference']-sample['reference'].mean()
                    gradient_error = sum(float(torch.diff(delta,dim=a).square().sum()) for a in [2,3,4])
                    gradient_norm = sum(float(torch.diff(reference,dim=a).square().sum()) for a in [2,3,4])
                    rows.append(dict(**identity,policy='neural-'+str(passes) if passes else 'full-numerical',
                        passes=passes,network_calls=count,relative_equation_residual=error,
                        mean_free_pressure_relative_l2=float(torch.linalg.vector_norm(centered))/max(float(torch.linalg.vector_norm(reference)),1e-300),
                        gradient_relative_l2=math.sqrt(gradient_error/max(gradient_norm,1e-300)),
                        timing_samples_s=times,median_policy_s=statistics.median(times),
                        first_invocation_s=first_seconds,incoming_closure_relative=closure))
                if index%16==0:
                    write(args.output/'progress.json',dict(completed_cases=index+1,total=len(cases),elapsed_s=time.perf_counter()-started))
                    print(args.name,index+1,'/',len(cases),flush=True)
    finally:
        hook.remove()
    summary = []
    for population in sorted({r['population'] for r in rows}):
        for passes in policies:
            selected = [r for r in rows if r['population']==population and r['passes']==passes]
            summary.append(dict(population=population,policy=selected[0]['policy'],cases=len(selected),
                qualified=sum(r['relative_equation_residual']<=1e-6 for r in selected),
                worst_relative_equation_residual=max(r['relative_equation_residual'] for r in selected),
                worst_pressure_relative_l2=max(r['mean_free_pressure_relative_l2'] for r in selected),
                worst_gradient_relative_l2=max(r['gradient_relative_l2'] for r in selected),
                median_policy_ms=1000*statistics.median(r['median_policy_s'] for r in selected)))
    write(args.output/'results.json',dict(status='completed',name=args.name,entry=entry,summary=summary,rows=rows,
        source_evidence=source_evidence,training_receipt_sha256=sha(args.study/'receipt.json'),
        source_sha256=sha(__file__),native_reader_source_sha256=sha(preparation),numerics_sha256=sha(HERE/'numerics.py'),
        timing_scope='Single-thread CPU from prepared physical request through final relative residual, including every numerical and neural stage. Excludes loading, IPC, simulation and oracle error metrics.',
        cache_scope='First invocation recorded separately; repeated samples use warm geometry caches. Single-track production timing still required.',
        declared_equation_tolerance=1e-6,untouched_scenes_consumed=0,optimizer_updates=0,
        cuda_initialized=torch.cuda.is_initialized(),threads=torch.get_num_threads(),elapsed_s=time.perf_counter()-started,
        deployment_qualified=False))
    print('Completed comparison',args.name,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project','study','output']:
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--name',required=True)
    parser.add_argument('--repetitions',type=int,default=3,choices=[1,3])
    args = parser.parse_args()
    for name in ['project','study','output']:
        setattr(args,name,getattr(args,name).resolve())
    evaluate(args)
