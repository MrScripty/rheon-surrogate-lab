"""Prespecified pass-count and independent-spacing follow-up measurements."""
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
sys.path[:0] = [str(ROOT/'experiments/pressure-four-pass-distillation-v1'),str(ROOT/'experiments/pressure-pass-sweep-v1')]
from common import read, write, sha
import inference
import numerics


def evaluate(args):
    if args.output.exists() or any(n not in range(1,17) for n in args.passes):
        raise ValueError('Fresh bounded follow-up required')
    if args.evaluation is not None and args.orderings != ['none']:
        raise ValueError('A trained method owns its numerical stages; do not add a second intervention')
    os.sched_setaffinity(0,{args.cpu if args.cpu is not None else sorted(os.sched_getaffinity(0))[-1]})
    material = read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    if args.evaluation is not None:
        previous = read(args.evaluation)
        if previous['status'] != 'completed' or previous['optimizer_updates'] != 0:
            raise ValueError('Completed actual CPU fit evaluation required')
        entry = previous['entry']
        state = inference.load(dict(material,models=[entry]),entry['id'])
        model_name = previous['name']
    else:
        state = inference.load(material,args.checkpoint)
        model_name = args.checkpoint
        entry = state.entry
    torch,np = state.torch,state.np
    evidence = {}
    if args.population == 'consumed128':
        inputs = ROOT/material['config']['validation_inputs']
        for name,expected in material['validation_sources'].items():
            if sha(inputs/name) != expected:
                raise ValueError('Consumed population changed')
        _,parents = state.data.load(tuple((inputs/name).read_bytes() for name in
            ['validation_dataset.npz','validation_dataset_extra.npz']),read(inputs/'validation_splits.json'))
        cases = [(dict(population='consumed128',trajectory_id=p['entry']['trajectory_id'],frame=p['frames'][i]['index'],
                       counts_xyz=p['counts_xyz'],cell_size_m=p['cell_size_m']),state.data.sample(p,i))
                 for p in parents for i in range(len(p['frames']))]
        if len(cases) != 128:
            raise ValueError('Exact consumed population required')
    else:
        source = HERE/'pack_training.py'
        loader = importlib.util.spec_from_file_location('spacing_followup_native_reader',source)
        native = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(native)
        reservation_path = ROOT/'runs/pressure-methods-v1/native-spacing-validation-03/reservation.json'
        reservation = read(reservation_path)
        if len(reservation['scenes']) != 12:
            raise ValueError('All twelve reserved independent spacing scenes required')
        evidence[str(reservation_path.relative_to(ROOT))] = sha(reservation_path)
        cases = []
        for row in reservation['scenes']:
            if row['split'] != 'validation' or row['campaign_group'] != 'spacing-development':
                raise ValueError('Independent validation only')
            directory,receipt = native.captured(row)
            evidence[str((directory/'receipt.json').relative_to(ROOT))] = sha(directory/'receipt.json')
            for frame in receipt['frames']:
                sample = {name:torch.from_numpy(native.field(directory,frame,'pressure' if name=='reference' else name).copy())[None,None]
                          for name in ['base','rhs','reference','native_input_residual']}
                sample.update(cell_size_m=torch.tensor([1/row['cells_per_metre']],dtype=torch.float64),
                    rho=torch.tensor([1000.],dtype=torch.float64),dt=torch.tensor([1/60],dtype=torch.float64))
                cases.append((dict(population='cpm'+str(row['cells_per_metre']),trajectory_id=row['trajectory_id'],
                    paired_profile=row['paired_profile'],frame=frame['index'],counts_xyz=row['counts_xyz'],cell_size_m=1/row['cells_per_metre']),sample))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    policies = [(ordering,passes) for ordering in args.orderings for passes in args.passes]+[('full-numerical',0)]
    rng = np.random.Generator(np.random.PCG64(2026101814))
    rows = []
    calls = []
    hook = state.network.register_forward_hook(lambda *_:calls.append(1))
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for index,(identity,sample) in enumerate(cases):
                norm = max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
                action = lambda value:state.model.full_operator(value,sample['cell_size_m'],sample['rho'])
                if float(torch.linalg.vector_norm(sample['rhs']-action(sample['base'])-sample['native_input_residual']))/norm>1e-10:
                    raise ValueError('Input physical closure failed')
                for position in rng.permutation(len(policies)):
                    ordering,passes = policies[int(position)]
                    def run():
                        pressure = sample['base']
                        if ordering=='full-numerical':
                            pressure = pressure+numerics.correction(sample['native_input_residual'],sample['cell_size_m'],sample['rho'],None)
                        if ordering=='before':
                            pressure = pressure+numerics.correction(sample['native_input_residual'],sample['cell_size_m'],sample['rho'],3)
                        for _ in range(passes):
                            pressure = inference.step(state,sample,pressure)
                        if ordering=='after':
                            pressure = pressure+numerics.correction(sample['rhs']-action(pressure),sample['cell_size_m'],sample['rho'],3)
                        return pressure,float(torch.linalg.vector_norm(sample['rhs']-action(pressure)))/norm
                    calls.clear()
                    prediction,error = run()
                    count = len(calls)
                    times = []
                    for _ in range(args.repetitions):
                        calls.clear()
                        tick = time.perf_counter()
                        repeated,repeated_error = run()
                        times.append(time.perf_counter()-tick)
                        if len(calls)!=count or error!=repeated_error or not torch.equal(prediction,repeated):
                            raise ValueError('Repeated follow-up inference changed')
                    if count>passes or not math.isfinite(error) or not bool(torch.isfinite(prediction).all()) or float(prediction.flatten()[0])!=0:
                        raise ValueError('Bounded finite inference required')
                    delta = prediction-sample['reference']
                    centered = delta-delta.mean()
                    target = sample['reference']-sample['reference'].mean()
                    gradient_error = sum(float(torch.diff(delta,dim=a).square().sum())for a in [2,3,4])
                    gradient_norm = sum(float(torch.diff(target,dim=a).square().sum())for a in [2,3,4])
                    rows.append(dict(**identity,ordering=ordering,passes=passes,network_calls=count,
                        relative_equation_residual=error,qualified=error<=1e-6,
                        pressure_relative_l2=float(torch.linalg.vector_norm(centered))/max(float(torch.linalg.vector_norm(target)),1e-300),
                        gradient_relative_l2=math.sqrt(gradient_error/max(gradient_norm,1e-300)),
                        timing_samples_s=times,median_policy_s=statistics.median(times)))
                if index%16==0:
                    print(model_name,args.population,index+1,'/',len(cases),flush=True)
    finally:
        hook.remove()
    summary = []
    for population in sorted({r['population']for r in rows}):
        for ordering,passes in policies:
            selected = [r for r in rows if r['population']==population and r['ordering']==ordering and r['passes']==passes]
            summary.append(dict(population=population,ordering=ordering,passes=passes,cases=len(selected),
                qualified=sum(r['qualified']for r in selected),worst_relative_equation_residual=max(r['relative_equation_residual']for r in selected),
                median_policy_ms=1000*statistics.median(r['median_policy_s']for r in selected)))
    write(args.output,dict(status='completed',model=model_name,entry=entry,population=args.population,summary=summary,rows=rows,
        source_sha256=sha(__file__),numerics_sha256=sha(HERE/'numerics.py'),source_evidence=evidence,
        evaluation_sha256=sha(args.evaluation)if args.evaluation is not None else None,
        declared_equation_tolerance=1e-6,elapsed_s=time.perf_counter()-started,optimizer_updates=0,untouched_scenes_consumed=0,
        cpu=next(iter(os.sched_getaffinity(0))),threads=torch.get_num_threads(),cuda_initialized=torch.cuda.is_initialized(),
        timing_scope='Prepared physical input through final residual; includes all model/numerical stages; loading, IPC, simulation and oracle metrics excluded.'))
    print('Completed follow-up',model_name,args.population,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument('--checkpoint',choices=['C53','C44','C37'])
    selected.add_argument('--evaluation',type=Path)
    parser.add_argument('--population',choices=['consumed128','spacing-development'],required=True)
    parser.add_argument('--passes',type=int,nargs='+',default=[1,2,4,8])
    parser.add_argument('--orderings',choices=['none','before','after'],nargs='+',default=['none'])
    parser.add_argument('--repetitions',type=int,choices=[1,3],default=1)
    parser.add_argument('--cpu',type=int)
    parser.add_argument('--output',required=True,type=Path)
    evaluate(parser.parse_args())
