"""Original C53 tensors on all development shapes via the qualified new wrapper."""
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments/pressure-four-pass-distillation-v1'),
             str(ROOT/'experiments/pressure-pass-sweep-v1'),str(ROOT/'experiments/pressure-methods-v1')]
from common import read,write,sha

def evaluate():
    output=ROOT/'runs/pressure-methods-v2/evaluation-01/C53-extended'
    if output.exists():raise ValueError('Fresh baseline evaluation required')
    os.sched_setaffinity(0,{23})
    import inference,cases,scalable,numerics
    material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    state=inference.load(material,'C53');torch=state.torch
    wrapper,qualification=scalable.qualify(state)
    inventory=state.data.helpers().state_inventory(state.network)
    rows=[];parity=[];started=time.perf_counter()
    with torch.no_grad():
        for index,(identity,sample) in enumerate(cases.load(state)):
            action=lambda pressure:numerics.fine_action(pressure,sample['cell_size_m'],sample['rho'])
            denominator=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
            closure=float(torch.linalg.vector_norm(sample['rhs']-action(sample['base'])-sample['native_input_residual']))/denominator
            if closure>1e-10:raise ValueError('Independent fine operator disagrees with native request')
            pressure=sample['base'];tick=time.perf_counter()
            for stage in range(1,9):
                pressure=scalable.fine_step(wrapper,sample,state.normalization,pressure)
                if stage not in [2,4,6,8]:continue
                relative=float(torch.linalg.vector_norm(sample['rhs']-action(pressure)))/denominator
                elapsed=time.perf_counter()-tick
                error=pressure-sample['reference'];error=error-error.mean()
                reference=sample['reference']-sample['reference'].mean()
                rows.append(dict(**identity,policy=dict(kind='extended-original-C53',passes=stage),relative_residual=relative,
                    qualified=relative<=1e-6,pressure_relative_l2=float(torch.linalg.vector_norm(error))/max(float(torch.linalg.vector_norm(reference)),1e-300),
                    network_calls=stage,policy_seconds=elapsed))
            if index in [0,31,63,95,127]:
                original=inference.fixed(state,sample,8)
                relative_difference=float(torch.linalg.vector_norm(original-pressure))/max(float(torch.linalg.vector_norm(original)),1e-300)
                if relative_difference>1e-9:raise ValueError('Extended physical recurrence changed original numerical output')
                parity.append(dict(case=index,pressure_relative_difference=relative_difference))
    if state.data.helpers().state_inventory(state.network)!=inventory:raise ValueError('Baseline tensors changed')
    summary=[]
    for population in sorted({r['population'] for r in rows}):
        for passes in [2,4,6,8]:
            group=[r for r in rows if r['population']==population and r['network_calls']==passes]
            summary.append(dict(population=population,passes=passes,cases=len(group),qualified=sum(r['qualified'] for r in group),
                worst_relative_residual=max(r['relative_residual'] for r in group),
                worst_pressure_relative_l2=max(r['pressure_relative_l2'] for r in group),
                median_policy_ms=1000*statistics.median(r['policy_seconds'] for r in group)))
    output.mkdir(parents=True)
    write(output/'results.json',dict(status='completed',entry=state.entry,rows=rows,summary=summary,
        source_sha256=sha(__file__),wrapper_source_sha256=sha(Path(__file__).with_name('scalable.py')),
        raw_forward_parity=qualification,original_recurrence_parity=parity,unchanged_tensor_inventory=True,
        timing_scope='Cumulative incremental recurrence through each sampled pass count; includes residual checks at earlier recorded stages. Final isolated timing required.',
        contract_scope='New offline shape ceiling; original adapter ceilings and captured source globals unchanged',
        optimizer_updates=0,blind_scenes_consumed=0,elapsed_s=time.perf_counter()-started))
    print('Evaluated original C53 on all 156 development requests',flush=True)

if __name__=='__main__':evaluate()
