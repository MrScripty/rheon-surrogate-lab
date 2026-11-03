"""Evaluate both predeclared continuation checkpoints from verified CPU restore artifacts."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import statistics
import time
from study_support import ROOT,BASE,read,write,sha,load_checkpoint,historical_cases,equation_error

def run(args):
    output=args.output.resolve()
    if output.exists():raise ValueError('Fresh checkpoint evaluation required')
    output.mkdir(parents=True);os.sched_setaffinity(0,{23})
    state=load_checkpoint(args.identifier);torch=state.torch
    restore=ROOT/state.entry['restore'];receipt=read(restore/'receipt.json');history=read(restore/'history.json')
    cases=historical_cases(state);results=[]
    project=ROOT/state.entry['project']
    for update in read(project/'config.json')['training']['checkpoints'][1:]:
        name=f'checkpoint-{update:03d}.npz';weights=restore/name
        if sha(weights)!=receipt['outputs'][name]['sha256']:raise ValueError('CPU-restored checkpoint artifact changed')
        inventory=history['checkpoints'][name]['inventory'];state.data.helpers().load_weights(state.network,weights,inventory)
        rows=[]
        with torch.no_grad():
            for identity,sample in cases:
                import physical_residual
                stages=[]
                pressure=physical_residual.predict_pressure(state.network,sample,state.normalization,state.model.owner,state.model.CORRECTION,stages)
                difference=pressure-sample['reference'];difference=difference-difference.mean()
                centered=sample['reference']-sample['reference'].mean()
                value=equation_error(state,pressure,sample)
                if not math.isfinite(value):raise ValueError('Checkpoint evaluation nonfinite')
                rows.append(dict(**identity,equation_relative_residual=value,
                    pressure_mean_free_relative_l2=float(torch.linalg.vector_norm(difference))/max(float(torch.linalg.vector_norm(centered)),1e-30),
                    first_stage_equation_relative_residual=equation_error(state,stages[0],sample),
                    output_sha256=hashlib.sha256(pressure.numpy().tobytes()).hexdigest()))
        errors=sorted(r['equation_relative_residual']for r in rows)
        result=dict(status='completed',identifier=args.identifier,update=update,weights_sha256=sha(weights),
            weights_source=str(weights.relative_to(ROOT)),inventory_verified=state.data.helpers().state_inventory(state.network)==inventory,
            rows=rows,cases=128,median_error=statistics.median(errors),p95_error=errors[math.ceil(.95*len(errors))-1],
            worst_error=max(errors),cases_at_target=sum(v<=7e-4 for v in errors),
            optimizer_updates_executed=0,source_sha256=sha(__file__),historical_population_consumed=True,
            scientific_scope='Prepared-request equation and pressure errors; no new native trajectory acceptance claim')
        write(output/f'checkpoint-{update:03d}.json',result);results.append({k:v for k,v in result.items()if k!='rows'})
        print(args.identifier,'update',update,'worst',max(errors),'target cases',result['cases_at_target'],flush=True)
    write(output/'summary.json',dict(status='completed',rows=results,source_entry=state.entry,
        original_cpu_restore_receipt_sha256=sha(restore/'receipt.json'),cuda_initialized=torch.cuda.is_initialized()))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identifier',required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
