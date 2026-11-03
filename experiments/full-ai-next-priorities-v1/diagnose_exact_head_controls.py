"""Offline inverse-gain controls; never candidate inference or temporal frontier evidence."""
import argparse
import math
import os
from pathlib import Path
import numpy as np
from study_support import ROOT,HERE,BASE,read,write,sha,load_checkpoint,equation_error
from fresh_pressure_data import cases


def run(args):
    if args.output.exists():
        raise ValueError('Fresh offline analytical control required')
    os.sched_setaffinity(0,{23})
    state=load_checkpoint('a-head-calibration-s2-routed')
    torch=state.torch
    inventory=state.data.helpers().state_inventory(state.network)
    original_forward=state.network.small_shape_head.forward
    rows=[]
    def evaluate(identity,sample):
        if max(sample['base'].shape[2:])>16:
            return
        ordinary=state.model.predict_pressure(state.network,sample,state.normalization)
        counts=tuple(sample['base'].shape[2:])
        mode=torch.arange(4,dtype=torch.float64)
        eigenvalues=sum((2*(1-torch.cos(math.pi*mode/n))).reshape([4 if axis==d else 1 for axis in range(3)])
                        for d,n in enumerate(counts))
        gain=eigenvalues.flatten()[1:].reciprocal()
        cap=max(counts)**2/2
        desired=torch.log(gain/(cap-gain)).float()
        state.network.small_shape_head.forward=lambda features:desired-state.network.small_coarse_logits.detach()
        try:
            oracle=state.model.predict_pressure(state.network,sample,state.normalization)
        finally:
            state.network.small_shape_head.forward=original_forward
        reference=sample['reference']-sample['reference'].mean()
        denominator=max(float(reference.norm()),1e-30)
        rows.append(dict(identity,ordinary_equation_error=equation_error(state,ordinary,sample),
            controlled_equation_error=equation_error(state,oracle,sample),
            ordinary_pressure_error=float((ordinary-ordinary.mean()-reference).norm())/denominator,
            controlled_pressure_error=float((oracle-oracle.mean()-reference).norm())/denominator))
    with torch.inference_mode():
        for identity,sample in cases(BASE/'pressure-confirmation-data-01/bank.json',torch):
            evaluate(dict(identity,source_population='consumed_independent_confirmation'),sample)
        bank=read(BASE/'temporal-data-01/bank.json')
        for row in bank['trajectories']:
            if max(row['counts_xyz'])>16:
                continue
            if row['split']not in('train','development'):
                raise ValueError('Reserved temporal scenes excluded')
            path=ROOT/row['receipt']
            if sha(path)!=row['receipt_sha256']:
                raise ValueError('Consumed receipt changed')
            receipt=read(path)
            scene=receipt['scene']
            shape=(1,1,*scene['counts'][::-1])
            for start in(0,8,63,120):
                for horizon in(1,2,3,4):
                    step=start+horizon
                    record=receipt['states'][step]
                    source=path.parent/record['path']
                    if sha(source)!=record['sha256']:
                        raise ValueError('Consumed state changed')
                    with np.load(source,allow_pickle=False)as arrays:
                        rhs=torch.from_numpy(arrays['rhs'].copy().reshape(shape))
                        reference=torch.from_numpy(arrays['pressure'].copy().reshape(shape))
                    sample=dict(base=torch.zeros_like(rhs),native_input_residual=rhs,rhs=rhs,reference=reference,
                        cell_size_m=torch.tensor([1/scene['cells_per_metre']],dtype=torch.float64),
                        rho=torch.tensor([1000.],dtype=torch.float64),dt=torch.tensor([1/60],dtype=torch.float64))
                    evaluate(dict(source_population='temporal_future_RHS_oracle',identifier=row['identifier'],split=row['split'],
                        counts_xyz=scene['counts'],start=start,horizon=horizon,step=step),sample)
    if inventory!=state.data.helpers().state_inventory(state.network):
        raise ValueError('Offline controls changed stored weights')
    summaries=[]
    for population in('consumed_independent_confirmation','temporal_future_RHS_oracle'):
        selected=[row for row in rows if row['source_population']==population]
        summaries.append(dict(population=population,requests=len(selected),
            ordinary_worst_equation_error=max(row['ordinary_equation_error']for row in selected),
            controlled_worst_equation_error=max(row['controlled_equation_error']for row in selected),
            ordinary_worst_pressure_error=max(row['ordinary_pressure_error']for row in selected),
            controlled_worst_pressure_error=max(row['controlled_pressure_error']for row in selected),
            controlled_equation_7e_4_passes=sum(row['controlled_equation_error']<=.0007 for row in selected),
            controlled_pressure_and_equation_4e_3_passes=sum(max(row['controlled_equation_error'],row['controlled_pressure_error'])<=.004 for row in selected)))
    write(args.output,dict(status='completed',classification='Offline analytical inverse control; never a candidate',rows=rows,
        summaries=summaries,source_sha256=sha(__file__),model_entry=state.entry,optimizer_updates=0,
        inverse_eigenvalues_used_offline=True,temporal_future_RHS_used_only_as_oracle=True,
        candidate_inference_or_accuracy_claim=False,weights_bitwise_unchanged=True,reserved_scenes_consumed=False))
    print('Measured exact-head offline controls on1056small-grid requests; not candidate performance',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
