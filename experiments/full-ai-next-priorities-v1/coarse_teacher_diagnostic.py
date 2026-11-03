"""Offline analytical low-mode control: diagnose a possible learned-head bottleneck.

This control invokes inverse eigenvalues. It is not a full-AI candidate and must
never appear on the candidate frontier or be used for an inference claim.
"""
import argparse
import math
import os
from pathlib import Path
from study_support import ROOT,read,write,sha,load_checkpoint,historical_cases,equation_error

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh offline analytical teacher diagnosis required')
    os.sched_setaffinity(0,{22});state=load_checkpoint('C107');torch=state.torch;torch.set_num_threads(1)
    initial=state.data.helpers().state_inventory(state.network);rows=[]
    original=state.network.shape_head.forward
    with torch.inference_mode():
        for identity,sample in historical_cases(state):
            ordinary=state.model.predict_pressure(state.network,sample,state.normalization)
            counts=tuple(sample['base'].shape[2:]);mode=torch.arange(4,dtype=torch.float64)
            eigenvalues=sum((2*(1-torch.cos(math.pi*mode/n))).reshape([4 if a==d else 1 for a in range(3)])for d,n in enumerate(counts))
            gain=eigenvalues.flatten()[1:].reciprocal();cap=max(counts)**2/2
            desired=torch.log(gain/(cap-gain)).float()
            def oracle(features):return desired-state.network.coarse_logits.detach()
            state.network.shape_head.forward=oracle
            try:controlled=state.model.predict_pressure(state.network,sample,state.normalization)
            finally:state.network.shape_head.forward=original
            rows.append(dict(identity,original_error=equation_error(state,ordinary,sample),
                analytical_low_mode_control_error=equation_error(state,controlled,sample)))
    if state.data.helpers().state_inventory(state.network)!=initial:raise ValueError('Offline control changed stored parameters')
    write(args.output,dict(status='completed',classification='offline_analytical_control_not_candidate',rows=rows,
        source_sha256=sha(__file__),C107_entry=state.entry,optimizer_updates=0,
        control_uses_analytical_inverse_eigenvalues=True,candidate_inference_or_target_claim=False,
        original_worst=max(r['original_error']for r in rows),control_worst=max(r['analytical_low_mode_control_error']for r in rows),
        original_cases_at_target=sum(r['original_error']<=7e-4 for r in rows),
        control_cases_at_target=sum(r['analytical_low_mode_control_error']<=7e-4 for r in rows),
        interpretation='Tests whether an ideal low-mode teacher plus unchanged fine CNN improves the bound; a learned replacement must be trained and independently timed/validated'))
    print('Offline analytical low-mode control worst',max(r['analytical_low_mode_control_error']for r in rows),'; not a candidate',flush=True)
