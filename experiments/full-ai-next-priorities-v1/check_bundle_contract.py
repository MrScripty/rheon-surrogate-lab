"""Exercise real joint outputs, causal feature construction and differentiable loss inputs."""
import os
import statistics
import time
import numpy as np
import torch
from study_support import ROOT,BASE,DOCS,read,write,sha
from bundle_model import BundleNet
from temporal_features import packed_state,auxiliary,FIELD_NAMES

def run():
    output=BASE/'bundle-contract-01.json'
    if output.exists():raise ValueError('Fresh bundle contract check required')
    os.sched_setaffinity(0,{23});torch.set_num_threads(1)
    bank=read(BASE/'temporal-data-01/bank.json')
    selected=next(r for r in bank['trajectories']if r['split']=='train'and r['counts_xyz']==[16,16,16])
    receipt_path=ROOT/selected['receipt'];receipt=read(receipt_path)
    spec=receipt['states'][8];path=receipt_path.parent/spec['path']
    if sha(path)!=spec['sha256']:raise ValueError('Source training state changed')
    with np.load(path,allow_pickle=False)as source:fields={name:source[name].copy()for name in FIELD_NAMES}
    # This provisional scale only qualifies the interface. Actual scales are fitted
    # on the full training bank before any temporal fit.
    normalization=dict(scales=dict(pressure=100.,velocity=.01,tracer=.05))
    rows=[]
    for horizons in (1,2,3,4):
        torch.manual_seed(2026101921);network=BundleNet(horizons).eval()
        with torch.inference_mode():
            outputs=network.predict(fields,receipt['scene'],8,normalization)
            if len(outputs)!=horizons:raise ValueError('Every future state must be returned')
            for state in outputs:
                if any(state[name].numel()!=fields[name].size for name in FIELD_NAMES):raise ValueError('Native staggering lost')
                if any(not bool(torch.isfinite(value).all())for value in state.values()):raise ValueError('Nonfinite joint output')
                if state['pressure'][0]!=0:raise ValueError('Pressure interchange gauge failed')
            calls=[];hook=network.register_forward_hook(lambda *_:calls.append(1))
            network.predict(fields,receipt['scene'],8,normalization);hook.remove()
            if len(calls)!=1:raise ValueError('Temporal prediction must invoke the spatial model once')
            samples=[]
            for _ in range(20):
                start=time.perf_counter();network.predict(fields,receipt['scene'],8,normalization);samples.append(time.perf_counter()-start)
        state=packed_state(fields,tuple(receipt['scene']['counts']),normalization['scales'])
        features=auxiliary(receipt['scene'],8,horizons,normalization['scales'])
        network.zero_grad(set_to_none=True);predicted=network(state,features)
        loss=predicted.square().mean();loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all())for p in network.parameters()):raise ValueError('Incomplete finite temporal gradients')
        rows.append(dict(horizons=horizons,parameters=sum(p.numel()for p in network.parameters()),
            returned_states=len(outputs),spatial_model_calls=1,all_native_shapes_preserved=True,
            complete_finite_backward=True,optimizer_updates=0,
            provisional_median_block_ms=1000*statistics.median(samples),
            provisional_median_per_step_ms=1000*statistics.median(samples)/horizons))
    write(output,dict(status='passed',rows=rows,source_sha256=sha(__file__),model_sha256=sha(BASE.parents[1]/'experiments/full-ai-next-priorities-v1/bundle_model.py'),
        scope='Untrained architecture/interface feasibility only; not an accuracy or qualified timing result',
        source_training_receipt_sha256=sha(receipt_path),uses_future_reference_inputs=False,cuda_initialized=torch.cuda.is_initialized()))
    print('Joint K=1,2,3,4 causal state contracts and finite backwards passed',flush=True)

if __name__=='__main__':run()
