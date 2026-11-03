"""Matched real-request inference execution experiment; no weight updates."""
import argparse
import copy
import os
import statistics
import time
import warnings
from v4_support import ROOT, BASE, read, write, sha, load, native, summary, pressure_error

def run(args):
    if args.output.exists(): raise ValueError('Fresh execution benchmark required')
    os.sched_setaffinity(0,{23})
    state,wrapper,parity=load()
    import torch, cases, scalable
    from frozen_matrix import run_policy
    inventory=state.data.helpers().state_inventory(state.network)
    population=native(state)
    if args.screen:
        population=[r for r in population if r[0]['cells_per_metre']==16 and r[0]['frame']==0]
        old=cases.load(state,include_old_large=False)
        population=[next(r for r in old if r[0]['population']=='consumed128' and max(r[0]['counts_xyz'])==16)]+population
    branch={k:v.detach() for k,v in state.network.state_dict().items() if k=='coarse_logits' or k.startswith('shape_head.')}
    rows=[]; qualifications=[]
    started=time.perf_counter()
    with torch.inference_mode():
        for identity,sample in population:
            scales=state.normalization['scales']; h=sample['cell_size_m'];rho=sample['rho']
            example=(sample['native_input_residual']*(rho/h).reshape(1,1,1,1,1)/scales['laplacian_rhs']['value']).float()
            eager=copy.deepcopy(wrapper).eval()
            last=copy.deepcopy(wrapper).eval().to(memory_format=torch.channels_last_3d)
            class ChannelsLast(torch.nn.Module):
                def forward(self,values):return last(values.contiguous(memory_format=torch.channels_last_3d))
            tick=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                traced=torch.jit.freeze(torch.jit.trace(eager,example,check_trace=True))
            compile_seconds=time.perf_counter()-tick
            variants=dict(eager=eager,channels_last=ChannelsLast().eval(),traced_frozen=traced)
            reference=eager(example)
            for name,variant in variants.items():
                actual=variant(example)
                difference=float(torch.linalg.vector_norm(actual-reference))/max(float(torch.linalg.vector_norm(reference)),1e-300)
                if difference>1e-5:raise ValueError('Inference execution changed forward beyond declared diagnostic tolerance')
                qualifications.append(dict(**identity,execution=name,relative_forward_difference=difference,
                    bitwise_forward_parity=bool(torch.equal(reference,actual)),compile_seconds=compile_seconds if name=='traced_frozen'else 0,
                    trace_warnings=sorted({str(w.message) for w in caught}) if name=='traced_frozen'else []))
            class Clocked(torch.nn.Module):
                def __init__(self,target):super().__init__();self.target=target;self.seconds=0.;self.calls=0
                def forward(self,values):
                    tick=time.perf_counter();out=self.target(values);self.seconds+=time.perf_counter()-tick;self.calls+=1;return out
            for variant in variants.values():run_policy(variant,sample,state.normalization,branch,'learned',max_passes=8)
            for rep,names in enumerate([list(variants),list(reversed(variants)),list(variants)]):
                for name in names:
                    target=Clocked(variants[name]);tick=time.perf_counter()
                    pressure,work=run_policy(target,sample,state.normalization,branch,'learned',max_passes=8)
                    seconds=time.perf_counter()-tick
                    rows.append(dict(**identity,execution=name,repeat=rep,seconds=seconds,cnn_seconds=target.seconds,
                        **work,pressure_relative_l2=pressure_error(pressure,sample)))
            print('Measured',identity['counts_xyz'],identity['frame'],flush=True)
    if inventory!=state.data.helpers().state_inventory(state.network) or torch.cuda.is_initialized():raise ValueError('Original state modified')
    summaries=[]
    for name in ['eager','channels_last','traced_frozen']:
        selected=[r for r in rows if r['execution']==name]
        summaries.append(dict(execution=name,**summary(selected),median_cnn_fraction=statistics.median(r['cnn_seconds']/r['seconds']for r in selected)))
    write(args.output,dict(status='completed',entry=state.entry,cpu=23,threads=1,screen=args.screen,
        wrapper_parity=parity,qualifications=qualifications,rows=rows,summary=summaries,
        optimizer_updates=0,unchanged_original_inventory=True,source_sha256=sha(__file__),
        runtime=dict(torch=torch.__version__,mkldnn=bool(torch.backends.mkldnn.enabled)),elapsed_s=time.perf_counter()-started,
        scope='Per-shape traced graphs, all input conversions and adaptive policy work included. Compilation recorded separately. Three balanced serial repeats after warmup. No coupled speed claim.'))
    print(summaries,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=__import__('pathlib').Path,required=True)
    parser.add_argument('--screen',action='store_true')
    run(parser.parse_args())
