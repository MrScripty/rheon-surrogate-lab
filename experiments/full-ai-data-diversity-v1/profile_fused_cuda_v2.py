"""Bounded GPU execution and input/output diagnosis; not a target benchmark."""
import argparse
import collections
import os
import statistics
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from support import ROOT,read,write,digest
from fused_cuda_graph_v1 import FusedCudaGraphGenerator
from expanded_cuda_generator_v1 import FIELDS
from cuda_known_controls_v2 import CudaKnownControls
from diagnose_causal_composition import inventory

def run(args):
    if args.output.exists():raise ValueError('Fresh profiling directory required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('GPU-only execution required')
    for path,want in reg['source_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('Frozen profile source changed: '+path)
    prerequisite=read(ROOT/reg['qualification'])
    if not prerequisite['all_accuracy_pass'] or not prerequisite['every_output_bitwise_equal_corrected_baseline']:
        raise ValueError('Complete corrected fused output qualification required')
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    model=FusedCudaGraphGenerator(reg['pair'],reg['pressure_pair'],reg['build'],'transport_and_linear');before=inventory(model)
    control_owner=CudaKnownControls(reg['control_build'])
    args.output.mkdir(parents=True);rows=[]
    population={r['identifier']:r for r in read(ROOT/reg['bank'])['trajectories']}
    def wall(call):
        torch.cuda.synchronize();tick=time.perf_counter();value=call();torch.cuda.synchronize()
        return value,1000*(time.perf_counter()-tick)
    def identities(outputs):
        import hashlib
        return [{name:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for name,v in r.items()} for r in outputs]
    try:
        with torch.inference_mode():
            for selected_id in reg['identifiers']:
                selected=population[selected_id];receipt=read(ROOT/selected['receipt']);scene=receipt['scene'];n=selected['reference_substeps']
                spec=next(v for v in selected['states'] if v['step']==reg['start']);path=ROOT/spec['path']
                if digest(path)!=spec['sha256']:raise ValueError('Actual source state changed')
                with np.load(path,allow_pickle=False) as a:host={name:torch.from_numpy(a[name].copy()) for name in FIELDS}
                state={name:v.to('cuda') for name,v in host.items()};model.prepare_geometry(tuple(scene['counts']))
                for k in reg['horizons']:
                    controls=control_owner.generate(scene,reg['start'],k,n);model.prepare_block(state,scene,reg['start'],k,controls,n)
                    for _ in range(reg['warmup_blocks']):model.predict(state,scene,reg['start'],k,controls,n)
                    emitted=model.predict(state,scene,reg['start'],k,controls,n);expected=identities(emitted)
                    durations=collections.defaultdict(list)
                    actions=dict(known_control_construct_and_upload=lambda:control_owner.generate(scene,reg['start'],k,n),
                        current_state_upload=lambda:{name:v.to('cuda') for name,v in host.items()},
                        complete_gpu_resident_generate=lambda:model.predict(state,scene,reg['start'],k,controls,n),
                        complete_output_download=lambda:[{name:v.detach().cpu().contiguous() for name,v in r.items()} for r in emitted])
                    names=list(actions)
                    for repeat in range(reg['repetitions']):
                        for name in names[repeat%len(names):]+names[:repeat%len(names)]:
                            values,elapsed=wall(actions[name]);durations[name].append(elapsed)
                            if name=='complete_gpu_resident_generate' and identities(values)!=expected:raise ValueError('Profile actual output changed')
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(reg['profiled_blocks']):
                            with torch.profiler.record_function('known_control_construct_and_upload'):
                                supplied=control_owner.generate(scene,reg['start'],k,n)
                            with torch.profiler.record_function('complete_learned_generator'):
                                model.predict(state,scene,reg['start'],k,supplied,n)
                        torch.cuda.synchronize()
                    events=[v for v in prof.events() if v.device_type==torch.autograd.DeviceType.CUDA]
                    kinds=collections.Counter(v.name for v in events);totals=collections.defaultdict(float)
                    for event in events:totals[event.name]+=event.device_time_total
                    rows.append(dict(identifier=selected_id,counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],K=k,internal_substeps=n,
                        diagnostic_block_wall_ms={name:dict(samples=values,median=statistics.median(values),maximum=max(values)) for name,values in durations.items()},
                        cuda_activity_events_per_generated_step=len(events)/(reg['profiled_blocks']*k),
                        top_cuda_activities=[dict(name=name,calls=kinds[name],instrumented_total_ms=value/1000) for name,value in sorted(totals.items(),key=lambda item:item[1],reverse=True)[:15]],
                        actual_output_hashes=expected,top_cpu_activities=[dict(name=e.key,self_cpu_ms=e.self_cpu_time_total/1000) for e in sorted(prof.key_averages(),key=lambda e:e.self_cpu_time_total,reverse=True)[:12]]))
                    print('Fused GPU diagnostic',selected_id,'K',k,'activities/output',rows[-1]['cuda_activity_events_per_generated_step'],flush=True)
                model.release_graphs()
            if inventory(model)!=before:raise ValueError('Profile changed parameters')
        for path,want in reg['source_sha256'].items():
            if digest(ROOT/path)!=want:raise ValueError('Profile source changed: '+path)
        write(args.output/'results.json',dict(status='completed_fused_gpu_execution_and_io_diagnosis',rows=rows,inference_device='cuda',
            gpu_model=torch.cuda.get_device_name(),source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol),
            diagnostic_only=True,latency_acceptance_claim=False,components_not_assumed_additive=True,optimizer_updates=0,cpu_model_forward=False,
            expanded_reserved52_references_opened=0))
    finally:control_owner.close();model.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
