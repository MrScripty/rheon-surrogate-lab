"""Actual trained student fields against consumed native labels; GPU only."""
import argparse
import collections
import hashlib
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from support import ROOT,read,write,digest
from absolute_pressure_graph_v1 import AbsolutePressureNativeEntry as AbsolutePressureGenerator
from cuda_native_transfer_v1 import FIELDS
from diagnose_causal_composition import inventory,measure


def hashes(fields):return {n:hashlib.sha256(v.numpy().tobytes()).hexdigest() for n,v in fields.items()}


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('Fresh GPU student qualification required')
    reg=read(args.protocol)
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Frozen student dependency changed: '+p)
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AbsolutePressureGenerator(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build'])
    before=inventory(model);norm=read(ROOT/reg['normalization']);cases=read(ROOT/reg['case_selection'])['cases']
    args.output.mkdir(parents=True);started=time.monotonic()
    result=dict(status='running',active=None,rows=[],summaries=[],snapshots=[],source_evidence={},
        inference_device='cuda',display_label='GPU (CUDA)',gpu_model=torch.cuda.get_device_name(),
        student_stages='OriginalN1/N8',model_identity=model.asset_identity,pressure_dtype='float64',carrier_dtype='float32',
        tf32=False,autocast=False,cpu_model_forward=False,optimizer_updates=0,latency_claim=False,
        expanded_reserved52_references_opened=0,new_independent_confirmation=False,full240history=False,
        source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol))
    def save():write(args.output/'results.json',result)
    try:
        for case in cases:
            if digest(ROOT/case['receipt'])!=case['receipt_sha256']:raise ValueError('Consumed native scene changed')
            scene=read(ROOT/case['receipt'])['scene'];specs={s['step']:s for s in case['states']};saved={}
            def native(step):
                spec=specs[step];p=ROOT/spec['path']
                if digest(p)!=spec['sha256']:raise ValueError('Native current/label bytes changed')
                result['source_evidence'][str(p.relative_to(ROOT))]=spec['sha256']
                with np.load(p,allow_pickle=False) as a:return {n:a[n].copy() for n in (*FIELDS,'rhs')}
            for start in case['diagnostic_starts']:
                initial=native(start);fields={n:torch.from_numpy(initial[n]) for n in FIELDS};input_hash=hashes(fields);prefix=[]
                for k in case['diagnostic_horizons']:
                    if time.monotonic()-started>reg['maximum_wall_seconds']:raise TimeoutError('Finite student native evaluation deadline')
                    result['active']=dict(identifier=case['identifier'],start=start,K=k);save()
                    model.prepare_native(fields,scene,start,k,case['reference_substeps'])
                    outputs=model.predict_native(fields,scene,start,k,case['reference_substeps'])
                    if len(outputs)!=k or hashes(fields)!=input_hash:raise ValueError('Complete causal outputs and unchanged caller fields required')
                    identities=[hashes(o) for o in outputs]
                    if identities[:len(prefix)]!=prefix:raise ValueError('Student common physical prefix changed acrossK')
                    prefix=identities
                    for lead,out in enumerate(outputs,1):
                        truth=native(start+lead)
                        if set(out)!=set(FIELDS) or any(v.device.type!='cpu' or v.dtype!=(torch.float64 if n=='pressure' else torch.float32) or tuple(v.shape)!=truth[n].shape for n,v in out.items()):
                            raise ValueError('Complete typed native reconstruction required')
                        nonfinite={n:int((~v.isfinite()).sum()) for n,v in out.items()};metrics=None;error=None
                        if not any(nonfinite.values()):
                            try:metrics=measure(out,truth,scene,norm)
                            except ValueError as e:error=repr(e)
                        result['rows'].append(dict(identifier=case['identifier'],split=case['split'],group=case['group'],
                            counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],start=start,K=k,lead=lead,step=start+lead,
                            teacher_substeps=case['reference_substeps'],student_stages=case['reference_substeps'],
                            actual_internal_advances_per_block=k*case['reference_substeps'],
                            metrics=metrics,metric_error=error,nonfinite_counts=nonfinite,
                            all_accuracy_pass=bool(metrics and all(v<=reg['maximum_error'] for v in metrics['relative'].values())),output_sha256=identities[lead-1]))
                        if case['reference_substeps']==8 and k==max(case['diagnostic_horizons']):
                            for name,v in out.items():saved[f'start{start}_lead{lead}_{name}']=v.numpy().copy()
            if saved:
                path=args.output/(case['identifier']+'-student-fields.npz');np.savez_compressed(path,**saved)
                result['snapshots'].append(dict(path=path.name,bytes=path.stat().st_size,sha256=digest(path)))
            if (torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>reg['maximum_peak_rss_bytes']
                    or sum(s['bytes'] for s in result['snapshots'])>reg['maximum_output_bytes']):
                raise ValueError('Student qualification resource ceiling exceeded')
            model.release_graphs()
            save();print('Absolute learned pressure generator',case['identifier'],'all native physical outputs scored',flush=True)
        eager=read(ROOT/reg['eager_results'])
        expected={(r['identifier'],r['start'],r['K'],r['lead']):r['output_sha256'] for r in eager['rows']}
        if eager['status']!='completed_absolute_pressure_full_state_blocks' or not eager['all_accuracy_pass']:
            raise ValueError('Complete native-scored eager absolute-pressure candidate required')
        if any(r['output_sha256']!=expected[(r['identifier'],r['start'],r['K'],r['lead'])] for r in result['rows']):
            raise ValueError('Graph/native entry changed eager absolute-pressure field bits')
        result['all1018_eager_output_hashes_exact']=True
        if len(result['rows'])!=reg['expected_complete_outputs'] or inventory(model)!=before:raise ValueError('Full registered population and unchanged student weights required')
        groups=collections.defaultdict(list)
        for row in result['rows']:groups[(row['split'],row['K'],row['cells_per_metre'],tuple(row['counts_xyz']),row['teacher_substeps'])].append(row)
        result['summaries']=[dict(split=split,K=k,cells_per_metre=cpm,counts_xyz=list(grid),teacher_substeps=n,complete_outputs=len(rows),
            failures=sum(not r['all_accuracy_pass'] for r in rows),all_accuracy_pass=all(r['all_accuracy_pass'] for r in rows),
            worst_relative={f:max((r['metrics']['relative'][f] for r in rows if r['metrics']),default=None) for f in ['equation','pressure','velocity','tracer']})
            for (split,k,cpm,grid,n),rows in sorted(groups.items())]
        for p,h in reg['source_sha256'].items():
            if digest(ROOT/p)!=h:raise ValueError('Student source changed during evaluation: '+p)
        result.update(status='completed_absolute_pressure_graph_native_blocks',active=None,
            all_accuracy_pass=all(r['all_accuracy_pass'] for r in result['rows']),every_K_intermediate_scored=True,
            causal_prefix_and_input_ownership_pass=True,learned_parameters_unchanged=True,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            elapsed_diagnostic_seconds=time.monotonic()-started,full_generator_target_pass=False)
    except BaseException as e:result.update(status='failed',error=repr(e));raise
    finally:model.close();save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
