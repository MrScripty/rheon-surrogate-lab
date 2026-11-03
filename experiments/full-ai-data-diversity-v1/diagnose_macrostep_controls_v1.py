"""N8 teacher-schedule versus untrained N1 control; GPU only, no speed claim."""
import argparse
import hashlib
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from support import ROOT, read, write, digest
from complete_native_entry_v1 import CompleteNativeEntry
from cuda_native_transfer_v1 import FIELDS
from diagnose_causal_composition import inventory, measure


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':
        raise ValueError('Fresh GPU-only macrostep control diagnostic required')
    reg=read(args.protocol)
    for name,h in reg['source_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Frozen macrostep dependency changed: '+name)
    cases=read(ROOT/reg['case_selection'])['cases']
    if len(cases)!=30 or sum(c['split']=='train' for c in cases)!=29 or any(c['reference_substeps']!=8 for c in cases):
        raise ValueError('Exactly the29training and one consumed development N8cases required')
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=CompleteNativeEntry(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build'])
    before=inventory(model);norm=read(ROOT/reg['normalization']);started=time.monotonic()
    args.output.mkdir(parents=True)
    result=dict(status='running',active=None,rows=[],scene_summaries=[],source_evidence={},
        inference_device='cuda',display_label='GPU (CUDA)',gpu_model=torch.cuda.get_device_name(),
        pressure_dtype='float64',carrier_dtype='float32',tf32=False,autocast=False,
        cpu_model_forward=False,optimizer_updates=0,new_model_trained=False,latency_claim=False,
        production_qualification=False,full_generator_target_pass=False,expanded_reserved52_references_opened=0,
        own_history_inside_block=True,full240history=False,source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol))
    def save():write(args.output/'results.json',result)
    try:
        with torch.inference_mode():
            for case in cases:
                if digest(ROOT/case['receipt'])!=case['receipt_sha256']:raise ValueError('Consumed scene receipt changed')
                scene=read(ROOT/case['receipt'])['scene'];specs={s['step']:s for s in case['states']}
                def load(step):
                    spec=specs[step];p=ROOT/spec['path']
                    if digest(p)!=spec['sha256']:raise ValueError('Native diagnostic state changed')
                    result['source_evidence'][str(p.relative_to(ROOT))]=spec['sha256']
                    with np.load(p,allow_pickle=False) as archive:return {n:archive[n].copy() for n in (*FIELDS,'rhs')}
                beginning=len(result['rows'])
                for start in case['diagnostic_starts']:
                    initial=load(start);fields={n:torch.from_numpy(initial[n]) for n in FIELDS}
                    for k in case['diagnostic_horizons']:
                        emitted={}
                        for name,n in [('teacher_schedule_N8_control',8),('untrained_macro_N1_control',1)]:
                            if time.monotonic()-started>reg['maximum_wall_seconds']:raise TimeoutError('Finite macrostep control deadline')
                            result['active']=dict(identifier=case['identifier'],start=start,K=k,mode=name);save()
                            model.prepare_native(fields,scene,start,k,n)
                            emitted[name]=model.predict_native(fields,scene,start,k,n)
                            if len(emitted[name])!=k:raise ValueError('Every intermediate physical output required')
                        # Only now open future labels; inference receives current
                        # fields and known prescribed controls exclusively.
                        for lead in range(1,k+1):
                            truth=load(start+lead)
                            for name,n in [('teacher_schedule_N8_control',8),('untrained_macro_N1_control',1)]:
                                out=emitted[name][lead-1]
                                nonfinite={field:int((~value.isfinite()).sum()) for field,value in out.items()}
                                metrics=None;error=None
                                if not any(nonfinite.values()):
                                    try:metrics=measure(out,truth,scene,norm)
                                    except ValueError as e:error=repr(e)
                                result['rows'].append(dict(identifier=case['identifier'],split=case['split'],group=case['group'],
                                    counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],start=start,K=k,lead=lead,step=start+lead,
                                    mode=name,candidate_internal_advances_per_block=k*n,teacher_internal_advances_per_block=k*8,
                                    nonfinite_counts=nonfinite,metrics=metrics,metric_error=error,
                                    all_accuracy_pass=bool(metrics and metrics['all_below_1e5']),
                                    output_sha256={f:hashlib.sha256(v.numpy().tobytes()).hexdigest() for f,v in out.items()}))
                        model.release_graphs();save()
                rows=result['rows'][beginning:]
                result['scene_summaries'].append(dict(identifier=case['identifier'],split=case['split'],
                    counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],modes={name:dict(outputs=sum(r['mode']==name for r in rows),
                        failures=sum(r['mode']==name and not r['all_accuracy_pass'] for r in rows),
                        worst_relative={field:max((r['metrics']['relative'][field] for r in rows if r['mode']==name and r['metrics']),default=None)
                                        for field in ('equation','pressure','velocity','tracer')})
                        for name in ('teacher_schedule_N8_control','untrained_macro_N1_control')}))
                if (torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes']
                        or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>reg['maximum_peak_rss_bytes']):
                    raise ValueError('Macrostep diagnostic memory ceiling exceeded')
                save();print('Macrostep GPU control',case['identifier'],'all registered physical states scored',flush=True)
        if len(result['rows'])!=reg['expected_complete_outputs'] or inventory(model)!=before:raise ValueError('Complete diagnostic population and unchanged learned values required')
        for name,h in reg['source_sha256'].items():
            if digest(ROOT/name)!=h:raise ValueError('Macrostep dependency changed during execution: '+name)
        result.update(status='completed_macrostep_GPU_untrained_controls',active=None,
            learned_parameters_unchanged=True,all_registered_states_scored=True,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            elapsed_diagnostic_seconds=time.monotonic()-started)
    except BaseException as e:
        result.update(status='failed',error=repr(e));raise
    finally:model.close();result['active']=None;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
