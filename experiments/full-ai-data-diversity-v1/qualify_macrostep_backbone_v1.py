"""Exact existing S1/S8 outputs and finite S3/S4 feature/readout controls."""
import argparse
import hashlib
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from support import ROOT,read,write,digest
from macrostep_backbone_v1 import FrozenMacroBackbone
from macrostep_readout_v1 import MacroReadout
from cuda_native_transfer_v1 import FIELDS
from diagnose_causal_composition import inventory


def hashes(fields):
    return {n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in fields.items()}


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('Fresh GPU qualification required')
    reg=read(args.protocol)
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Frozen macrostep source changed: '+p)
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    gold=read(ROOT/reg['control_results']);cases=read(ROOT/reg['case_selection'])['cases']
    if gold['status']!='completed_macrostep_GPU_untrained_controls' or len(gold['rows'])!=196:raise ValueError('Actual original controls required')
    expected={(r['identifier'],r['start'],r['K'],r['lead'],r['mode']):r['output_sha256'] for r in gold['rows']}
    model=FrozenMacroBackbone(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build']);before=inventory(model)
    heads={s:MacroReadout(s) for s in (3,4)}
    args.output.mkdir(parents=True);started=time.monotonic()
    result=dict(status='running',rows=[],feature_rows=[],inference_device='cuda',display_label='GPU (CUDA)',
        gpu_model=torch.cuda.get_device_name(),optimizer_updates=0,cpu_model_forward=False,latency_claim=False,
        expanded_reserved52_references_opened=0,protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'])
    def save():write(args.output/'results.json',result)
    try:
        with torch.no_grad():
            for case in cases:
                scene=read(ROOT/case['receipt'])['scene'];specs={s['step']:s for s in case['states']}
                for start in case['diagnostic_starts']:
                    spec=specs[start]
                    if digest(ROOT/spec['path'])!=spec['sha256']:raise ValueError('Actual current fields changed')
                    with np.load(ROOT/spec['path'],allow_pickle=False) as a:
                        fields={n:torch.from_numpy(a[n].copy()).to('cuda') for n in FIELDS}
                    for k in case['diagnostic_horizons']:
                        for mode,s in [('teacher_schedule_N8_control',8),('untrained_macro_N1_control',1)]:
                            packet=model.packet(scene,start,k,s)
                            if not torch.equal(packet,model.controls.parameters(scene,start,k,s)):
                                raise ValueError('Known input packet bytes changed for existing stage counts')
                            blocks=model.feature_states(fields,scene,start,s,k)
                            for lead,features in enumerate(blocks,1):
                                identity=hashes(features[-1])
                                if identity!=expected[(case['identifier'],start,k,lead,mode)]:raise ValueError('Frozen stage port changed actual original output bytes')
                                result['rows'].append(dict(identifier=case['identifier'],start=start,K=k,lead=lead,stages=s,all_output_bits_exact=True,output_sha256=identity))
                        del blocks
                    for s,head in heads.items():
                        features=model.feature_states(fields,scene,start,s,1)[0]
                        if len(features)!=s+1 or any(not bool(v.isfinite().all()) for f in features for v in f.values()):
                            raise ValueError('Complete finite stage features required')
                        emitted=head.read(features)
                        if hashes(emitted)!=hashes(features[-1]):raise ValueError('Zero readout weights do not preserve exact anchor fields')
                        result['feature_rows'].append(dict(identifier=case['identifier'],start=start,stages=s,feature_states=len(features),finite=True,zero_head_exact_identity=True))
                    if time.monotonic()-started>reg['maximum_wall_seconds']:raise TimeoutError('Finite backbone qualification deadline')
                if (torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>reg['maximum_peak_rss_bytes']):
                    raise ValueError('Backbone/readout memory cap exceeded')
                save();print('Frozen macrostep backbone',case['identifier'],'S1/S8bytes and S3/S4features passed',flush=True)
        if len(result['rows'])!=196 or len(result['feature_rows'])!=124 or inventory(model)!=before:
            raise ValueError('Complete backbone population and unchanged frozen parameters required')
        for p,h in reg['source_sha256'].items():
            if digest(ROOT/p)!=h:raise ValueError('Macrostep source changed during qualification: '+p)
        result.update(status='passed_macrostep_frozen_backbone_and_zero_readouts',existing_complete_outputs_exact=196,
            feature_blocks=124,known_S1_S8_packets_exact=True,all_zero_readouts_exact=True,frozen_parameters_unchanged=True,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            elapsed_diagnostic_seconds=time.monotonic()-started)
    except BaseException as e:result.update(status='failed',error=repr(e));raise
    finally:model.close();save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
