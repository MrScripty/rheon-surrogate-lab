"""RED-capable actual0->32 replay for the new64cubed100cells/metre failure."""
import argparse
import hashlib
import os
import sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from support import ROOT, read, write, digest
from debug.cube128_exact_transform_candidate_v2 import Cube128ExactTransformCandidate
from debug.affine_generator_graph_v2 import AffineGeneratorGraph
from cuda_native_transfer_v1 import FIELDS, upload_state, download_outputs
from diagnose_causal_composition import inventory, measure


def run(args):
    if args.output.exists(): raise ValueError('Fresh regression evidence required')
    reg = read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda': raise ValueError('GPU-only learned inference required')
    for p, h in reg['source_sha256'].items():
        if digest(ROOT/p) != h: raise ValueError('Frozen replay input changed: '+p)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    case = reg['case']; scene = read(ROOT/case['receipt'])['scene']; n = case['reference_substeps']
    observation = read(ROOT/reg['observation'])
    if observation['status'] != 'completed_streamed_expanded_gpu_240step_scene': raise ValueError('Original failure must finish first')
    expected = {r['step']: r['output_sha256'] for r in observation['rows'] if r['K']==1}
    reference = {r['step']: r['actual_field_sha256'] for r in observation['reference_rows']}
    initial_spec = next(s for s in case['states'] if s['step']==0)
    with np.load(ROOT/initial_spec['path'], allow_pickle=False) as a:
        initial = {name:torch.from_numpy(a[name].copy()) for name in FIELDS}
    with np.load(ROOT/reg['native_endpoint'], allow_pickle=False) as a:
        truth = {name:a[name].copy() for name in (*FIELDS,'rhs')}
    norm = read(ROOT/reg['normalization']); rows=[]
    args.output.mkdir(parents=True)
    for label, cls, build in [('original_math',Cube128ExactTransformCandidate,reg['baseline_build']),
                              ('affine_fused',AffineGeneratorGraph,reg['build'])]:
        model = cls(reg['transport_pair'],reg['shared_pair'],build,reg['control_build']); before=inventory(model)
        try:
            with torch.inference_mode():
                current = upload_state(initial); model.prepare_geometry(tuple(scene['counts']))
                model.prepare_block(current,scene,0,1,n)
                actual_rows=[]
                for start in range(reg['endpoint_step']):
                    current=model.predict(current,scene,start,1,n)[0]
                    emitted=download_outputs([current])[0]
                    hashes={name:hashlib.sha256(v.numpy().tobytes()).hexdigest() for name,v in emitted.items()}
                    actual_rows.append(dict(step=start+1,output_sha256=hashes,original_full_run_exact=hashes==expected[start+1],
                        native_nonidentical_fields=[name for name in FIELDS if hashes[name]!=reference[start+1][name]]))
                metrics=measure(emitted,truth,scene,norm)
            if inventory(model)!=before:raise ValueError('Replay changed restored learned tensors')
            rows.append(dict(label=label,build=build,steps=actual_rows,metrics=metrics,
                original_full_run_exact=all(r['original_full_run_exact'] for r in actual_rows),
                accuracy_pass=all(v<=reg['maximum_error'] for v in metrics['relative'].values())))
        finally:model.close()
    write(args.output/'results.json',dict(status='passed_target' if all(r['accuracy_pass'] for r in rows) else 'reproduced_actual_cube64_density100_equation_failure',
        rows=rows,endpoint_step=reg['endpoint_step'],inference_device='cuda',display_label='GPU (CUDA)',gpu_model=torch.cuda.get_device_name(),
        native_teacher_executed=False,cpu_model_forward=False,optimizer_updates=0,latency_claim=False,expanded_reserved52_references_opened=0,
        protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256']))
    for row in rows:print('RED-capable actual0->32',row['label'],'equation',row['metrics']['relative']['equation'],'original bytes',row['original_full_run_exact'],flush=True)
    if not all(r['accuracy_pass'] for r in rows):raise AssertionError('64cubed100cells/metre actual own-history equation error exceeds1e-5')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
