"""DEBUG: GPU own-history replay, changing only the known-force provider."""
import argparse,hashlib,os,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from support import ROOT,read,write,digest
from fused_cuda_graph_v1 import FusedCudaGraphGenerator
from dynamic_cuda_controls_v1 import CudaKnownControls
from expanded_cuda_generator_v1 import prescribed_controls,FIELDS
from cuda_native_transfer_v1 import upload_state,download_outputs
from diagnose_causal_composition import inventory,measure

def run(args):
    if args.output.exists():raise ValueError('Fresh force-source diagnostic required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('Every learned stage must run onGPU')
    for path,want in reg['source_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('Frozen force-source probe changed: '+path)
    os.sched_setaffinity(0,{4});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    case=reg['case'];scene=dict(read(ROOT/case['receipt'])['scene'],reference_substeps=case['reference_substeps'],forcing_control_substeps=1)
    spec=next(s for s in case['states'] if s['step']==0)
    with np.load(ROOT/spec['path'],allow_pickle=False) as a:initial={name:torch.from_numpy(a[name].copy()) for name in FIELDS}
    f=read(ROOT/reg['fixture_receipt'])
    with np.load(ROOT/f['reference_path'],allow_pickle=False) as a:truth={name:a[name].copy() for name in (*FIELDS,'rhs')}
    expected=read(ROOT/reg['observation'])['first_failing_K1']['output_sha256'];norm=read(ROOT/reg['normalization'])
    args.output.mkdir(parents=True);rows=[];control=CudaKnownControls(reg['control_build'])
    try:
        for mode in ('gpu_dynamic','native_cpu_known_input'):
            model=FusedCudaGraphGenerator(reg['transport_pair'],reg['shared_pair'],reg['build'],'transport_and_linear');before=inventory(model)
            try:
                with torch.inference_mode():
                    current=upload_state(initial);model.prepare_geometry(tuple(scene['counts']))
                    supplier=control.generate if mode=='gpu_dynamic' else prescribed_controls
                    external=supplier(scene,0,1,case['reference_substeps']);model.prepare_block(current,scene,0,1,external,case['reference_substeps'])
                    first_hash=None
                    for start in range(reg['endpoint_step']):
                        current=model.predict(current,scene,start,1,supplier(scene,start,1,case['reference_substeps']),case['reference_substeps'])[0]
                        if start==0:first_hash={name:hashlib.sha256(v.numpy().tobytes()).hexdigest() for name,v in download_outputs([current])[0].items()}
                    emitted=download_outputs([current])[0];actual={name:hashlib.sha256(v.numpy().tobytes()).hexdigest() for name,v in emitted.items()}
                    if mode=='gpu_dynamic' and actual!=expected:raise ValueError('Control-isolation baseline does not exactly reproduce original complete-graph endpoint')
                    metrics=measure(emitted,truth,scene,norm)
                    if inventory(model)!=before:raise ValueError('Probe changed learned weights')
                    rows.append(dict(mode=mode,metrics=metrics,all_accuracy_pass=all(v<=reg['maximum_error'] for v in metrics['relative'].values()),
                        original_endpoint_bitwise_equal=actual==expected,first_step_output_sha256=first_hash,endpoint_output_sha256=actual))
                    print('One-variable force probe',mode,'endpoint equation',metrics['relative']['equation'],flush=True)
            finally:model.close()
        for path,want in reg['source_sha256'].items():
            if digest(ROOT/path)!=want:raise ValueError('Probe dependency changed: '+path)
        write(args.output/'results.json',dict(status='completed_one_variable_known_force_source_probe',rows=rows,
            learned_inference_device='cuda',known_external_input_devices=['cuda','cpu'],cpu_model_forward=False,optimizer_updates=0,
            both_modes_same_unchanged_learned_core_graph=True,protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'],latency_claim=False))
    finally:control.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
