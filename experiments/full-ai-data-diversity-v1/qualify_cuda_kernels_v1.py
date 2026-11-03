"""CUDA primitive parity against the unchanged learned tensor implementation."""
import argparse
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from support import ROOT,read,write,digest
from expanded_cuda_generator_v2 import ExpandedCudaGeneratorV2
from fused_cuda_generator_v1 import FusedCudaGenerator
from recurrent_sampler_native import dimensions
from diagnose_causal_composition import inventory


def run(args):
    if args.output.exists():raise ValueError('Fresh native CUDA parity evidence required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('CUDA-only parity qualification required')
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Frozen CUDA qualification dependency changed: '+path)
    os.sched_setaffinity(0,{4});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(reg['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ExpandedCudaGeneratorV2(reg['pair'],reg['shared_pair'])
    fused=FusedCudaGenerator(reg['pair'],reg['shared_pair'],reg['build'],'transport_and_linear')
    before=inventory(fused);reference=inventory(base)
    if any(before[name]!=value for name,value in reference.items()):raise ValueError('Learned tensors differ before execution')
    args.output=args.output.resolve();args.output.mkdir(parents=True);rows=[]
    def compare(name,left,right,context):
        bits=torch.int32 if left.dtype==torch.float32 else torch.int64
        equal=torch.equal(left.contiguous().view(bits),right.contiguous().view(bits))
        row=dict(operation=name,context=context,shape=list(left.shape),dtype=str(left.dtype),bitwise_equal=equal)
        if not equal:
            difference=left.double()-right.double();row.update(mismatched_values=int((left!=right).sum()),maximum_absolute_error=float(difference.abs().max()),
                signed_zero_or_bit_differences=int((left.contiguous().view(bits)!=right.contiguous().view(bits)).sum()))
        rows.append(row)
        return equal
    try:
        with torch.inference_mode():
            for selected in read(ROOT/reg['bank'])['trajectories']:
                receipt=read(ROOT/selected['receipt']);scene=receipt['scene'];counts=tuple(scene['counts']);shapes=dimensions(counts)
                base.prepare_geometry(counts);fused.prepare_geometry(counts);states={s['step']:s for s in selected['states']}
                for step in reg['states']:
                    spec=states[step];path=ROOT/spec['path']
                    if digest(path)!=spec['sha256']:raise ValueError('Actual current-field reference changed')
                    with np.load(path,allow_pickle=False) as archive:
                        fields=[torch.from_numpy(archive[name].copy()).to('cuda').reshape(shape) for name,shape in zip(('vx','vy','vz','tracer'),shapes)]
                        pressure=torch.from_numpy(archive['pressure'].copy()).to('cuda').reshape(tuple(reversed(counts)))
                    context=dict(identifier=selected['identifier'],step=step,counts_xyz=list(counts),cells_per_metre=scene['cells_per_metre'])
                    alpha=(1/(60*selected['reference_substeps']))*scene['cells_per_metre']
                    expected=base.transport(fields,counts,alpha);actual=fused.transport(fields,counts,alpha)
                    for kind,(left,right) in enumerate(zip(expected,actual)):compare('transport'+str(kind),left,right,context)
                    for head,size in enumerate(shapes[:3]):
                        compare('viscosity'+str(head),base.linear(fields[:3],size,'viscosity',head),fused.linear(fields[:3],size,'viscosity',head),context)
                        compare('momentum'+str(head),base.linear([pressure],size,'momentum',head),fused.linear([pressure],size,'momentum',head),context)
                    target=tuple(reversed(counts))
                    for channel in (None,0,1,2):
                        compare('float_flux'+str(channel),base.linear(fields[:3],target,'flux',0,channel),fused.linear(fields[:3],target,'flux',0,channel),context)
                    # Generic double-input reduction, including nonphysical random
                    # fixtures, proves dtype/sign/channel handling beyond pressure.
                    random_fields=[torch.randn(shape,dtype=torch.float64,device='cuda') for shape in shapes[:3]]
                    for channel in (None,0,1,2):
                        compare('double_negative_flux'+str(channel),base.linear(random_fields,target,'flux',0,channel,-1),fused.linear(random_fields,target,'flux',0,channel,-1),context)
                print('CUDA primitive parity',selected['identifier'],'complete',flush=True)
            if inventory(fused)!=before or inventory(base)!=reference:raise ValueError('Qualification changed learned tensors')
            all_equal=all(row['bitwise_equal'] for row in rows)
        for path,expected in reg['source_sha256'].items():
            if digest(ROOT/path)!=expected:raise ValueError('Parity source changed during execution: '+path)
        write(args.output/'results.json',dict(status='passed' if all_equal else 'failed_primitive_bitwise_parity',rows=rows,
            every_tensor_bitwise_equal=all_equal,primitive_tensors=len(rows),inference_device='cuda',gpu_model=torch.cuda.get_device_name(),
            model_identity=fused.asset_identity,optimizer_updates=0,cpu_model_forward=False,latency_claim=False,
            protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256']))
        if not all_equal:raise RuntimeError('Native learned CUDA primitive output differs from tensor control')
    finally:fused.close();base.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
