"""Export fixed learned pressure coefficients for a future causal full-state model.

This derives coefficients from qualified weights only. It executes no optimizer,
physical inverse, reference capture or temporal prediction. The four geometry
caches are responses of the existing learned shape heads, not analytical gains.
"""
import argparse
import inspect
import math
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from materialized_fine_precise import SpatialDouble
from study_support import ROOT,HERE,DOCS,read,write,sha,load_checkpoint


def describe(module):
    if isinstance(module,nn.Conv3d):
        if module.bias is not None:raise ValueError('Qualified bias-free spatial convolution required')
        return dict(type='Conv3d',in_channels=module.in_channels,out_channels=module.out_channels,
            kernel_size=list(module.kernel_size),stride=list(module.stride),padding=list(module.padding),
            dilation=list(module.dilation),groups=module.groups,bias=False,padding_mode=module.padding_mode)
    if isinstance(module,nn.Identity):return dict(type='Identity')
    if isinstance(module,nn.AdaptiveAvgPool3d):return dict(type='AdaptiveAvgPool3d',output_size=module.output_size)
    if isinstance(module,(nn.Sequential,nn.ModuleList)):
        return dict(type='Sequential'if isinstance(module,nn.Sequential)else'ModuleList',
            children={name:describe(child)for name,child in module.named_children()})
    if hasattr(module,'layers')and isinstance(module.layers,nn.Sequential):
        return dict(type='LayerContainer',layers=describe(module.layers))
    raise ValueError('Unqualified fine-network layer '+type(module).__name__)


def run(args):
    if args.output.exists():raise ValueError('Fresh frozen pressure export required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    selection=read(DOCS/'pressure-confirmation-v2/selection.json')
    if any(sha(HERE/name)!=digest for name,digest in selection['source_sha256'].items()):
        raise ValueError('Confirmed pressure source changed')
    state=load_checkpoint('a-tail8');fine=SpatialDouble(state.network).float().eval()
    roots=('encoders','bottleneck','global_pool','decoders','head')
    architecture={name:describe(getattr(fine,name))for name in roots}
    weights={name:value.detach().cpu().numpy().astype('<f4',copy=True)for name,value in fine.state_dict().items()}
    if any(not np.isfinite(value).all()for value in weights.values()):raise ValueError('Nonfinite learned spatial weights')
    import physical_residual as physical
    scales=physical.historical_physical.scales(state.normalization)
    gains={};geometries=[]
    with torch.inference_mode():
        for index,counts in enumerate(((16,16,16),(11,11,9),(8,6,15),(32,32,32))):
            features=torch.tensor([math.log(n/16)for n in counts],dtype=torch.float32)
            logits=state.network.logits_for_shape(features,counts)
            positive=max(counts)**2/2*torch.sigmoid(logits.double())
            low=torch.cat((positive.new_zeros(1),positive)).reshape(4,4,4)
            gains[f'g{index}_low']=low.numpy().astype('<f8',copy=True)
            use_tail=max(counts)<=16
            if use_tail:
                positive=max(counts)**2/2*torch.sigmoid((state.network.tail_logits+state.network.tail_shape_head(features)).double())
                tail=torch.cat((positive.new_zeros(1),positive)).reshape(8,8,8)
                gains[f'g{index}_tail']=tail.numpy().astype('<f8',copy=True)
            geometries.append(dict(counts_zyx=list(counts),low_gain_key=f'g{index}_low',
                tail_gain_key=f'g{index}_tail'if use_tail else None))
    args.output.mkdir(parents=True)
    np.savez(args.output/'fine-weights.npz',**weights)
    np.savez(args.output/'geometry-gains.npz',**gains)
    config=dict(schema='rheon-frozen-learned-pressure-core-assets-v1',architecture=architecture,geometries=geometries,
        steps=state.model.CORRECTION['steps'],laplacian_rhs_scale=scales['laplacian_rhs'],
        fine_output_scale=scales[state.normalization.get('output_scale','pressure')],
        fine_gauge_correction_source=inspect.getsource(state.model.owner.frozen.gauge_correction),
        spatial_parameters=sum(value.size for value in weights.values()),parent_learned_parameters=128938,
        gains_are_cached_learned_head_outputs=True,unsupported_geometry='Reject; export/qualify new geometry from parent weights first',
        construction_reference_fields_used=False,optimizer_updates=0,
        parent_entry=state.entry,parent_confirmation_selection_sha256=sha(DOCS/'pressure-confirmation-v2/selection.json'))
    write(args.output/'configuration.json',config)
    files={name:dict(bytes=(args.output/name).stat().st_size,sha256=sha(args.output/name))
           for name in ('fine-weights.npz','geometry-gains.npz','configuration.json')}
    write(args.output/'receipt.json',dict(status='exported_for_qualification',files=files,source_sha256=sha(__file__),
        parent='a-tail8',optimizer_updates=0,local_derived_assembly=True,managed_parent_provenance_retained=True,
        full_state_temporal_candidate=False,CPU_latency_claim=False,new_accuracy_claim=False,
        source_pressure_files_unchanged=True,physical_inverse_used_in_export=False,
        inference_pressure_algorithm='Existing fixed-depth learned pressure network; standalone implementation not qualified yet'))
    print('Exported frozen learned pressure coefficients and four geometry-head caches; zero updates, no reference data',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
