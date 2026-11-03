"""Fixed uniform head-precision policies after exact raw managed CPU restoration."""
import argparse
from pathlib import Path
import torch
import evaluate_recurrent_sampler as evaluation
from restored_recurrent_sampler import load_model as restored_model
from profile_frozen_components import read,write


def construct(pair,library_path,step):
    model=restored_model(pair,library_path);changes={}
    with torch.no_grad():
        for name in ('mid_gate','mix'):
            value=getattr(model,name);rounded=(value/step).round()*step
            changes[name]=dict(maximum_absolute_change=float((rounded-value).abs().max()),nonzero_before=int(value.count_nonzero()),nonzero_after=int(rounded.count_nonzero()))
            value.copy_(rounded)
    model.frozen_versions={name:p._version for name,p in model.named_parameters()}
    model.asset_identity['transport_precision_policy']=dict(kind='uniform_absolute_quantization',step=step,scope='All48new double head parameters',
        changes=changes,source='Exact managed CPU-restored tensors before this fixed inference precision transform',optimizer_updates=0,analytic_coefficient_assignment=False)
    return model


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();reg=read(args.protocol);args.mode='rollout'
    evaluation.load_model=lambda pair,library_path:construct(pair,library_path,reg['quantization_step'])
    evaluation.run(args)
    result=read(args.output/'results.json');result.update(status='completed_uniform_head_precision_diagnostic',
        inference_precision_policy_changed=True,raw_managed_restore_unchanged=True,parameter_learning_executed=0,
        diagnostic_does_not_establish_all_horizon_or_independent_target_pass=True)
    write(args.output/'results.json',result)
