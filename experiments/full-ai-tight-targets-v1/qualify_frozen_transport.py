"""Bitwise address-path checks against the already-qualified compiled sampler."""
import argparse
import os
from pathlib import Path
from profile_frozen_components import ROOT,read,write,digest
from recurrent_sampler_constructor import construct
from restored_recurrent_sampler import load_model
from recurrent_sampler_integer_v2 import library as baseline_library
from recurrent_sampler_frozen_v7 import library as candidate_library
from frozen_sampler_core_support import prepare
from recurrent_sampler_native import dimensions
import recurrent_sampler_native as native
from diagnose_causal_composition import inventory
import torch


def run(args):
    if args.output.exists():raise ValueError('Fresh sampler address-path qualification required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Sampler qualification source changed: '+path)
    team=prepare(reg);torch.manual_seed(reg['seed'])
    rows=[];models=[('neutral',construct(reg['library_path'])),('seeded',construct(reg['library_path'],'seeded',reg['seed'])),
                    ('trained',load_model(reg['pair'],reg['library_path']))]
    try:
        with torch.inference_mode():
            for initializer,model in models:
                before=inventory(model)
                variants=[(initializer,model.mid_gate,model.mix)]
                if initializer=='trained':variants.append(('extreme_execution_control',torch.linspace(-4,4,12,dtype=torch.float64).reshape(4,3),torch.linspace(4,-4,36,dtype=torch.float64).reshape(4,3,3)))
                for kind,gate,mix in variants:
                    for counts in reg['counts_xyz']:
                        for distribution in reg['distributions']:
                            fields=[]
                            for shape in dimensions(counts):
                                if distribution=='zero':value=torch.zeros(shape,dtype=torch.float32)
                                elif distribution=='negative_zero':value=-torch.zeros(shape,dtype=torch.float32)
                                elif distribution=='subnormal':value=torch.randn(shape).sign()*torch.finfo(torch.float32).tiny*(2.**-20)
                                elif distribution=='constant':value=torch.full(shape,.125,dtype=torch.float32)
                                else:
                                    amplitude={'small':1e-5,'moderate':.02,'large':2e5}[distribution]
                                    value=(torch.randn(shape,dtype=torch.float32)*amplitude).clamp(-1e6,1e6)
                                fields.append(value)
                            for alpha in reg['dt_over_h']:
                                native.library=baseline_library
                                expected=native.transport(fields,counts,model.coefficients['point'],gate,mix,alpha)
                                native.library=candidate_library
                                actual=native.transport(fields,counts,model.coefficients['point'],gate,mix,alpha)
                                if any(a.numpy().tobytes()!=b.numpy().tobytes()for a,b in zip(actual,expected)):
                                    raise ValueError('Address optimization changed carrier bytes: '+str((kind,counts,distribution,alpha)))
                                rows.append(dict(initializer=kind,counts_xyz=counts,distribution=distribution,dt_over_h=alpha,all_four_carriers_bitwise_equal=True))
                if inventory(model)!=before:raise ValueError('Address qualification changed learned tensors')
        if len(rows)!=reg['expected_cases']or torch.cuda.is_initialized():raise ValueError('Qualification population/device changed')
        write(args.output,dict(status='passed_bitwise_sampler_address_qualification',rows=rows,optimizer_updates=0,
            all_learned_tensors_unchanged=True,cuda_initialized=False,source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol),
            execution_controls_are_not_trained_results=True,component_only=True,cores=reg['cores'],execution_team=team))
    finally:
        native.library=baseline_library
        for _,model in models:model.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
