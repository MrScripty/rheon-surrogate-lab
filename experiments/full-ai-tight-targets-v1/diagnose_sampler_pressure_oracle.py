"""Explicitly offline causal pressure-floor control; excluded from AI candidates."""
import argparse
from pathlib import Path
import torch
import evaluate_recurrent_sampler as evaluation
from restored_recurrent_sampler import load_model as restored_model
from profile_frozen_components import read,write
from numerics import correction,fine_action


def construct(pair,library_path):
    model=restored_model(pair,library_path)
    stats=dict(role='OFFLINE numerical pressure-floor control, never a full-AI candidate',calls=0,maximum_self_equation_residual=0.,
        future_reference_inputs=False,analytic_inverse_used_only_in_offline_control=True)
    def reference(q):
        one=torch.ones((1,),dtype=torch.float64)
        pressure=correction(q,one,one,max_mode=None)
        residual=float((fine_action(pressure,one,one)-q).norm()/q.norm().clamp_min(1e-30))
        stats['calls']+=1;stats['maximum_self_equation_residual']=max(stats['maximum_self_equation_residual'],residual)
        if not torch.isfinite(pressure).all()or residual>1e-10:raise ValueError('Offline pressure reference self-equation qualification failed')
        return pressure
    model.pressure_core.forward=reference
    model.asset_identity['offline_pressure_oracle']=stats
    return model


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.mode='rollout'
    evaluation.load_model=construct;evaluation.run(args)
    result=read(args.output/'results.json');result.update(status='completed_OFFLINE_sampler_pressure_floor_control',
        full_ai_candidate=False,candidate_frontier_point=False,numerical_pressure_oracle_used=True,
        no_future_reference_state_used=True,full_generator_target_pass=False,latency_claim=False)
    write(args.output/'results.json',result)
