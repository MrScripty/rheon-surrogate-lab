"""Consumed-scene causal controls of known forcing, with explicit oracle separation."""
import argparse
from pathlib import Path
from profile_frozen_components import read, write
import evaluate_recurrent_sampler as evaluation
import recurrent_sampler_model as model_module
from decoder_modal_execution import load_model as learned_model
from diagnose_sampler_pressure_oracle import construct as offline_model
from strict_prescribed_conditioning import forcing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reg = read(args.protocol)
    if reg['pressure_control'] not in ('managed_learned', 'OFFLINE_numerical_oracle'):
        raise ValueError('Explicit pressure-control role required')
    args.mode = 'rollout'
    model_module.forcing = lambda scene, step, horizons: forcing(scene, step, horizons, reg['conditioning_policy'])
    evaluation.load_model = learned_model if reg['pressure_control'] == 'managed_learned' else offline_model
    evaluation.run(args)
    result = read(args.output / 'results.json')
    result.update(status='completed_consumed_scene_conditioning_diagnostic',
                  conditioning_policy=reg['conditioning_policy'], pressure_control=reg['pressure_control'],
                  candidate_frontier_point=False, independent_target_pass=False,
                  numerical_pressure_oracle_used=reg['pressure_control'] == 'OFFLINE_numerical_oracle',
                  fixed_prescribed_conditioning_only_changed=True, optimizer_updates=0,
                  no_new_learning_or_independent_confirmation=True, latency_claim=False)
    write(args.output / 'results.json', result)
