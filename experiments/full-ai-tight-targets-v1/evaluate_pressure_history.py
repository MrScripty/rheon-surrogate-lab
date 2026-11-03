"""Consumed-scene learned pressure-history prototype and explicit offline control."""
import argparse
from pathlib import Path
import torch
from profile_frozen_components import read, write
from mixed_modal_sampler import load_model as load_base
from learned_pressure_history import LearnedPressureHistory
import restored_recurrent_sampler as restored
from recurrent_sampler_contiguous_v3 import library
import evaluate_recurrent_sampler as evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reg = read(args.protocol)
    if reg['pressure_control'] not in ('managed_learned', 'OFFLINE_numerical_oracle'):
        raise ValueError('Explicit learned or OFFLINE pressure role required')
    restored.library = library
    override = None
    if reg['pressure_control'] == 'OFFLINE_numerical_oracle':
        from numerics import correction
        one = torch.ones(1, dtype=torch.float64)
        override = lambda q: correction(q, one, one, max_mode=None)
    evaluation.load_model = lambda pair, lib: LearnedPressureHistory(load_base(pair, lib), reg['history_mode'], override)
    args.mode = 'rollout'
    evaluation.run(args)
    result = read(args.output / 'results.json')
    result.update(status='completed_consumed_scene_pressure_history_prototype',
                  history_mode=reg['history_mode'], pressure_control=reg['pressure_control'],
                  numerical_pressure_oracle_used=override is not None,
                  prototype_composes_frozen_managed_learned_modules=True,
                  new_parameter_fitting_executed=False, optimizer_updates=0,
                  candidate_frontier_point=False, independent_target_pass=False, latency_claim=False)
    write(args.output / 'results.json', result)
