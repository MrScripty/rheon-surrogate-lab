"""Consumed-bank head precision and explicitly offline pressure-floor controls."""
import argparse
from pathlib import Path
from profile_frozen_components import read, write
from refined_pressure_sampler import load_model as load_refined
from diagnose_sampler_pressure_oracle import construct as offline_model
import evaluate_sampler_quantization as quantization
import evaluate_recurrent_sampler as evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reg = read(args.protocol)
    if reg['pressure_control'] not in ('managed_mixed_queries', 'OFFLINE_numerical_oracle') or reg['quantization_step'] not in (2.**-30, 2.**-40):
        raise ValueError('Predeclared head precision and pressure role required')
    quantization.restored_model = (lambda pair, lib: load_refined(pair, lib, reg['pressure_pair'])) if reg['pressure_control'] == 'managed_mixed_queries' else offline_model
    evaluation.load_model = lambda pair, lib: quantization.construct(pair, lib, reg['quantization_step'])
    args.mode = 'rollout'
    evaluation.run(args)
    result = read(args.output / 'results.json')
    result.update(status='completed_consumed_mixed_model_precision_or_floor_diagnostic',
                  pressure_control=reg['pressure_control'], quantization_step=reg['quantization_step'],
                  numerical_pressure_oracle_used=reg['pressure_control'] == 'OFFLINE_numerical_oracle',
                  candidate_frontier_point=False, independent_target_pass=False,
                  no_new_learning_or_independent_confirmation=True, optimizer_updates=0,
                  raw_managed_artifacts_preserved=True, latency_claim=False)
    write(args.output / 'results.json', result)
