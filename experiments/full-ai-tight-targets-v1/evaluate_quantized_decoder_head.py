"""Consumed-scene precision controls after exact learned checkpoint restoration."""
import argparse
from pathlib import Path
from profile_frozen_components import read, write
import evaluate_recurrent_sampler as evaluation
import evaluate_sampler_quantization as quantization
from decoder_modal_execution import load_model as learned_model
from diagnose_sampler_pressure_oracle import construct as offline_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reg = read(args.protocol)
    if reg['pressure_control'] not in ('managed_learned', 'OFFLINE_numerical_oracle') or reg['quantization_step'] not in (2.**-30, 2.**-40):
        raise ValueError('Only the two registered uniform head precisions and pressure roles allowed')
    args.mode = 'rollout'
    quantization.restored_model = learned_model if reg['pressure_control'] == 'managed_learned' else offline_model
    evaluation.load_model = lambda pair, library_path: quantization.construct(pair, library_path, reg['quantization_step'])
    evaluation.run(args)
    result = read(args.output / 'results.json')
    result.update(status='completed_consumed_scene_head_precision_diagnostic',
                  pressure_control=reg['pressure_control'], quantization_step=reg['quantization_step'],
                  raw_managed_restore_unchanged=True, optimizer_updates=0,
                  numerical_pressure_oracle_used=reg['pressure_control'] == 'OFFLINE_numerical_oracle',
                  candidate_frontier_point=False, independent_target_pass=False,
                  no_new_learning_or_independent_confirmation=True, latency_claim=False)
    write(args.output / 'results.json', result)
