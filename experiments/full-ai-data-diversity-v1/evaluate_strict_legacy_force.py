"""Score every complete state with qualified strict external inputs and learned dynamics."""
import argparse
from pathlib import Path
from support import ROOT, read, write, digest
from legacy_exact_forcing import bind_for_diagnostic
from learned_pressure_history import LearnedPressureHistory
from mixed_modal_sampler import load_model as load_base
from recurrent_sampler_contiguous_v3 import library
import restored_recurrent_sampler as restored
import evaluate_recurrent_sampler as evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.mode = 'rollout'
    reg = read(args.protocol)
    qualification = read(ROOT/reg['legacy_force_qualification'])
    if qualification['status'] != 'passed_strict_legacy_external_forcing':
        raise ValueError('Native-exact known external inputs must qualify first')
    if reg['fresh_holdout'] or reg['pressure_control'] != 'managed_learned':
        raise ValueError('This is a consumed-scene learned-model regression, without a numerical oracle')
    restored.library = library
    previous_loader = evaluation.load_model
    evaluation.load_model = lambda pair, lib: LearnedPressureHistory(load_base(pair, lib), mode='reuse')
    try:
        with bind_for_diagnostic():
            evaluation.run(args)
    finally:
        evaluation.load_model = previous_loader
    result = read(args.output/'results.json')
    result.update(status='completed_consumed_scene_strict_force_learned_generator_regression',
        known_force_role='Qualified native-scalar construction of known legacy external inputs',
        legacy_force_qualification_sha256=digest(ROOT/reg['legacy_force_qualification']),
        numerical_solver_in_candidate=False, new_parameter_fitting_executed=False,
        independent_confirmation=False, full_generator_target_pass=False, latency_claim=False)
    write(args.output/'results.json', result)
