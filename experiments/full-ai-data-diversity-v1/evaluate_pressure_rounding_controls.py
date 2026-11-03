"""Consumed-scene complete-history precision probes; numerical oracle is OFFLINE only."""
import argparse
from pathlib import Path
from support import ROOT, read, write, digest
from legacy_table_forcing import bind_for_diagnostic
from learned_pressure_history import LearnedPressureHistory
from learned_divisor_pressure import LearnedDivisorPressure
from mixed_modal_sampler import load_model as load_base
from recurrent_sampler_contiguous_v3 import library
import restored_recurrent_sampler as restored
import evaluate_recurrent_sampler as evaluation
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.mode = 'rollout'
    reg = read(args.protocol)
    role = reg['pressure_execution']
    if role not in ('learned_divisor', 'OFFLINE_numerical_inverse') or reg['fresh_holdout']:
        raise ValueError('Explicit consumed-scene precision diagnostic role required')
    if read(ROOT/reg['table_force_qualification'])['status'] != 'passed_known_force_table_bitwise_parity':
        raise ValueError('Qualified known external inputs required')
    restored.library = library
    previous = evaluation.load_model

    def load(pair, lib):
        base = load_base(pair, lib)
        override = None
        if role == 'learned_divisor':
            source = base.pressure_core
            source.close()
            base.pressure_core = LearnedDivisorPressure(source)
            base.asset_identity['pressure_execution'] = 'Tensor basis, division by reciprocals of unchanged learned gains'
        else:
            from numerics import correction
            one = torch.ones(1, dtype=torch.float64)
            override = lambda q: correction(q, one, one, max_mode=None)
        return LearnedPressureHistory(base, mode='reuse', pressure_map=override)

    evaluation.load_model = load
    try:
        with bind_for_diagnostic():
            evaluation.run(args)
    finally:
        evaluation.load_model = previous
    result = read(args.output/'results.json')
    if len(result['rows']) != reg['expected_complete_states']:
        raise ValueError('Every registered actual state must be scored')
    accuracy = all(r['metrics']['all_accuracy_criteria_pass'] for r in result['rows'])
    result.update(status='completed_consumed_pressure_rounding_control',
        pressure_execution=role, numerical_oracle_used=role.startswith('OFFLINE_'),
        numerical_oracle_allowed_in_candidate=False, independent_confirmation=False,
        optimizer_updates=0, new_parameter_fitting_executed=False, latency_claim=False,
        full_generator_target_pass=False)
    write(args.output/'results.json', result)
    print('[DEBUG-rounding]', role, 'actual states', len(result['rows']), 'all accuracy pass', accuracy, flush=True)
