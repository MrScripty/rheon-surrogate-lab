"""Full learned-generator regression or frozen independent accuracy confirmation."""
import argparse
from pathlib import Path
from support import ROOT, read, write, digest
from legacy_table_forcing import bind_for_diagnostic
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
    qualification = read(ROOT/reg['table_force_qualification'])
    if qualification['status'] != 'passed_known_force_table_bitwise_parity':
        raise ValueError('Exact known-input table construction must qualify first')
    if reg['pressure_control'] != 'managed_learned':
        raise ValueError('No numerical pressure oracle permitted in this learned generator')
    restored.library = library
    previous = evaluation.load_model
    evaluation.load_model = lambda pair, lib: LearnedPressureHistory(load_base(pair, lib), mode='reuse')
    try:
        with bind_for_diagnostic():
            evaluation.run(args)
    finally:
        evaluation.load_model = previous
    result = read(args.output/'results.json')
    if len(result['rows']) != reg['expected_complete_states']:
        raise ValueError('All registered physical outputs must be scored')
    parity = None
    if reg.get('output_identity_control'):
        control = read(ROOT/reg['output_identity_control'])
        parity = len(control['rows']) == len(result['rows']) and all(
            all(a[key] == b[key] for key in ('identifier', 'horizons', 'start', 'step', 'lead', 'output_identity'))
            for a, b in zip(result['rows'], control['rows']))
        if not parity:
            raise ValueError('Complete model output bytes changed under scalar-table reuse')
    accuracy = all(row['metrics']['all_accuracy_criteria_pass'] for row in result['rows'])
    result.update(status='completed_table_force_learned_generator_evaluation',
        known_force_role='Known external scalars reused without changing their arithmetic',
        table_force_qualification_sha256=digest(ROOT/reg['table_force_qualification']),
        complete_output_control_bitwise_equal=parity,
        numerical_solver_in_candidate=False, new_parameter_fitting_executed=False,
        independent_confirmation=reg['fresh_holdout'],
        independent_accuracy_pass=reg['fresh_holdout'] and accuracy,
        full_generator_target_pass=False, latency_claim=False)
    write(args.output/'results.json', result)
    print('Complete states:', len(result['rows']), 'all accuracy pass:', accuracy,
          'output parity:', parity, 'independent:', reg['fresh_holdout'], flush=True)
