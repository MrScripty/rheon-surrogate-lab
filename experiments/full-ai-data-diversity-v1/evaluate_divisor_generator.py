"""Complete-state evaluation of the frozen callable learned divisor generator."""
import argparse
import collections
from pathlib import Path
from support import ROOT, read, write, digest
from divisor_generator import load_model
import evaluate_recurrent_sampler as evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.mode = 'rollout'
    reg = read(args.protocol)
    if reg['fresh_holdout']:
        selection = ROOT/reg['frozen_selection']
        bank = read(ROOT/reg['bank'])
        if digest(selection) != reg['frozen_selection_sha256'] or bank['frozen_selection_sha256'] != digest(selection):
            raise ValueError('Actual selection must precede this independent reference capture')
    previous = evaluation.load_model
    evaluation.load_model = load_model
    try:
        evaluation.run(args)
    finally:
        evaluation.load_model = previous
    result = read(args.output/'results.json')
    if len(result['rows']) != reg['expected_complete_states']:
        raise ValueError('Every registered complete physical output must be scored')
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Frozen source changed during evaluation: '+path)
    grouped = collections.defaultdict(list)
    for row in result['rows']:
        grouped[(row['horizons'], row['cells_per_metre'], tuple(row['counts_xyz']))].append(row)
    result['legacy_size_summaries'] = result['summaries']
    result['summaries'] = [dict(horizons=k, cells_per_metre=density, counts_xyz=list(grid),
        physical_outputs=len(rows), scenes=len({r['identifier'] for r in rows}),
        failed_outputs=sum(not r['metrics']['all_accuracy_criteria_pass'] for r in rows),
        worst={field: max(r['metrics']['relative'][field] for r in rows)
               for field in ('equation', 'pressure', 'velocity', 'tracer')},
        all_fields_pass=all(r['metrics']['all_accuracy_criteria_pass'] for r in rows))
        for (k, density, grid), rows in sorted(grouped.items())]
    accuracy = all(row['metrics']['all_accuracy_criteria_pass'] for row in result['rows'])
    result.update(status='completed_callable_learned_divisor_evaluation',
        independent_confirmation=reg['fresh_holdout'], independent_accuracy_pass=reg['fresh_holdout'] and accuracy,
        all_accuracy_pass=accuracy, numerical_solver_in_candidate=False,
        analytical_physical_eigenvalues_in_candidate=False, new_parameter_fitting_executed=False,
        full_generator_target_pass=False, latency_claim=False,
        report_grouping='Every K, cells/metre and exact grid dimensions separately')
    write(args.output/'results.json', result)
    print('Callable divisor states', len(result['rows']), 'all accuracy pass', accuracy,
          'independent', reg['fresh_holdout'], flush=True)
