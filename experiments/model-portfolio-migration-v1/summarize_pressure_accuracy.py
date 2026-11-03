"""Compare retained CUDA pressure results only after exact case/input pairing."""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'runs/model-portfolio-migration-v1'
METRICS = ('equation_relative_l2', 'pressure_mean_free_relative_l2',
           'pressure_absolute_rms', 'pressure_absolute_max')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return row['trajectory_id'], row['parent'], row['native_frame']


def case_binding(row):
    return {k: row[k] for k in ('trajectory_id', 'parent', 'native_frame', 'counts_xyz',
                                'cell_size_m', 'input_sha256', 'reference_pressure_sha256',
                                'reference_rhs_sha256')}


def summarize(rows):
    return dict(cases=len(rows), worst={name: max(r['metrics'][name] for r in rows) for name in METRICS},
                worst_case={name: {k: v for k, v in max(rows, key=lambda r: r['metrics'][name]).items()
                                  if k not in ('input_sha256', 'metrics')}
                            for name in METRICS})


def run(report, markdown, details):
    if report.exists() or markdown.exists() or details.exists():
        raise ValueError('Fresh comparison output paths required')
    batch_path = RUNS / 'pressure-material-GPU-accuracy-01/batch.json'
    batch = json.loads(batch_path.read_text())
    if batch['status'] != 'completed_original21_pressure_GPU_accuracy' or len(batch['completed']) != 21:
        raise ValueError('Complete qualified original21 batch required')
    paths = {r['identifier']: ROOT / r['result'] for r in batch['completed']}
    if any(sha(ROOT / r['result']) != r['sha256'] for r in batch['completed']):
        raise ValueError('Saved batch result changed')
    paths.update({name: RUNS / ('pressure-GPU-' + name + '-accuracy-01/results.json')
                  for name in ('C106', 'C107')})
    baseline = json.loads(paths['C107'].read_text())
    bindings = {key(r): case_binding(r) for r in baseline['rows']}
    if len(bindings) != 128:
        raise ValueError('Exact complete case identity required')
    models, operator_definitions, all_rows = {}, set(), {}
    for name, path in paths.items():
        result = json.loads(path.read_text())
        if sha(path.parent / result['output_archive']['path']) != result['output_archive']['sha256']:
            raise ValueError('Saved pressure output archive changed: ' + name)
        if (result['status'] != 'completed_captured_pressure_GPU_accuracy'
                or result['cases'] != 128 or result['CPU_model_forward']
                or result['neural_module_forward_calls']['cpu']
                or not result['learned_parameters_unchanged'] or not result['architecture_unchanged']):
            raise ValueError('Incomplete or changed pressure model: ' + name)
        rows = result['rows']
        if {key(r): case_binding(r) for r in rows} != bindings or len(rows) != 128:
            raise ValueError('Unpaired case/input/reference population: ' + name)
        all_rows[name] = {key(r): r for r in rows}
        density_groups, grid_groups = defaultdict(list), defaultdict(list)
        for row in rows:
            density = 1. / row['cell_size_m']
            density_groups[str(density)].append(row)
            grid_groups[(density, tuple(row['counts_xyz']))].append(row)
        operator = ROOT / result['source_entry']['project'] / 'portable_model.py'
        tree = ast.parse(operator.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'full_operator')
        operator_ast = hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest()
        operator_definitions.add(operator_ast)
        models[name] = dict(**summarize(rows), weights_sha256=result['weights_sha256'],
                            result=dict(path=str(path.relative_to(ROOT)), sha256=sha(path)),
                            source_entry=result['source_entry'],
                            numerical_execution='captured_CUDA_eager',
                            precision={k: result[k] for k in ('network_dtype', 'pressure_dtype', 'tf32', 'autocast')},
                            neural_module_forward_calls=result['neural_module_forward_calls'],
                            by_cells_per_metre={k: summarize(v) for k, v in density_groups.items()},
                            by_density_and_grid=[dict(cells_per_metre=k[0], counts_xyz=list(k[1]), **summarize(v))
                                                 for k, v in sorted(grid_groups.items())],
                            portable_operator=dict(path=str(operator.relative_to(ROOT)), sha256=sha(operator),
                                                   function_AST_sha256=operator_ast))
    if len(operator_definitions) != 1:
        raise ValueError('Different captured operator definitions require review before comparison')
    case_wins = {}
    for metric in METRICS[:2]:
        counts = dict(C53_lower=0, C107_lower=0, equal=0)
        for identity in bindings:
            a, b = (all_rows[name][identity]['metrics'][metric] for name in ('C53', 'C107'))
            counts['C53_lower' if a < b else 'C107_lower' if b < a else 'equal'] += 1
        case_wins[metric] = counts
    value = dict(schema='rheon.paired-pressure-GPU-accuracy-portfolio.v1', created_unix=time.time(),
                 status='23_original_checkpoints_paired_on128_pressure_requests',
                 models=models, requests=23 * 128, unique_cases=128,
                 all_case_input_reference_bindings_identical=True,
                 captured_portable_operator_function_AST_identical=True,
                 C53_C107_per_case_comparison=case_wins,
                 source_sha256=sha(Path(__file__)),
                 batch=dict(path=str(batch_path.relative_to(ROOT)), sha256=sha(batch_path)),
                 latency_claim=False, independent_confirmation=False, complete_portfolio_benchmark=False,
                 new_parameter_fits=0, CPU_neural_forward=False,
                 limitations=['Consumed original pressure requests; no independent validation or fresh timing.',
                              'Pressure-only proposals do not constitute complete simulation steps or own-history rollout tests.',
                              'Pressure field error, equation residual and spatial maximum absolute pressure error are separate measurements.',
                              'These 23 checkpoints are a subset; other checkpoints, components and execution variants remain pending.'])
    details.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    compact = {k: v for k, v in value.items() if k != 'models'}
    compact['details'] = dict(path=str(details.relative_to(ROOT)), sha256=sha(details))
    compact['models'] = {}
    for name, model in models.items():
        item = {k: v for k, v in model.items() if k not in ('source_entry', 'by_cells_per_metre', 'by_density_and_grid')}
        item['source_entry'] = {k: model['source_entry'][k] for k in ('id', 'project', 'restore')}
        item['by_cells_per_metre'] = {k: {field: v[field] for field in ('cases', 'worst')}
                                      for k, v in model['by_cells_per_metre'].items()}
        item['by_density_and_grid'] = [{k: v[k] for k in ('cells_per_metre', 'counts_xyz', 'cases', 'worst')}
                                      for v in model['by_density_and_grid']]
        compact['models'][name] = item
    report.write_text(json.dumps(compact, indent=2, allow_nan=False) + '\n')
    lines = ['# Paired pressure accuracy on GPU', '',
             'Fresh CUDA accuracy on the same 128 consumed native requests per model.',
             'All input/reference hashes pair exactly. Weights and captured architectures stayed unchanged.',
             'No fresh timing or independent-validation claim. These are pressure-only proposals.', '',
             '| Model | Worst relative equation residual | Worst mean-free pressure relative L2 |',
             '| --- | --- | --- |']
    for name in sorted(models, key=lambda n: int(n[1:])):
        worst = models[name]['worst']
        lines.append(f"| {name} | {worst['equation_relative_l2']:.8g} | {worst['pressure_mean_free_relative_l2']:.8g} |")
    lines.extend(['', 'C53 has lower worst pressure-field error than C107 on this population, while',
                  'C107 has lower worst equation residual and lower pressure error on 122 of 128',
                  'individual requests. Neither metric establishes superiority',
                  'on all cases, larger domains or complete trajectories. The retained result',
                  'includes every per-case output, denominator and field hash, with separate',
                  'density and grid summaries in the [comparison report](reports/pressure-GPU-portfolio-01.json).',
                  '', 'The full applicable portfolio benchmark and native project migration remain open.'])
    markdown.write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: value[k] for k in ('status', 'requests', 'unique_cases', 'C53_C107_per_case_comparison')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, required=True)
    parser.add_argument('--details', type=Path, required=True)
    args = parser.parse_args()
    run(args.report.resolve(), args.markdown.resolve(), args.details.resolve())
