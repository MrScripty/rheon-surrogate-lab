"""Score all available named pressure checkpoints with one frozen equation.

Only CPU tensor scoring of already generated CUDA predictions occurs here.
No neural module, training, GPU work or new reference generation is executed.
The stencil body is checked against the captured C107 finite-volume operator.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import pressure_cases


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'runs/model-portfolio-migration-v1'
DOCS = ROOT / 'docs/plans/model-portfolio-migration-v1'
OPERATOR = ROOT / 'runs/pressure-methods-v4/v4-student50-control-project-01/portable_model.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equation(pressure, cell_size, rho):
    spacing=cell_size[:,None].expand(-1,3);result=torch.zeros_like(pressure)
    for physical_axis,tensor_axis in enumerate((4,3,2)):
        other=[axis for axis in range(3) if axis!=physical_axis]
        area=spacing[:,other[0]]*spacing[:,other[1]]
        weight=(area*(1.0/(rho*spacing[:,physical_axis]))).reshape(-1,1,1,1,1)
        difference=pressure.narrow(tensor_axis,1,pressure.shape[tensor_axis]-1)-pressure.narrow(tensor_axis,0,pressure.shape[tensor_axis]-1)
        lower=[0]*6;upper=[0]*6;lower[2*physical_axis]=1;upper[2*physical_axis+1]=1
        result=result+weight*(F.pad(difference,lower)-F.pad(difference,upper))
    return result


def operator_check():
    captured = next(n for n in ast.parse(OPERATOR.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'full_operator')
    start = next(i for i, n in enumerate(captured.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'spacing' for t in n.targets))
    expected = ast.Module(body=captured.body[start:], type_ignores=[])
    actual = ast.Module(body=ast.parse(inspect.getsource(equation)).body[0].body, type_ignores=[])
    if ast.dump(expected, include_attributes=False) != ast.dump(actual, include_attributes=False):
        raise ValueError('Common equation operations differ from frozen source')


def bindings(row):
    return {k: row[k] for k in ('trajectory_id', 'parent', 'native_frame', 'counts_xyz',
                                'cell_size_m', 'input_sha256', 'reference_pressure_sha256', 'reference_rhs_sha256')}


def worst(rows):
    fields = ('common_equation_relative_l2', 'pressure_mean_free_relative_l2',
              'pressure_absolute_rms', 'pressure_absolute_max')
    return {name: max(r['metrics'][name] for r in rows) for name in fields}


def run(protocol, report, details, markdown):
    if any(p.exists() for p in (report, details, markdown)):
        raise ValueError('Fresh common-score output paths required')
    plan = json.loads(protocol.read_text())
    for path, digest in plan['source_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Scoring protocol input changed: ' + path)
    operator_check()
    torch.set_num_threads(1)
    cases = pressure_cases.load(torch, np)
    initial = json.loads((DOCS / 'reports/pressure-GPU-portfolio-01.json').read_text())
    paths = {name: ROOT / row['result']['path'] for name, row in initial['models'].items()}
    for row in initial['models'].values():
        if sha(ROOT / row['result']['path']) != row['result']['sha256']:
            raise ValueError('Initial accepted pressure result changed')
    repeats = []
    for name in ('pressure-portfolio-GPU-qualification-01', 'pressure-portfolio-GPU-qualification-02'):
        batch = json.loads((RUNS / name / 'batch.json').read_text())
        if batch['status'] != 'completed_qualification_attempts_with_explicit_gaps':
            raise ValueError('Terminal source-verified qualification batch required')
        for row in batch['qualified']:
            path = ROOT / row['result']
            if sha(path) != row['sha256']:
                raise ValueError('Accepted qualification result changed')
            identifier = row['identifier']
            if identifier in paths:
                prior = json.loads(paths[identifier].read_text())
                current = json.loads(path.read_text())
                if prior['rows'] != current['rows']:
                    raise ValueError('Shared decoder changes previously qualified outputs: ' + identifier)
                repeats.append(dict(identifier=identifier, original=str(paths[identifier].relative_to(ROOT)),
                                    repeated=str(path.relative_to(ROOT)), all_rows_exact=True))
            else:
                paths[identifier] = path
    material = json.loads((ROOT / plan['material']).read_text())
    expected_models = {e['id']: e for e in material['models']}
    if set(paths) != set(expected_models) or len(paths) != 127:
        raise ValueError('Available named checkpoint coverage incomplete')
    base_rows = json.loads(paths['C107'].read_text())['rows']
    population = [bindings(r) for r in base_rows]
    records, models, max_operator_delta = {}, {}, 0.
    for identifier in sorted(paths, key=lambda n: int(n[1:])):
        path = paths[identifier]
        result = json.loads(path.read_text())
        if (result['status'] != 'completed_captured_pressure_GPU_accuracy'
                or result['weights_sha256'] != expected_models[identifier]['weights_sha256']
                or result['neural_module_forward_calls']['cpu']
                or not result['learned_parameters_unchanged'] or not result['architecture_unchanged']
                or [bindings(r) for r in result['rows']] != population):
            raise ValueError('Unqualified or unpaired pressure result: ' + identifier)
        archive = path.parent / result['output_archive']['path']
        if sha(archive) != result['output_archive']['sha256']:
            raise ValueError('Pressure prediction archive changed')
        rows, density, grid = [], defaultdict(list), defaultdict(list)
        with np.load(archive, allow_pickle=False) as arrays:
            if set(arrays.files) != {f'case{i:03d}_pressure' for i in range(128)}:
                raise ValueError('Complete pressure output population required')
            for index, ((identity, sample), prior) in enumerate(zip(cases, result['rows'], strict=True)):
                value = arrays[f'case{index:03d}_pressure']
                if (value.dtype.str != '<f8' or tuple(value.shape) != tuple(sample['base'].shape)
                        or hashlib.sha256(value.tobytes()).hexdigest() != prior['output_sha256']):
                    raise ValueError('Saved pressure field identity differs')
                pressure = torch.from_numpy(value)
                residual = equation(pressure, sample['cell_size_m'], sample['rho']) - sample['rhs']
                difference = pressure - sample['reference'];difference -= difference.mean()
                reference = sample['reference'] - sample['reference'].mean()
                rhs_norm = float(torch.linalg.vector_norm(sample['rhs']))
                p_norm = float(torch.linalg.vector_norm(reference))
                metrics = dict(common_equation_relative_l2=float(torch.linalg.vector_norm(residual)) / max(rhs_norm, 1e-30),
                               pressure_mean_free_relative_l2=float(torch.linalg.vector_norm(difference)) / max(p_norm, 1e-30),
                               pressure_absolute_rms=float(difference.square().mean().sqrt()),
                               pressure_absolute_max=float(difference.abs().max()))
                if any(metrics[k] != prior['metrics'][k] for k in metrics if k != 'common_equation_relative_l2'):
                    raise ValueError('Independent pressure-field scoring differs: ' + identifier)
                delta = abs(metrics['common_equation_relative_l2'] - prior['metrics']['equation_relative_l2'])
                max_operator_delta = max(max_operator_delta, delta)
                row = dict(**identity, metrics=metrics, captured_equation_relative_l2=prior['metrics']['equation_relative_l2'],
                           equation_operator_metric_delta=delta, reference_rhs_l2=rhs_norm,
                           reference_mean_free_pressure_l2=p_norm, RHS_floor_active=rhs_norm < 1e-30,
                           pressure_floor_active=p_norm < 1e-30, output_sha256=prior['output_sha256'])
                rows.append(row);density[1. / identity['cell_size_m']].append(row)
                grid[(1. / identity['cell_size_m'], tuple(identity['counts_xyz']))].append(row)
        records[identifier] = dict(rows=rows, by_density_and_grid=[dict(cells_per_metre=k[0], counts_xyz=list(k[1]),
                                  cases=len(v), worst=worst(v)) for k, v in sorted(grid.items())])
        model = dict(name=expected_models[identifier]['name'], weights_sha256=result['weights_sha256'],
                     cases=128, worst=worst(rows),
                     worst_case={key: {k: v for k, v in max(rows, key=lambda r: r['metrics'][key]).items()
                                       if k in ('trajectory_id', 'native_frame', 'counts_xyz', 'cell_size_m')}
                                 for key in ('common_equation_relative_l2', 'pressure_mean_free_relative_l2')},
                     by_cells_per_metre={str(k): dict(cases=len(v), worst=worst(v)) for k, v in density.items()},
                     execution_variant=result.get('execution_variant', 'captured_CUDA_eager'),
                     result=dict(path=str(path.relative_to(ROOT)), sha256=sha(path)),
                     CNN_operation_AST_preserved=result.get('CNN_operation_AST_preserved'),
                     CPU_neural_forward=False)
        models[identifier] = model
        if len(models) % 20 == 0:
            print(len(models), 'pressure checkpoints independently scored', flush=True)
    if torch.cuda.is_initialized():
        raise ValueError('This scoring process must not execute neural inference or GPU operations')
    detail = dict(schema='rheon.named-pressure-common-score-details.v1', records=records,
                  protocol=dict(path=str(protocol.relative_to(ROOT)), sha256=sha(protocol)))
    details.write_text(json.dumps(detail, indent=2, allow_nan=False) + '\n')
    summary = dict(schema='rheon.named-pressure-common-score.v1', created_unix=time.time(),
                   status='all127_available_named_checkpoints_scored_on_same128_native_requests',
                   models=models, unique_requests_per_checkpoint=128, pressure_requests=127 * 128,
                   unavailable=[r for r in material['dispositions'] if r['status'] == 'original_checkpoint_unavailable'],
                   decoder_controls=repeats, independent_pressure_scoring_exact=True,
                   maximum_captured_versus_common_equation_metric_difference=max_operator_delta,
                   common_operator_AST_matches_frozen_source=True,
                   protocol=detail['protocol'], source_sha256=sha(Path(__file__)),
                   details=dict(path=str(details.relative_to(ROOT)), sha256=sha(details)),
                   CPU_neural_forward=False, new_parameter_fits=0, latency_claim=False,
                   independent_confirmation=False, complete_portfolio_benchmark=False)
    report.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    lines = ['# Named pressure checkpoints: common GPU accuracy', '',
             'All 127 available named checkpoints completed the same 128 consumed native requests.',
             'Every saved pressure field was rehashed and independently rescored. C01/C02 remain unavailable.',
             'No new fits, CPU neural forwards, independent-validation or fresh latency claims.', '',
             '| Checkpoint | Name | Worst common equation relative L2 | Worst mean-free pressure relative L2 |',
             '| --- | --- | --- | --- |']
    for identifier, model in models.items():
        lines.append(f"| {identifier} | {model['name']} | {model['worst']['common_equation_relative_l2']:.8g} | {model['worst']['pressure_mean_free_relative_l2']:.8g} |")
    lines.extend(['', 'Results are diagnostics. Completing these pressure requests does not establish',
                  'full simulation accuracy, own-history stability, generalization or speed.',
                  'Density and domain dimensions remain separate in the retained per-case results.',
                  'C03/C04 use GPU CNN execution with their captured CPU preprocessing/reconstruction;',
                  'the CNN operation bodies match the original sources. Other checkpoints use their',
                  'captured CUDA implementations. No trained parameter changed.', '',
                  '[Evidence and density summaries](reports/pressure-common-score-01.json).',
                  'The full applicable portfolio benchmark and native project migration remain open.'])
    markdown.write_text('\n'.join(lines) + '\n')
    print(summary['status'], summary['pressure_requests'], 'requests; maximum operator metric delta', max_operator_delta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'report', 'details', 'markdown'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.report.resolve(), args.details.resolve(), args.markdown.resolve())
