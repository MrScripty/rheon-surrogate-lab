"""Freeze selection and retain compact, hash-bound results from completed runs."""
import argparse
import math
from pathlib import Path
import statistics

from common import ROOT, read, sha, write


def identity(row):
    return row['trajectory_id'], row['native_frame'], tuple(row['counts_xyz']), row['cell_size_m'], row['passes']


def snapshots(directory):
    results = {}
    expected = None
    for identifier in ['C53', 'C55', 'C57', 'C58', 'C59']:
        path = directory / identifier / 'snapshots.json'
        result = read(path)
        membership = {identity(row) for row in result['rows']}
        if (result['status'] != 'passed' or result['cuda_initialized'] or result['torch_threads'] != 1
                or len(result['rows']) != 512 or len(membership) != 512
                or expected is not None and expected != membership):
            raise ValueError('Exact finite CPU-only matched 128 by four settings required')
        if any(not math.isfinite(row[key]) for row in result['rows'] for key in [
                'relative_residual', 'inference_s', 'same_request_pressure_rms_pa',
                'same_request_mean_free_pressure_relative_l2']):
            raise ValueError('Nonfinite candidate measurement')
        expected = membership
        results[identifier] = dict(summary=result['summary'], elapsed_s=result['elapsed_s'],
            peak_rss_bytes=result['peak_rss_bytes'], weights_sha256=result['weights_sha256'],
            material_sha256=result['material_sha256'], source=str(path.relative_to(ROOT)), sha256=sha(path))
    return results


def select(directory, output):
    if output.exists():
        raise ValueError('Selection already frozen; do not reselect from independent results')
    measured = snapshots(directory)
    def rank(identifier):
        four = next(row for row in measured[identifier]['summary'] if row['passes'] == 4)
        return -four['equation_passes'], four['worst_equation_residual'], four['median_inference_ms'], identifier
    selected = min(['C57', 'C58', 'C59'], key=rank)
    policy = ROOT / 'docs/plans/pressure-four-pass-distillation-v1/selection-policy.json'
    write(output, dict(status='frozen_before_fresh_candidate_evaluation', selected=selected,
        ranking=sorted(['C57', 'C58', 'C59'], key=rank), policy_sha256=sha(policy),
        matched_snapshots=measured, promotion=False))
    print('Frozen diagnostic selection', selected, flush=True)


def training(config, projects):
    records = []
    common = None
    for number, arm in enumerate(config['arms'], 57):
        directory = projects / (arm['id'] + '-study-01')
        path = directory / 'receipt.json'
        result = read(path)
        train, restore = result['training_receipt'], result['restore_receipt']
        history = read(directory / 'train/history.json')
        closure = train['closure']
        matched = {key: closure[key] for key in ['inputs', 'normalization_sha256', 'sample_order_sha256', 'runtime_manifest_sha256']}
        initial = history['checkpoints']['checkpoint-000.npz']['sha256']
        matched['initial_weights_sha256'] = initial
        if (result['status'] != 'completed' or result['optimizer_updates'] != config['updates']
                or not all(result['artifact_parity'].values()) or train['execution']['device'] != 'cuda:0'
                or restore['execution']['device'] != 'cpu' or restore['execution']['cuda_initialized']
                or restore['optimizer_updates'] != 0 or common is not None and common != matched):
            raise ValueError('Matched CUDA training and exact fresh CPU restore required')
        if any(not math.isfinite(t['loss']) or not math.isfinite(t['gradient_norm_before_clip']) for t in history['trace']):
            raise ValueError('Nonfinite retained training trace')
        common = matched
        terminal = read(directory / ('terminal-' + result['training_run_id'] + '.json'))
        restore_terminal = read(directory / ('terminal-' + result['restore_run_id'] + '.json'))
        if any(run['state'] != 'completed' or run['custody_phase'] != 'drained' for run in [terminal, restore_terminal]):
            raise ValueError('Managed descendants must be drained')
        records.append(dict(id='C' + str(number), arm=arm, updates=result['optimizer_updates'],
            training_run_id=result['training_run_id'], restore_run_id=result['restore_run_id'],
            project_id=result['project_id'], study_id=result['study_id'],
            optimizer_and_bank_preparation_s=history['training_finished_unix'] - history['training_started_unix'],
            optimizer_updates_per_s=history['optimizer_updates_per_s'],
            trainer_elapsed_s=train['elapsed_s'], managed_workflow_elapsed_s=result['elapsed_s'],
            restore_elapsed_s=restore['elapsed_s'], execution=train['execution'], runtime=train['runtime'],
            restore_execution=restore['execution'], fine_tuning_statistics=train['fine_tuning_statistics'],
            optimizer_work=history['optimizer_work'], trace=history['trace'],
            trace_scope='Training loss before update; diagnostic physical ratio uses the unchanged default TWO-pass predictor on that update sample, not the four-pass acceptance sweep.',
            weights_sha256=result['weights_sha256'], matched=matched,
            source=str(path.relative_to(ROOT)), sha256=sha(path), artifact_parity=result['artifact_parity'],
            acknowledgement_recovery=read(directory / 'recovery/receipt.json') if (directory / 'recovery/receipt.json').exists() else None,
            managed_operations=[{k: read(p).get(k) for k in ['phase', 'operation', 'status', 'elapsed_s', 'reconciliation']}
                                for p in sorted((directory / 'mutations').glob('*.json'))],
            run_times={name: {k: run[k] for k in ['created_at', 'started_at', 'finished_at']}
                       for name, run in [('training', terminal), ('restore', restore_terminal)]}))
    return records


def episodes(directory):
    execution_path = directory / 'execution.json'
    execution = read(execution_path)
    if execution['status'] != 'completed' or not execution['all_owned_children_reaped'] or len(execution['rows']) != 8:
        raise ValueError('Complete drained eight-scene campaign required')
    rows = []
    for index in range(8):
        path = directory / f'scene-{index:02d}/episode.json'
        row = read(path)
        native_path = directory / f'scene-{index:02d}/native/receipt.json'
        native = read(native_path)
        if not row['native_reaped']:
            raise ValueError('Native child not reaped')
        rows.append(dict(scene_index=index, **row, source=str(path.relative_to(ROOT)), sha256=sha(path),
            native_receipt_sha256=sha(native_path), binary_sha256=native['binary_sha256'],
            native_policy={k: native.get(k) for k in ['native_pressure_refinement', 'native_fallback',
                'execution_policy', 'cpu_threads', 'cuda_initialized', 'learned_proposals_per_pressure_solve']}))
    return dict(execution=execution, execution_sha256=sha(execution_path), episodes=rows)


def collect(args):
    config = read(args.config)
    selection = read(args.selection)
    records = training(config, args.projects)
    measured = snapshots(args.isolated)
    if selection['matched_snapshots'] != measured:
        raise ValueError('Selection measurements changed after freeze')
    selected = selection['selected']
    campaigns = {name: episodes(args.projects / directory) for name, directory in [
        ('native-control', 'fresh-native-01'), ('C53-p8', 'fresh-C53-p8-01'),
        (selected + '-p4', 'fresh-selected-p4-01'), (selected + '-p8', 'fresh-selected-p8-01')]}
    for index in range(8):
        hashes = {campaign['episodes'][index]['scene_sha256'] for campaign in campaigns.values()}
        if len(hashes) != 1:
            raise ValueError('Independent episode scenes changed between policies')
    frozen_material = read(args.isolated / 'material.json')
    if any(row['binary_sha256'] != frozen_material['binary_sha256']
           for campaign in campaigns.values() for row in campaign['episodes']):
        raise ValueError('Native reference implementation changed during qualification')
    config_source = str(args.config.relative_to(ROOT))
    controller_path = args.projects / 'controller-execution.json'
    controller = read(controller_path)
    if controller['status'] not in ['completed', 'has_failed_arms'] or not all(c['reaped'] for c in controller['controllers']):
        raise ValueError('Managed controller completion required')
    if controller['status'] == 'has_failed_arms' and not any(r['acknowledgement_recovery'] for r in records):
        raise ValueError('Failed controller outcome must have retained exact acknowledgement reconciliation')
    model4 = next(s for s in measured[selected]['summary'] if s['passes'] == 4)
    model8 = next(s for s in measured[selected]['summary'] if s['passes'] == 8)
    fresh4 = campaigns[selected + '-p4']['episodes']
    fresh8 = campaigns[selected + '-p8']['episodes']
    baseline_rows = read(args.isolated / 'C53/snapshots.json')['rows']
    baseline_four = {identity(r): r['relative_residual'] for r in baseline_rows if r['passes'] == 4}
    paired = {}
    for identifier in measured:
        rows = [r for r in read(args.isolated / identifier / 'snapshots.json')['rows'] if r['passes'] == 4]
        ratios = [r['relative_residual'] / baseline_four[identity(r)] for r in rows]
        paired[identifier] = dict(cases_better_than_C53=sum(r < 1 for r in ratios),
            cases_worse_than_C53=sum(r > 1 for r in ratios), median_paired_residual_ratio=statistics.median(ratios),
            largest_paired_residual_ratio=max(ratios))
    baseline_diagnosis_path = ROOT / 'docs/plans/pressure-four-pass-distillation-v1/C53-diagnosis.json'
    postfit_path = args.projects / 'postfit-diagnosis-C59.json'
    baseline_diagnosis, postfit = read(baseline_diagnosis_path), read(postfit_path)
    if (postfit['weights_sha256'] != measured['C59']['weights_sha256']
            or baseline_diagnosis['weights_sha256'] != measured['C53']['weights_sha256']
            or any(row['split'] != 'train' for row in postfit['rows'])):
        raise ValueError('Postfit spectral diagnosis must match the selected frozen checkpoint and training-only population')
    training_error_ratio = (sum(r['pressure_error_energy_pa2'] for r in postfit['rows'] if r['passes'] == 4)
                           / sum(r['pressure_error_energy_pa2'] for r in baseline_diagnosis['rows'] if r['passes'] == 4))
    monitor_path = args.projects / 'resource-samples.json'
    monitor = read(monitor_path)
    samples = monitor['samples']
    resource_observation = dict(source=str(monitor_path.relative_to(ROOT)), sha256=sha(monitor_path),
        status=monitor['status'], samples=len(samples), first_utc=samples[0]['utc'], last_utc=samples[-1]['utc'],
        lowest_host_available_bytes=min(s['host_available_bytes'] for s in samples),
        highest_whole_gpu_memory_mib=max(s['gpu_memory_mib'] for s in samples),
        highest_whole_gpu_utilization_percent=max(s['gpu_utilization_percent'] for s in samples),
        scope='Whole GPU and host samples; observation began during the first training arm. This is not full-window per-job utilization or a training-only average.')
    result = dict(schema='rheon-four-pass-results-v1', status='completed', config=read(args.config),
        source_config=config_source, source_config_sha256=sha(args.config), training=records,
        selection=selection, snapshots=measured, campaigns=campaigns,
        paired_four_pass_results=paired,
        postfit_train_diagnosis=dict(model='C59', source=str(postfit_path.relative_to(ROOT)), sha256=sha(postfit_path),
            summary=postfit['summary'], four_pass_total_pressure_error_sum_of_squares_ratio_vs_C53=training_error_ratio,
            scope='Original 32 training snapshots only. Aggregate pressure sum-of-squares weights grid cells; not a fresh-scene selection metric.'),
        controller=dict(source=str(controller_path.relative_to(ROOT)), sha256=sha(controller_path), **controller),
        isolated_execution=read(args.isolated / 'execution.json'),
        resource_observation=resource_observation,
        desired_four_pass_snapshot_target_met=model4['equation_passes'] == 128,
        eight_pass_snapshot_target_retained=model8['equation_passes'] == 128,
        fresh_four_pass_complete=sum(r['status'] == 'completed' for r in fresh4),
        fresh_eight_pass_complete=sum(r['status'] == 'completed' for r in fresh8),
        no_deployment_promotion=True, historical_evidence_modified=False,
        measurement_limits=[
            'One seed and 1500-update warm-start screening; not a from-scratch or equal-GPU-time comparison.',
            'Fixed-pass snapshot timing is serial on one CPU core, recurrence plus final equation check; excludes input construction and IPC.',
            'Added training equation feedback is normalized by the initial correction-request residual. Snapshot and native acceptance residuals are normalized by the full pressure RHS; these denominators must not be conflated.',
            'Independent episodes use concurrent bounded CPU workers; their elapsed time is not isolated inference latency.',
            'Episode peak RSS describes the Python inference owner, excluding its separate native child. GPU allocation is measured by Torch; whole-GPU samples may include desktop work.',
            'Absolute independent pressure drift is a triangle-inequality bounded estimate in the common cell-zero gauge, not a directly recorded absolute norm.',
            'Fresh scenes are eight independent parameter combinations, not proof of arbitrary geometry, fluid or boundary generalization.',
            '512-step requested horizons are 8.533 seconds, with 15 native warm-up steps. Failed pressure checks stop the candidate trajectory.',
            'Float32 neural compute; float64 residual/reconstruction. The separate storage-precision study is not a universal network precision floor.',
            'Default deployment remains two passes. Four/eight passes are explicit experiment policies, not silently promoted defaults.'
        ])
    args.output.mkdir(exist_ok=True)
    write(args.output / 'results.json', result)
    print('Retained results', selected, model4['equation_passes'], result['fresh_four_pass_complete'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    selection = modes.add_parser('select')
    selection.add_argument('--isolated', type=Path, required=True)
    selection.add_argument('--output', type=Path, required=True)
    report = modes.add_parser('collect')
    for name in ['config', 'projects', 'isolated', 'selection', 'output']:
        report.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.mode == 'select':
        select(args.isolated, args.output)
    else:
        collect(args)
