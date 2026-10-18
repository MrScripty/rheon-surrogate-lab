"""Retain actual work and restored accuracy for the matched-example screen."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, write


def report(config_path, projects, cpu, controller_path, qualification, output):
    if output.exists():
        raise ValueError('Fresh immutable report directory required')
    config = read(config_path)
    arm = config['arms'][0]
    identifier = arm['model_id']
    controller, execution = read(controller_path), read(cpu/'execution.json')
    assert controller['status'] == execution['status'] == 'completed'
    assert controller['owned_controller_reaped'] and execution['all_owned_children_reaped']
    assert controller['qualification_sha256'] == sha(qualification)
    proof = read(qualification)
    assert proof['status'] == proof['balanced_cycle']['status'] == 'passed'
    study = projects/(arm['id']+'-study-01')
    receipt = read(study/'receipt.json')
    assert receipt['status'] == 'completed' and all(receipt['artifact_parity'].values())
    trained, restored = receipt['training_receipt'], receipt['restore_receipt']
    assert trained['optimizer_updates'] == config['updates'] == 200
    assert trained['execution']['device'] == 'cuda:0' and trained['execution']['model_dtype'] == 'float32'
    assert not trained['execution']['amp'] and not trained['execution']['tf32']
    assert trained['execution']['parameters_gradients_adam_moments_cuda']
    assert restored['optimizer_updates'] == 0 and restored['execution']['device'] == 'cpu'
    assert not restored['execution']['cuda_initialized']
    intent = read(study/'intent.json')
    assembly_path = Path(intent['assembly'])
    assembly = read(assembly_path)
    project = (ROOT/assembly['project']['path']).parent
    assert sha(assembly_path) == receipt['assembly_sha256']
    assert sha(project/'project.json') == assembly['project']['sha256']
    assert sha(project/'source-witness.json') == proof['source_witness_sha256']
    assert sha(project/'fit_bank.py') == sha(HERE/'fit_bank.py')
    objective, history = read(project/'fine-tuning.json'), read(study/'train/history.json')
    assert history['checkpoints']['checkpoint-000.npz']['inventory'] == objective['initial_inventory']
    control_path = ROOT/config['control_result']
    control = read(control_path)
    previous = control['training'][0]
    stats = trained['fine_tuning_statistics']
    bank = stats['fundamental_bank']
    cases = bank['cases']
    count = len(cases)
    assert count == 8 and bank['optimizer_updates'] == 200 and bank['optimizer_examples'] == config['examples'] == 1600
    assert bank['aggregation'] == 'mean-complete-balanced-cycle'
    assert bank['expected_exposures'] == bank['optimizer_exposures'] == [200]*count
    assert bank['sampling_order_sha256'] == previous['fundamental_bank']['sampling_order_sha256']
    assert bank['cases'] == previous['fundamental_bank']['cases']
    assert bank['amplitude_rms_pa'] == previous['fundamental_bank']['amplitude_rms_pa']
    assert stats['teacher_forwards'] == 0
    assert history['optimizer_work'] == previous['optimizer_work']
    assert history['optimizer_work']['network_forwards'] == config['examples']*arm['steps'] == 6400
    assert history['optimizer_work']['forward_cells'] == 200*arm['steps']*sum(math.prod(c['counts_xyz']) for c in cases) == 7424000
    assert [row['update'] for row in bank['trace']] == list(range(10, 201, 10))
    assert all(row['optimizer_updates'] == row['exposures_per_case'] == row['update'] and row['examples'] == 8*row['update'] for row in bank['trace'])
    for row in history['trace']:
        rate = config['learning_rate']
        for change in config['schedule']:
            if change['update'] <= row['update']:
                rate = change['learning_rate']
        assert row['learning_rate'] == rate
    snapshots = {i:read(cpu/i/'snapshots.json') for i in ('C53', identifier)}
    fitted = {i:read(cpu/(i+'-fitted.json')) for i in ('C53', identifier)}
    diagnostics = {'C53':read(ROOT/'docs/plans/pressure-low-mode-tail-v1/C53-diagnosis.json'),
        identifier:read(cpu/(identifier+'-diagnosis.json'))}
    for group in (snapshots, fitted, diagnostics):
        for key, row in group.items():
            assert row['status'] == 'passed' and not row['cuda_initialized']
            assert row['weights_sha256'] == (receipt['weights_sha256'] if key == identifier else objective['initial_weights_sha256'])
    for row in fitted.values():
        assert row['cases'] == cases and math.isclose(row['amplitude_rms_pa'], bank['amplitude_rms_pa'], rel_tol=1e-12)
    key = lambda row:(row['trajectory_id'], row['native_frame'], row['passes'])
    historical = {key(row):row for row in control['snapshots']['C53']['rows']}
    assert len(historical) == 512
    assert all(row[field] == historical[key(row)][field] for row in snapshots['C53']['rows']
        for field in ('relative_residual', 'same_request_pressure_rms_pa', 'same_request_mean_free_pressure_relative_l2'))
    custody = []
    for phase in ('training', 'restore'):
        terminal = read(study/('terminal-'+receipt[phase+'_run_id']+'.json'))
        assert terminal['state'] == 'completed' and terminal['custody_phase'] == 'drained'
        custody.append(dict(phase=phase, run_id=terminal['id'], state=terminal['state'], custody_phase=terminal['custody_phase']))
    values = lambda model:{(row['case'], row['passes']):row['relative_equation_residual'] for row in fitted[model]['measurements']}
    baseline, candidate = values('C53'), values(identifier)
    assert set(candidate) == set(baseline) == {(i,n) for i in range(count) for n in (1,2,4,8)}
    checks = dict(four_pass_all_fitted=all(candidate[i,4] <= 1e-6 for i in range(count)),
        eight_pass_all_fitted=all(candidate[i,8] <= 1e-6 for i in range(count)),
        one_pass_no_case_worse=all(candidate[i,1] <= baseline[i,1] for i in range(count)),
        two_pass_no_case_worse=all(candidate[i,2] <= baseline[i,2] for i in range(count)))
    decision = dict(status='capacity_screen_completed', candidate=identifier, baseline_retained='C53',
        capacity_gate_passed=all(checks.values()), capacity_checks=checks, model_promoted=False,
        original_untouched_holdouts_consumed=0, next_action=(
            'Propose a separate 24-case aggregate screen and real-frame retention; retain C53.' if all(checks.values()) else
            'Stop this screen; retain C53 and inspect aggregate learning and shared-gradient scale before another training proposal.'),
        interpretation='One-seed matched-example aggregation screen; 200 versus 1600 Adam updates changes moment dynamics. No causal isolation, equal-update, deployment or solver-speed claim.')
    training = dict(id=identifier, arm=arm, source=str((study/'receipt.json').relative_to(ROOT)), sha256=sha(study/'receipt.json'),
        project_id=receipt['project_id'], training_run_id=receipt['training_run_id'], restore_run_id=receipt['restore_run_id'],
        weights_sha256=receipt['weights_sha256'], initialization_sha256=objective['initial_weights_sha256'],
        updates=trained['optimizer_updates'], examples=bank['optimizer_examples'],
        trainer_elapsed_s=trained['elapsed_s'], training_loop_s=history['training_finished_unix']-history['training_started_unix'],
        managed_workflow_elapsed_s=receipt['elapsed_s'], peak_cuda_allocated_bytes=trained['execution']['peak_allocated_bytes'],
        peak_host_rss_bytes=trained['runtime']['peak_rss_bytes'], execution=trained['execution'], runtime=trained['runtime'],
        restore_runtime=restored['runtime'], optimizer_work=history['optimizer_work'], fundamental_bank=bank,
        managed_custody=custody, artifact_parity=receipt['artifact_parity'])
    sources = [config_path, assembly_path, project/'project.json', project/'config.json', project/'fine-tuning.json',
        project/'source-witness.json', controller_path, qualification, cpu/'execution.json', cpu/'material.json',
        cpu/'C53-fitted.json', cpu/(identifier+'-fitted.json'), cpu/'C53/snapshots.json',
        cpu/identifier/'snapshots.json', cpu/(identifier+'-diagnosis.json'), study/'receipt.json',
        study/'readback-profile.json', control_path, *sorted(HERE.glob('*.py'))]
    decision['capacity_evidence'] = {str(path.relative_to(ROOT)):sha(path) for path in
        [cpu/'C53-fitted.json', cpu/(identifier+'-fitted.json'), study/'receipt.json', qualification]}
    result = dict(schema='rheon-balanced-shapes-results-v1', status='completed', training=[training],
        snapshots=snapshots, fitted=fitted, diagnostics=diagnostics, decision=decision, cpu_execution=execution,
        control=dict(id=previous['id'], updates=previous['updates'], training=previous,
            fitted=control['fitted']['C66'], real_summary=control['snapshots']['C66']['summary'],
            manufactured_summary=control['diagnostics']['C66']['summary'],
            source=str(control_path.relative_to(ROOT)), sha256=sha(control_path)),
        historical_baseline_quality_exact_rows=512, readback=read(study/'readback-profile.json'),
        sources={str(path.relative_to(ROOT)):sha(path) for path in sources})
    output.mkdir(parents=True)
    (output/'results.json').write_text(json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n')
    write(output/'decision.json', decision)
    print(identifier, checks, 'gate', all(checks.values()), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'projects', 'cpu', 'controller', 'qualification', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    report(args.config.resolve(), args.projects.resolve(), args.cpu.resolve(), args.controller.resolve(),
        args.qualification.resolve(), args.output.resolve())
