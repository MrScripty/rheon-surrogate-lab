"""Retain matched training, CPU qualification and spectral diagnosis evidence."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, write


def report(config_path, projects, cpu, output):
    config = read(config_path)
    policy = read(output/'selection-policy.json')
    if (policy['configuration_sha256'] != sha(config_path) or policy['baseline'] != 'C53'
            or policy['candidates'] != [a['model_id'] for a in config['arms']]
            or policy['relative_equation_target'] != config['target_relative_residual']):
        raise ValueError('Frozen pre-evaluation selection policy differs')
    execution = read(cpu/'execution.json')
    if execution['status'] != 'completed' or not execution['all_owned_children_reaped']:
        raise ValueError('Complete drained CPU measurements required')
    material = read(cpu/'material.json')
    identifiers = config['baselines']+[arm['model_id'] for arm in config['arms']]
    snapshots = {identifier:read(cpu/identifier/'snapshots.json') for identifier in identifiers}
    training, captured_configs, objectives, closures, banks = [], [], [], [], []
    for arm in config['arms']:
        directory = projects/(arm['id']+'-study-01')
        project = projects/(arm['id']+'-project-01')
        receipt = read(directory/'receipt.json')
        trained, restored = receipt['training_receipt'], receipt['restore_receipt']
        history = read(directory/'train/history.json')
        qualification = read(projects/(arm['id']+'-qualification.json'))
        if (receipt['status'] != 'completed' or receipt['optimizer_updates'] != config['updates']
                or not all(receipt['artifact_parity'].values()) or trained['execution']['device'] != 'cuda:0'
                or not trained['execution']['parameters_gradients_adam_moments_cuda']
                or trained['execution']['model_dtype'] != 'float32' or trained['execution']['amp'] or trained['execution']['tf32']
                or restored['optimizer_updates'] != 0 or restored['execution']['device'] != 'cpu'
                or restored['execution']['cuda_initialized']
                or sha(directory/'restore/weights.npz') != receipt['weights_sha256']
                or history['supplemental_feedback']['sampling_order_sha256'] != qualification['hard_bank']['order_sha256']):
            raise ValueError('Matched actual CUDA/CPU/bank custody differs')
        bank = history['supplemental_feedback']
        cpu_amplitude = read(projects/'C53-diagnosis.json')['amplitude_reference_rms_pa']
        if (bank['resident_tensor_bytes'] != qualification['hard_bank']['bytes']
                or bank['grid_exposures'] != qualification['hard_bank']['grid_exposures']
                or not math.isclose(bank['amplitude_rms_pa'], cpu_amplitude, rel_tol=1e-12, abs_tol=0.)):
            raise ValueError('Training bank differs from independently qualified coverage/scales')
        banks.append((bank['bank_description_sha256'], bank['sampling_order_sha256']))
        for phase in ['training', 'restore']:
            terminal = read(directory/('terminal-'+receipt[phase+'_run_id']+'.json'))
            if terminal['state'] != 'completed' or terminal['custody_phase'] != 'drained':
                raise ValueError('Both managed Runs must complete and drain')
        captured = read(project/'config.json')
        captured_configs.append({k:v for k,v in captured.items() if k != 'id'})
        objective = read(project/'fine-tuning.json')
        if history['checkpoints']['checkpoint-000.npz']['inventory'] != objective['initial_inventory']:
            raise ValueError('Actual CUDA initialization differs from qualified C53 tensors')
        objectives.append({k:v for k,v in objective.items() if k not in ['arm', 'low_mode_feedback']})
        closures.append({k:v for k,v in trained['closure'].items() if k not in ['experiment_sha256', 'source_witness_sha256']})
        by_operation = defaultdict(list)
        profile = read(directory/'readback-profile.json')
        for call in profile['calls']:
            by_operation[call['operation']].append(call['elapsed_s'])
        training.append(dict(id=arm['model_id'], arm=arm, source=str((directory/'receipt.json').relative_to(ROOT)),
            sha256=sha(directory/'receipt.json'), weights_sha256=receipt['weights_sha256'],
            project_id=receipt['project_id'], training_run_id=receipt['training_run_id'], restore_run_id=receipt['restore_run_id'],
            optimizer_updates=receipt['optimizer_updates'], artifact_parity=receipt['artifact_parity'],
            trainer_elapsed_s=trained['elapsed_s'], training_loop_and_bank_s=history['training_finished_unix']-history['training_started_unix'],
            managed_workflow_elapsed_s=receipt['elapsed_s'], peak_cuda_allocated_bytes=trained['execution']['peak_allocated_bytes'],
            peak_host_rss_bytes=trained['runtime']['peak_rss_bytes'], execution=trained['execution'],
            initialization_sha256=objective['initial_weights_sha256'], bank={**qualification['hard_bank'],
                'actual_training_description_sha256':bank['bank_description_sha256'],
                'actual_training_amplitude_rms_pa':bank['amplitude_rms_pa'],
                'cpu_gpu_description_exact':bank['bank_description_sha256']==qualification['hard_bank']['description_sha256']},
            actual_initial_inventory_matches_C53=True,
            trace=history['trace'], final_history=history['history'], optimizer_work=history['optimizer_work'],
            training_started_unix=history['training_started_unix'], training_finished_unix=history['training_finished_unix'],
            closure=trained['closure'],
            teacher_statistics=trained['fine_tuning_statistics'],
            readback=dict(reused_artifacts=profile['reused_artifacts'],
                scope='Typed operation calls only; owner session negotiation and raw restore approval contribute to workflow elapsed but lack separate timings.',
                operations={k:dict(calls=len(v), total_s=sum(v), maximum_s=max(v)) for k,v in by_operation.items()})))
    if not all(value == captured_configs[0] for value in captured_configs):
        raise ValueError('Training configurations differ beyond selected objective')
    if not all(value == objectives[0] for value in objectives) or not all(value == closures[0] for value in closures):
        raise ValueError('Initialization, teacher, data, order, scales or runtime differ')
    if not all(value == banks[0] for value in banks):
        raise ValueError('Actual CUDA bank descriptions/order differ between matched arms')
    def key(row):
        return row['trajectory_id'], row['native_frame'], row['passes']
    baseline = {key(row):row for row in snapshots['C53']['rows']}
    if len(baseline) != 512:
        raise ValueError('128 cases at four pass settings required')
    comparisons = {}
    for identifier, snapshot in snapshots.items():
        if (snapshot['status'] != 'passed' or snapshot['cuda_initialized'] or snapshot['torch_threads'] != 1
                or {key(row) for row in snapshot['rows']} != set(baseline)):
            raise ValueError('Unmatched CPU population/environment')
        entry = next(m for m in material['models'] if m['id'] == identifier)
        if snapshot['weights_sha256'] != entry['weights_sha256']:
            raise ValueError('Measured checkpoint identity differs')
        paired = []
        for passes in config['validation_passes']:
            selected = [r for r in snapshot['rows'] if r['passes'] == passes]
            ratios = [r['relative_residual']/baseline[key(r)]['relative_residual'] for r in selected]
            paired.append(dict(passes=passes, median_error_ratio_to_C53=statistics.median(ratios),
                largest_error_ratio_to_C53=max(ratios), improved=sum(r < 1 for r in ratios), worsened=sum(r > 1 for r in ratios)))
        comparisons[identifier] = paired
    old = ROOT/'runs/pressure-four-pass-distillation-v1/isolated-cpu-02'
    replay = {}
    for identifier in config['baselines']:
        previous = {key(row):row for row in read(old/identifier/'snapshots.json')['rows']}
        replay[identifier] = all(r['relative_residual'] == previous[key(r)]['relative_residual'] for r in snapshots[identifier]['rows'])
    if not all(replay.values()):
        raise ValueError('Historical baseline numerical replay changed')
    diagnoses = {}
    amplitude_sensitivity = {}
    for identifier in identifiers:
        source = projects/(identifier+'-diagnosis.json') if identifier in config['baselines'] else cpu/(identifier+'-diagnosis.json')
        diagnosis = read(source)
        if diagnosis['weights_sha256'] != snapshots[identifier]['weights_sha256']:
            raise ValueError('Spectral and snapshot checkpoints differ')
        diagnoses[identifier] = diagnosis
        spreads = []
        for grid in config['hard_bank']['grids']:
            for mode in config['diagnostic_modes_zyx']:
                values = [r['relative_equation_residual'] for r in diagnosis['rows'] if r['passes']==4
                          and r['counts_xyz']==grid['counts_xyz'] and r['mode_zyx']==mode and r['amplitude_multiplier']>0]
                spreads.append(max(values)/min(values))
        amplitude_sensitivity[identifier] = dict(passes=4, positive_amplitude_range=[1e-4,100.],
            median_error_spread_ratio=statistics.median(spreads), maximum_error_spread_ratio=max(spreads),
            definition='For each grid/mode: maximum divided by minimum relative residual across four positive amplitudes; these 48 ratios are summarized, not pooled pressure energies.')
    def summary(identifier, passes):
        return next(r for r in snapshots[identifier]['summary'] if r['passes'] == passes)
    def manufactured(identifier, passes):
        return sum(r['relative_equation_residual'] <= config['target_relative_residual'] for r in diagnoses[identifier]['rows'] if r['passes'] == passes)
    eligible = []
    gates = []
    for arm in config['arms']:
        identifier = arm['model_id']
        row = dict(model=identifier, four_pass_all_128=summary(identifier,4)['equation_passes']==128,
            eight_pass_all_128=summary(identifier,8)['equation_passes']==128,
            earlier_pass_worst_no_worse_than_C53=all(summary(identifier,p)['worst_equation_residual']<=summary('C53',p)['worst_equation_residual'] for p in [1,2]),
            four_pass_worst_no_worse_than_C53=summary(identifier,4)['worst_equation_residual']<=summary('C53',4)['worst_equation_residual'],
            manufactured_eight_pass_count_no_worse_than_C53=manufactured(identifier,8)>=manufactured('C53',8))
        row['eligible_for_independent_scenes'] = all(v for k,v in row.items() if k != 'model')
        gates.append(row)
        if row['eligible_for_independent_scenes']:
            eligible.append(identifier)
    eligible.sort(key=lambda identifier:(summary(identifier,4)['worst_equation_residual'], summary(identifier,4)['median_inference_ms'], identifier))
    decision = dict(status='screen_completed', baseline_retained='C53', eligible_for_independent_scenes=eligible,
        provisional_candidate=eligible[0] if eligible else None,
        gates=gates, model_promoted=False, longer_training_started=False, original_untouched_holdouts_consumed=0,
        next_action='Independent scene qualification required before longer training' if eligible else
            'Do not extend these fits. First test fitting one fundamental mode on the hardest small grid; if it fits, test a frequency-balanced low-mode curriculum. If it cannot fit, investigate optimization/objective interference before a structural correction change.',
        qualification_population='Consumed 128 real pressure requests; manufactured 240 cases reported separately',
        attribution='C60/C61 are matched. C59 is a historical reference with different random sampling; its difference from C60 does not isolate the bank change.')
    output.mkdir(exist_ok=True)
    exposure = read(output/'bank-exposure.json')
    if exposure['sampling_order_sha256'] != training[0]['bank']['order_sha256']:
        raise ValueError('Exposure analysis differs from matched actual bank sampling')
    result = dict(status='completed', config=read(config_path), config_sha256=sha(config_path), training=training,
        bank_exposure=exposure,
        snapshots=snapshots, comparisons=comparisons, diagnosis=diagnoses, amplitude_sensitivity=amplitude_sensitivity, decision=decision,
        material_sha256=sha(cpu/'material.json'),
        selection_policy=policy, selection_policy_sha256=sha(output/'selection-policy.json'),
        historical_numerical_replay_exact=replay, cpu_execution=execution,
        managed_execution=read(projects/'controller-execution.json'), resource_observations=read(projects/'resource-observations.json'),
        timing_scope='Isolated single-thread CPU recurrence plus final equation check; median of three timed requests after one warm-up. Data assembly excluded.',
        error_conventions='Real snapshot residual/full pressure RHS. Manufactured zero-base residual/initial residual. Training log equation feedback/initial correction-request residual. These denominators are not interchangeable.',
        source_sha256=sha(__file__))
    write(output/'results.json', result)
    write(output/'decision.json', decision)
    for identifier in identifiers:
        write(output/(identifier+'-diagnosis.json'), diagnoses[identifier])
    shutil.copyfile(cpu/'execution.json', output/'cpu-execution.json')
    shutil.copyfile(cpu/'material.json', output/'material.json')
    print('Retained screen; eligible:', eligible, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'projects', 'cpu', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    report(args.config.resolve(), args.projects.resolve(), args.cpu.resolve(), args.output.resolve())
