"""Calibrate float32 output targets, then diagnose actual restored predictions."""
import argparse
import json
from pathlib import Path
import sys

from targets import assess, calibrated_relative_target, load as load_profile

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
sys.path.insert(0, str(LAB/'experiments/variable-pressure-v2'))
import build_native
import data
import evaluate as native_evaluation
import model
import pipeline
from policy import MIB, canonical, decode, fresh, read, require, sha, write
import numpy as np
import torch


def request_for(parent, frame_index, pressure, oracle):
    root = Path(parent['capture_root'])
    raw = read(root/'manifest.json', 4*MIB)
    require(sha(raw) == parent['capture_manifest_sha256'], 'capture_drift', 'same qualified physical history')
    trajectory = next(t for t in decode(raw)['trajectories'] if t['trajectory_id'] == parent['entry']['trajectory_id'])
    frame = trajectory['frames'][frame_index]; counts = parent['counts_xyz']
    velocity = [oracle.load_array(root, frame['arrays']['pre_'+axis], oracle.shape(d, counts), '<f4', 'm/s')
                for d, axis in enumerate('uvw')]
    fields = {name: parent['arrays'][name][frame_index].reshape(-1).tolist() for name in ('rhs', 'base', 'reference')}
    return {'schema': 'rheon-variable-pressure-teacher-case-v1', 'counts_xyz': counts,
            'spacing_m': [parent['cell_size_m']]*3, 'density_kg_m3': 1.25, 'dt_s': parent['entry']['dt_s'],
            'original_rhs': fields['rhs'], 'reference_pressure': fields['reference'], 'previous_pressure': fields['base'],
            'pre_velocity': velocity, 'candidate_pressure': np.asarray(pressure, dtype='<f8').reshape(-1).tolist(),
            'candidate_error': None}, trajectory


def diagnose(binary, parent, frame, pressure, oracle, validator):
    request, trajectory = request_for(parent, frame, pressure, oracle)
    response, process = build_native.call(binary, '--evaluate', request)
    independent = native_evaluation.check_response(request, response, oracle, validator)
    return response, process, independent, trajectory


def append(path, record):
    with path.open('ab') as stream:
        stream.write(canonical(record)+b'\n')


def run(args):
    output = fresh(args.output.absolute()); observation = pipeline.observer(args.live_root.absolute())
    profile_path = args.profile.absolute(); profile = load_profile(profile_path)
    try:
        native = build_native.admit(args.binary)
        manifest_raw = read(args.packet/'splits.json', 4*MIB); manifest = decode(manifest_raw)
        _, parents = data.load(read(args.packet/'dataset.npz', 32*MIB), manifest)
        require(manifest['native_build_receipt'] == native, 'native_identity', 'same qualified physical operator')
        norm_raw = read(args.restored/'normalization.json', MIB); normalization = decode(norm_raw)
        trained_raw = read(args.training/'receipt.json', MIB); restored_raw = read(args.restored/'receipt.json', MIB)
        trained, restored = decode(trained_raw), decode(restored_raw)
        require(trained['schema'] == restored['schema'] == 'rheon-variable-pressure-training-v2'
                and trained['mode'] == 'train' and restored['mode'] == 'restore' and restored['optimizer_updates'] == 0
                and trained['closure'] == restored['closure'] and restored['originating_training_run'] is not None,
                'managed_restore', 'same retained completed CPU experiment')
        require(normalization == data.fit_normalization(parents) and sha(norm_raw) == restored['closure']['normalization_sha256'],
                'normalization', 'exact train-only statistics')
        for name, item in restored['outputs'].items():
            raw = read(args.restored/name, 4*MIB)
            require(len(raw) == item['bytes'] and sha(raw) == item['sha256'], 'restored_artifact', name)
            if name != 'summary.json':
                require(raw == read(args.training/name, 4*MIB), 'restore_parity', name)
        predictions = data.helpers().read_npz(read(args.restored/'predictions.npz', 2*MIB), restored['prediction_inventory'], 2*MIB)
        oracle, validator = native_evaluation.historical_oracles(); model.configure_cpu()
        write(output/'profile.json', profile)
        calibration = []
        for split in ('train', 'validation'):
            for parent in parents:
                if parent['entry']['split'] != split:
                    continue
                for frame in range(24):
                    sample = data.sample(parent, frame)
                    # Ideal supervised correction through exactly the current f32
                    # head representation, output rescale and gauge operations.
                    scale = normalization['scales']['pressure']['value']
                    raw = ((sample['reference']-sample['base'])/scale).to(torch.float32)
                    pressure = sample['base']+model.frozen.gauge_correction(raw, scale)
                    result, process, independent, trajectory = diagnose(args.binary, parent, frame, pressure.numpy(), oracle, validator)
                    outcome = assess(result['raw_learned'], profile)
                    record = {'split': split, 'trajectory_id': parent['entry']['trajectory_id'], 'frame': frame,
                              'candidate_kind': 'ideal reference correction rounded through float32 head; not learned',
                              'pressure_sha256': sha(pressure.numpy().tobytes()), 'native': result,
                              'profile_assessment': outcome, 'process': process, 'independent_residual_l2': independent}
                    append(output/'calibration.jsonl', record); calibration.append(record)
                    if len(calibration)%6 == 0:
                        entry = {**parent['entry'], 'counts_xyz': parent['counts_xyz'], 'extents_m': trajectory['geometry']['extents_m']}
                        preview = observation.frame(Path(parent['capture_root']), entry, frame, pressure=pressure.numpy(),
                                  pressure_label='Float32-rounded ideal reference correction; precision calibration, not learned prediction')
                        observation.publish('Float32 precision calibration', f'{split}: ideal reference correction through float32; {len(calibration)}/144 frames',
                                            len(calibration), sum(len(p['frames']) for p in parents), preview=preview)
            if split == 'train':
                training = [r for r in calibration if r['split'] == 'train']
                peak = max(r['native']['raw_learned']['relative_residual'] for r in training)
                selected = calibrated_relative_target(peak, profile['calibration'])
                require(selected == profile['relative_residual'], 'target_calibration', 'profile must match train-only rounding margin')
                write(output/'selection.json', {'population': 'train_only', 'frames': len(training), 'maximum_relative_residual': peak,
                      'selected_relative_residual': selected, 'selection_completed_before_validation': True,
                      'policy': profile['calibration']})
        candidates = []
        for index, case in enumerate(restored['cases']):
            parent = parents[case['parent']]; pressure = predictions[f'case_{index:03d}']
            result, process, independent, trajectory = diagnose(args.binary, parent, case['frame'], pressure, oracle, validator)
            outcome = assess(result['raw_learned'], profile)
            record = {**case, 'candidate_kind': 'actual restored seven-channel model', 'native': result,
                      'profile_assessment': outcome, 'process': process, 'independent_residual_l2': independent}
            append(output/'model-cases.jsonl', record); candidates.append(record)
            entry = {**parent['entry'], 'counts_xyz': parent['counts_xyz'], 'extents_m': trajectory['geometry']['extents_m']}
            preview = observation.frame(Path(parent['capture_root']), entry, case['frame'], pressure=pressure,
                                       pressure_label=f'Actual learned proposal; float32 target {profile["relative_residual"]}; '
                                                      f'{"passed" if outcome["qualified"] else "failed"}; reference tracer')
            observation.publish('Float32 model-target diagnostics', f'Actual restored model: {index+1}/{len(restored["cases"])} snapshots',
                                index+1, len(restored['cases']), preview=preview)
        require(build_native.admit(args.binary) == native, 'native_drift', 'same bound operator and historical gates')
        def summarize(rows):
            return {'frames': len(rows), 'native_strict_qualified': sum(r['native']['raw_learned']['qualified'] for r in rows),
                    'float32_profile_qualified': sum(r['profile_assessment']['qualified'] for r in rows),
                    'maximum_relative_residual': max(r['native']['raw_learned']['relative_residual'] for r in rows),
                    'maximum_predicted_divergence_s_inv': max(r['native']['raw_learned']['predicted_divergence_max_s_inv'] for r in rows),
                    'maximum_actual_f32_divergence_s_inv': max(r['native']['raw_learned']['actual_divergence_max_s_inv'] for r in rows)}
        report = {'schema': 'rheon-float32-target-comparison-v1', 'status': 'complete', 'profile': profile,
                  'calibration': {split: summarize([r for r in calibration if r['split'] == split]) for split in ('train','validation')},
                  'actual_model': summarize(candidates), 'optimizer_updates': 0,
                  'weights_sha256': sha(read(args.restored/'weights.npz', 2*MIB)),
                  'packet_sha256': manifest['dataset_sha256'], 'splits_sha256': sha(manifest_raw),
                  'normalization_sha256': sha(norm_raw), 'training_receipt_sha256': sha(trained_raw), 'restore_receipt_sha256': sha(restored_raw),
                  'source_sha256': {(p.relative_to(LAB).as_posix() if p.is_relative_to(LAB) else str(p)): sha(read(p))
                                    for p in [HERE/'targets.py', HERE/'run.py', profile_path]},
                  'native_binary_sha256': native['binary_sha256'],
                  'unchanged_physical_limits': {'predicted_divergence_s_inv': 1e-7, 'actual_f32_divergence_s_inv': 1e-5,
                       'absolute_net_flux_m3_s': 1e-12, 'relative_net_flux': 1e-8, 'courant': .5, 'wall_normal_flux_exact_zero': True},
                  'scope': 'float32 learned-output acceptance; native reference and conventional solver gates remain stricter; no training, rollout or acceleration claim'}
        write(output/'report.json', report)
        observation.publish('Float32 model target calibrated', f'Relative residual target {profile["relative_residual"]}; '
                            f'rounded-reference {sum(r["profile_assessment"]["qualified"] for r in calibration)}/{len(calibration)}; '
                            f'actual model {report["actual_model"]["float32_profile_qualified"]}/{len(candidates)}',
                            len(candidates),len(candidates),state='completed')
        return report
    except BaseException as error:
        write(output/'failure.json', {'status':'failed','error':str(error)})
        observation.publish('Float32 target diagnostic failed', 'Retained prefix is diagnostic; no automatic retry',state='failed')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet','binary','training','restored','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--profile',type=Path,default=HERE/'configs/development.json')
    parser.add_argument('--live-root',type=Path,default=LAB/'runs/live-project')
    print(json.dumps(run(parser.parse_args())))


if __name__ == '__main__':
    main()
