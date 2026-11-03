"""Closed managed fit, four typed restores, and complete development evidence."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from report_temporal_control_guides_v2 import summary


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def run(args):
    if args.output.exists():
        raise ValueError('Fresh guided fit report required')
    pair, evaluation = read(args.pair / 'receipt.json'), read(args.evaluation)
    trained, restored = read(args.pair / 'train/receipt.json'), read(args.pair / 'restore/receipt.json')
    history = read(args.pair / 'restore/history.json')
    if (pair['status'] != 'completed_managed_guided_fit_and_four_checkpoint_CPUrestore'
            or not all(pair['artifact_parity'].values())
            or evaluation['status'] != 'completed_guided_all_development_native_accuracy_and_GPU_cost'
            or not evaluation['all_requested_checkpoints_covered']):
        raise ValueError('Closed fit/restoration and complete development evaluation required')
    result = dict(status='completed_pressure_fit_and_all_development_evaluation',
        created_unix=time.time(), new_parameter_fits=1, cumulative_parameter_fits=11,
        training_run=pair['training_run_id'], CPUrestore_run=pair['restore_run_id'],
        parameter_trial_updates=trained['optimizer_updates'], accepted_steps=trained['accepted_steps'],
        independent_training_masters=24, selected_training_windows=240,
        initial_training_objective=history['initial_full_score']['objective'],
        selected_training_score=history['selected_worst_score'],
        mean_loss_percent_reduction=100*(1-history['best_mean_score']['objective']/history['initial_full_score']['objective']),
        iterations=[{k:r[k] for k in ('iteration','gradient_norm','proposal','accepted')} for r in history['iterations']],
        training_trials=[dict(label=r['label'], step=r['step'], objective=r['score']['objective'],
            worst_relative=r['score']['worst_relative'], worst_locations=r['score']['worst_locations']) for r in history['trials']],
        artifact_parity=pair['artifact_parity'], copied_GPU_probe_shape=history['probe_shape'],
        cpu_model_forward=False, inference_device='GPU (CUDA)',
        checkpoint_aliases=evaluation['checkpoint_aliases'], alias_policy=evaluation['alias_rule'],
        independent_development_masters=12, development_is_not_untouched_confirmation=True,
        native_development_blocks=len(evaluation['rows']),
        native_development_physical_outputs=sum(r['K'] for r in evaluation['rows']),
        complete_GPU_timing_calls=sum(len(r['block_wall_ms']) for r in evaluation['rows']),
        development={}, reserved52_references_opened=0, model_promotion=False,
        full_generator_target_pass=False, long600step_phase_started=False)
    for checkpoint in evaluation['representative_checkpoints']:
        rows = [r for r in evaluation['rows'] if r['checkpoint'] == checkpoint]
        item = summary(rows)
        item['worst_locations'] = {}
        for metric in ('equation', 'pressure', 'velocity', 'tracer'):
            value, location = -1., None
            for row in rows:
                for lead, measured in enumerate(row['native_metrics_per_output'], 1):
                    actual = measured['relative'][metric]
                    if actual >= value:
                        value = actual
                        location = {k:row[k] for k in ('identifier', 'counts_xyz', 'cells_per_metre',
                            'force_profile', 'force_motion', 'start_fine_step', 'output_steps_per_s', 'K')}
                        location.update(lead=lead, relative_error=actual)
            item['worst_locations'][metric] = location
        item['all_eager_graph_outputs_byte_identical'] = all(r['eager_graph_complete_byte_parity'] for r in rows)
        item['stratified'] = {}
        for group, key in [('density',lambda r:r['cells_per_metre']),('counts_xyz',lambda r:tuple(r['counts_xyz'])),
                           ('cadence_K',lambda r:(r['output_steps_per_s'],r['K'])),
                           ('force_profile',lambda r:r['force_profile']),('force_motion',lambda r:r['force_motion'])]:
            groups = defaultdict(list)
            for row in rows: groups[str(key(row))].append(row)
            item['stratified'][group] = {name:summary(rs) for name,rs in groups.items()}
        result['development'][checkpoint] = item
    initial_metrics = result['development']['initial']['worst_relative']
    for checkpoint, measured in result['development'].items():
        measured['worst_error_change_from_initial'] = {n: v-initial_metrics[n]
            for n,v in measured['worst_relative'].items()}
        measured['no_field_worst_error_regression_from_initial'] = all(
            v <= initial_metrics[n] for n,v in measured['worst_relative'].items())
    result['training_selected_field_regressions_from_initial'] = [n
        for n,v in history['selected_worst_score']['worst_relative'].items()
        if v > history['initial_full_score']['worst_relative'][n]]
    initial = history['checkpoint_inventories']['initial-weights.npz']
    selected = history['checkpoint_inventories']['weights.npz']
    result['changed_new_network_tensors'] = [n for n in selected if selected[n] != initial[n]]
    evidence = [args.evaluation,args.pair/'receipt.json',args.pair/'train/receipt.json',
        args.pair/'restore/receipt.json',args.pair/'restore/history.json',
        *[args.pair/'restore'/name for name in ('weights.npz','best-mean-weights.npz','initial-weights.npz','last-evaluated-weights.npz','probe.npy','normalization.json')]]
    result['evidence_sha256'] = {str(p):sha(p) for p in evidence}
    result['source_sha256'] = {str(Path(__file__).resolve()):sha(Path(__file__).resolve())}
    result['interpretation'] = ('Independent pressure-RHS and projected velocity/tracer corrections use the same '
        'full24 training data and frozen native losses. Report exact checkpoint-specific development results; '
        'gradient scale or mean training improvement alone does not establish worst-error improvement. '
        'Both architecture and maximum calibrated ray radius changed; this is not an optimizer-identical '
        'architecture ablation. No promotion or target pass is inferred from development selection.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],mean_loss_percent_reduction=result['mean_loss_percent_reduction'],
        checkpoints={c:{k:v for k,v in r.items() if k in ('blocks','accuracy_pass_blocks','latency_pass_blocks','worst_relative',
            'maximum_GPU_ms_per_physical_output','median_of_block_median_GPU_ms_per_physical_output')} for c,r in result['development'].items()}),indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
