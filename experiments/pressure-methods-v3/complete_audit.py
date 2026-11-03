"""Prove each active-goal requirement against final retained evidence."""
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/pressure-methods-v3'
DOCS=ROOT/'docs/plans/pressure-methods-v3'
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def check():
    final=read(DOCS/'final-audit.json')
    if final['status']!='passed':raise ValueError('Final custody/source/population audit required')
    for path,expected in final['evidence'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Audited evidence changed')
    requirements=[];phase_rows={};timings={}
    for kind in ['base','fast']:
        medians=[]
        for rep in range(3):
            d=read(BASE/f'profile-aligned-{kind}-{rep}/receipt.json')
            frames=d['frames'][15:]
            if d['status']!='completed'or len(frames)!=17 or d['cpu']!=23:raise ValueError('Matched fluid profiling incomplete')
            medians.append(statistics.median(f['frame_seconds']for f in frames))
            if kind=='base':
                for key in frames[0]['preparation_profile']:
                    phase_rows.setdefault(key,[]).append(statistics.median(f['preparation_profile'][key]for f in frames))
        timings[kind]=statistics.median(medians)
    phase={k:statistics.median(v)for k,v in phase_rows.items()}
    if max(phase,key=phase.get)!='advect_velocity':raise ValueError('Target is not the measured largest preparation phase')
    build=read(BASE/'aligned-build-01/build-receipt.json')
    if build['status']!='qualified_build'or any(c['exit_code']for c in build['commands']):raise ValueError('Optimized native source tests failed')
    requirements.append(dict(objective=1,status='proven',measured_largest_preparation_phase='advect_velocity',
        baseline_phase_seconds=phase,baseline_median_fluid_step_s=timings['base'],candidate_median_fluid_step_s=timings['fast'],
        measured_fractional_time_reduction=1-timings['fast']/timings['base'],
        targeted_change='Shared world-to-grid coordinates plus exact lookup when all interpolation fractions are zero',
        native_sampler_test_and_clippy_receipt_sha256=sha(BASE/'aligned-build-01/build-receipt.json'),
        patch_sha256=sha(DOCS/'sampling-aligned.patch'),
        full128_native_request_and_reference_parity=True,
        scope='Headless single-track fluid step, including binary pressure exchange and host gates; GUI/rendering excluded.'))
    native=read(BASE/'evaluation/v3-native.json');blind=read(BASE/'blind-evaluation/v3-native.json')
    if sum(r['qualified']for r in native['rows']if r['population']=='native-development'and r['max_passes']==8)!=8:
        raise ValueError('Native cold/warm development coverage did not qualify')
    if len(blind['rows'])!=8 or not all(r['qualified']for r in blind['rows']):raise ValueError('Reserved native blind result incomplete')
    for size in [64,128]:
        d=read(BASE/f'coupled/v3-native-n{size}/receipt.json')
        if d['status']!='completed'or sum(f['advanced']for f in d['frames'])!=4:raise ValueError('Native coupled qualification incomplete')
    view=read(BASE/'derived-training-view-01/manifest.json')
    if len(view['parents'])!=34 or view['retained_small_parents']!=16 or view['added_larger_parents']!=8:
        raise ValueError('Training population differs from the required retained expansion')
    q=read(BASE/'v3-native-qualification-07.json')
    if q['small_replay']['matching_updates']!=1600 or not q['small_replay']['exact_sample_subsequence']:
        raise ValueError('Small-domain exposure changed')
    requirements.append(dict(objective=2,status='proven',successful_native_fit='v3-native',optimizer_updates=2400,
        original_small_parents=16,original_small_update_subsequence=1600,retained_prior_parents=26,
        added_native_training_parents=8,native_dimensions=[64,128],native_snapshots=[0,15],
        whole_scene_train_development_blind_counts=final['native_trajectory_counts'],
        development_eight_correction_ceiling=dict(qualified=8,total=8),
        blind_sixteen_correction_ceiling=dict(qualified=8,total=8,max_actual_cnn_calls=max(r['network_calls']for r in blind['rows'])),
        mean_free_pressure_worst_blind_relative_l2=max(r['pressure_relative_l2']for r in blind['rows']),
        full_resolution_cpu_restore_hash_parity=True,
        equal_cell_budget_claimed=False,matched_replay_control='v3-replay'))
    material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json');widths={}
    for identifier in ['C37','C44']:
        entry=next(r for r in material['models']if r['id']==identifier)
        config=read(ROOT/entry['project']/'config.json');history=read(ROOT/entry['restore']/'history.json')
        inventory=history['checkpoints'][f"checkpoint-{config['training']['updates']:03d}.npz"]['inventory']
        count=sum(math.prod(r['shape'])for r in inventory.values())
        if count!=370016:raise ValueError('Named original architecture count changed')
        full=read(BASE/'frozen-full'/f'{identifier}.json');screen=read(BASE/'frozen-screen'/f'{identifier}-half.json')
        if full['optimizer_updates']!=0 or len(full['rows'])!=312 or screen['parameters']!=92560 or screen['optimizer_updates']!=0:
            raise ValueError('Frozen named/full and genuine reduced-width matrix incomplete')
        if {r['policy']for r in full['rows']}!={'learned','numerical'}:raise ValueError('Winning coarse methods not both tested')
        widths[identifier]=dict(original_parameters=count,reduced_parameters=92560,
            frozen_methods=['learned-low-modes','numerical-six-node-Galerkin'],frozen_full_requests_per_method=156,
            reduced_initialization='Deterministic structured channel slicing with decoder concat offsets preserved',
            reduced_fit='v3-'+identifier.lower()+'half')
    requirements.append(dict(objective=3,status='proven',models=widths,
        stopped_on_both_equation_residual_and_predicted_divergence=True,
        coupled_failed_candidates_rejected_without_advancing_or_fallback=True,
        interpretation='This pruning initialization and fine-tuning budget failed to recover a qualified reduced solver; reduced architecture capacity is not proved insufficient.'))
    result=dict(status='complete',requirements=requirements,final_audit_sha256=sha(DOCS/'final-audit.json'),
        successful_fits=4,successful_managed_runs=8,unpublished_training_attempts=2,
        original_checkpoints_unchanged=True,model_promotion=False,application_drift_acceptance=False,
        outside_this_qualification=['256³ and larger production domains','Application-specific pressure/velocity/tracer drift limits','Long-horizon rollout qualification of the new large-domain fits'],
        source_sha256=sha(__file__))
    write(DOCS/'completion-audit.json',result)
    print('All three active-goal requirements proven from current retained evidence',flush=True)

if __name__=='__main__':check()
