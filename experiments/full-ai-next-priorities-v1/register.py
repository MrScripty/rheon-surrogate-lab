"""Register the original plan, experiments and scene splits before any fitting."""
import random
import shutil
import time
from study_support import ROOT, BASE, DOCS, read, write, sha

REQUIREMENTS = {
    'A-restore': 'Exact C107 restore and reproduction of all 128 historical residuals',
    'A-timing': 'Five balanced CPU timing repeats of C106 and C107 with declared warmup',
    'A-profile': 'Stage, convolution, global branch, operator, conversion and allocation profiling plus residual diagnostics',
    'A-execution': 'Cache, buffer and CPU layout or compiled execution comparisons with output, gauge and odd-grid qualification',
    'A-four-fits': 'Four 1200-update continuations from C107 with 600 and 1200 checkpoints and the declared learning rates',
    'A-tail': 'Frozen training-only error ranking and capped weights preserving spatial tier proportions',
    'A-selection': 'Development selection using worst error, latency and population regression metrics',
    'A-architecture': 'Up to two conditional architecture fits if continuation leaves a gap',
    'A-seeds': 'Two further seeds for up to two winning training recipes',
    'A-fresh': 'Frozen finalists evaluated on fresh reserved pressure requests and native trajectories',
    'A-target': 'C107 successor at approximately 4ms median inference and worst residual <=7e-4',
    'B-contract': 'Complete causal state, staggering, forcing and integrator restart contract',
    'B-restart': 'At least eight subsequent native reference steps reproduced after restoring a saved state',
    'B-bank': '32 scene trajectories, four grid groups, two spacings, split 16/8/8 before window construction',
    'B-frames': 'Initial state plus consecutive 128-step train/development and 240-step reserved captures',
    'B-causality': 'No future reference input and no overlap of scenes across train/development/reserved sets',
    'B-four-models': 'Shared architecture family fitted for K=1,2,3,4 with all K outputs available',
    'B-loss': 'Equal horizon weights, train-only scales, pressure gauge removal and declared physical losses',
    'B-continuation': 'Training across at least two blocks of predicted history including K=1 control',
    'B-seeds': 'Conditional seed replications for promising bundled models and one-step control',
    'B-rollout': 'Single-block and own-history rollout errors at 12,48,240 physical steps for every horizon',
    'B-physical': 'Pressure equation, pressure/velocity/tracer relative and absolute errors, divergence, flux, mass, bounds and energy',
    'B-timing': 'Five balanced CPU timing repeats with complete block, per-step, tensor-only and application timing',
    'B-controls': 'Sequential K=1, native C107-pressure and numerical reference comparisons at matched physical intervals',
    'B-events': 'Unexpected interactions invalidate future outputs; discarded work and any fallback counted',
    'B-target': 'K=2,3,4 each under2ms per generated step with worst equation and field errors <=4e-3',
    'P-custody': 'Finite resource limits, source/data identities, actual training receipts and exact CPU restores',
    'P-report': 'Separate frontiers, profiles, manifests, per-case errors, repeated timings, rollouts and failed outcomes',
    'P-immutable': 'Prior completed studies unchanged and no numerical pressure solve in candidate inference',
}

def run():
    if (DOCS / 'registration.json').exists():
        raise ValueError('Registration already exists; inspect current state rather than replaying')
    BASE.mkdir(parents=True, exist_ok=True)
    baseline = read(DOCS / 'baseline-evidence.json')
    for source, digest in baseline['sources_sha256'].items():
        if sha(ROOT / source) != digest:
            raise ValueError('Original baseline evidence changed: ' + source)
    shutil.copyfile(DOCS / 'plan.md', DOCS / 'requested-scope.md')
    registration = dict(schema='rheon-full-ai-study-v1', status='registered', registered_unix=time.time(),
        requested_scope_sha256=sha(DOCS / 'requested-scope.md'), plan_sha256=sha(DOCS / 'plan.md'),
        baseline_evidence_sha256=sha(DOCS / 'baseline-evidence.json'),
        user_authorization='Active goal: complete all objectives in docs/plans/full-ai-next-priorities-v1/plan.md',
        accuracy_interpretation=dict(equation=4e-3, pressure=4e-3, velocity=4e-3, tracer=4e-3,
            choice='Plan stronger criterion; accepted as the pre-fit research criterion under the active whole-plan goal',
            application_limits=False, denominator_floor='Training-only calibration before temporal fitting'),
        timing=dict(cpu=23, threads=1, batch_size=1, repeats=5, warmup='One full population pass before measurement',
                    multi_step_target_ms=2., target_scope='Complete model block inference divided by K',
                    report_full_block=True, report_tensor_only=True, report_application=True),
        track_a=dict(base='C107', updates=1200, checkpoints=[0,600,1200], base_lr=3e-5,
            schedule=[dict(update=901,learning_rate=1e-5)], arms=['a-control','a-head','a-tail','a-combined'],
            head_multipliers={'a-control':10/3,'a-head':1.,'a-tail':10/3,'a-combined':1.},
            tail_multiplier_cap=4, seed=2026101907, temporal_corrections=2,
            cpu_target_ms=4., worst_equation_target=7e-4,
            screen_selection='Minimum development worst residual among finite regression-qualified checkpoints satisfying latency, with complete Pareto record'),
        track_b=dict(horizons=[1,2,3,4], train_steps=128, development_steps=128, reserved_steps=240,
            rollout_horizons=[12,48,240], numerical_solve_at_candidate_inference=False,
            restart_comparison=dict(steps=8, tolerance='Bitwise equality preferred; explicit tolerance must be frozen before comparison'),
            update_budget='Freeze after qualified per-update pilot, before full fits'),
        resource_policy=dict(max_training_attempt_seconds=1800, automatic_retry=False,
            cpu_timing_serial=True, cuda_fits_serial=True, original_archives_immutable=True),
        requirements=[dict(id=k,requirement=v,status='unproven',evidence=[]) for k,v in REQUIREMENTS.items()],
        promotion=False)
    write(DOCS / 'registration.json', registration)
    rng=random.Random(2026101917)
    scenes=[]
    for group,counts in enumerate(([16,16,16],[9,11,11],[15,6,8],[32,32,32])):
        for cpm in (8,16):
            for rep,split in enumerate(('train','train','development','reserved')):
                scenes.append(dict(identifier=f'temporal-g{group}-cpm{cpm}-r{rep}', split=split,
                    seed=rng.randrange(1,2**31), counts=list(counts), cells_per_metre=cpm,
                    steps=240 if split=='reserved' else 128, warmup_steps=0,
                    center=[rng.uniform(.25,.65) for _ in range(3)], acceleration=rng.uniform(.08,.16),
                    radius=rng.uniform(.15,.25), swirl=rng.uniform(-.6,.6),
                    fluctuation=rng.uniform(.1,.3),frequency=rng.uniform(1.,2.5),sgs=False))
    write(DOCS / 'temporal-scene-reservation.json',dict(status='reserved_before_capture',
        seed=2026101917, rows=scenes, registration_sha256=sha(DOCS/'registration.json'),
        counts={s:sum(r['split']==s for r in scenes) for s in ('train','development','reserved')},
        reserved_consumption='Only after model and execution selection are frozen'))
    write(BASE/'work-state.json',dict(status='active',stage='baseline_reproduction_and_state_audit',
        original_goal_complete=False,requirements=len(REQUIREMENTS), live_workers=[], updated_unix=time.time()))
    print('Registered',len(REQUIREMENTS),'requirements and',len(scenes),'temporal scene declarations',flush=True)

if __name__=='__main__':
    run()
