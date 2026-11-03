"""Publish completed prototype evidence without claiming the wider targets passed."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
DOCS=ROOT/'docs/plans/full-ai-tight-targets-v1'
BASE=ROOT/'runs/full-ai-tight-targets-v1'


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run():
    destinations=[DOCS/'recurrent-sampler-results-01.md',DOCS/'recurrent-sampler-summary-01.json',DOCS/'work-state-04.json']
    if any(p.exists()for p in destinations):raise ValueError('Fresh complete prototype publication required')
    paths=dict(pair=BASE/'sampler-head-pair-01/receipt.json',qualification=BASE/'trained-sampler-qualification-01.json',
        development_blocks=BASE/'trained-sampler-development-blocks-01/results.json',
        development_rollout=BASE/'trained-sampler-development-rollout-01/results.json',
        fresh_blocks=BASE/'trained-sampler-fresh-blocks-01/results.json',
        fresh_rollout=BASE/'trained-sampler-fresh-rollout-01/results.json',
        development_cost=BASE/'trained-sampler-cpu23-cost-01.json',
        fresh_cost=BASE/'trained-sampler-fresh-cpu23-cost-01.json',
        application=BASE/'trained-sampler-application-01/results.json',
        application_accuracy=BASE/'trained-sampler-application-accuracy-01.json')
    data={key:read(path)for key,path in paths.items()}
    if data['pair']['status']!='completed'or not all(data['pair']['artifact_parity'].values()):raise ValueError('Exact managed pair incomplete')
    if data['qualification']['status']!='passed_trained_complete_graph_execution'or data['qualification']['complete_states']!=64:raise ValueError('Actual graph qualification incomplete')
    for key,count in [('development_blocks',160),('fresh_blocks',320),('development_rollout',7680),('fresh_rollout',15360)]:
        if not data[key]['status'].startswith('completed_')or len(data[key]['rows'])!=count:raise ValueError('Complete declared physical population missing: '+key)
    app=data['application'];timed=[row for row in app['rows']if not row['warmup']]
    if app['status']!='completed_matched_application_comparison'or len(timed)!=120 or len(app['rows'])!=144:raise ValueError('Application comparison incomplete')
    if data['application_accuracy']['status']!='completed_deferred_application_accuracy':raise ValueError('Deferred physical accuracy missing')
    checks=[]
    for path in sorted(DOCS.glob('*.json')):
        value=read(path)
        for field in ('source_sha256','source_evidence'):
            sources=value.get(field)
            if not isinstance(sources,dict):continue
            for name,expected in sources.items():
                if not isinstance(expected,str)or len(expected)!=64:continue
                if digest(ROOT/name)!=expected:raise ValueError('Registered source/evidence changed: '+str(path)+' '+name)
            checks.append(dict(registration=str(path.relative_to(ROOT)),field=field,sources_verified=len(sources)))
    activation=read(DOCS/'activation-01.json')
    for path,expected in activation['source_evidence'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Original completed study changed')
    evidence={str(path.relative_to(ROOT)):digest(path)for path in paths.values()}
    horizon_equality={}
    for key in ('development_rollout','fresh_rollout'):
        baseline={(row['identifier'],row['step']):row['output_identity']for row in data[key]['rows']if row['horizons']==1}
        horizon_equality[key]=all(row['output_identity']==baseline[row['identifier'],row['step']]for row in data[key]['rows'])
    summary=dict(status='completed_trained_prototype_evaluation_targets_unmet',updated_unix=time.time(),
        managed_optimizer_calls=440,new_trainable_parameters=48,total_learned_parameters=40017,
        training_scenes=16,training_points=77998,independent_reserved_scenes=16,
        complete_scored_short_states=480,complete_scored_own_history_states=23040,
        own_history_steps=240,all_horizons=[1,2,3,4],own_history_cross_horizon_bytes_equal=horizon_equality,
        source_registrations_verified=checks,evidence=evidence,
        complete_block_costs=data['fresh_cost']['summaries'],
        independent_short_accuracy=data['fresh_blocks']['summaries'],
        independent_own240_accuracy=data['fresh_rollout']['summaries'],
        matched_application=app['summaries'],matched_application_accuracy=data['application_accuracy']['summaries'],
        target_maximum_ms_per_state=1.2,target_maximum_worst_error=1e-5,
        full_generator_targets_verified={str(k):False for k in (1,2,3,4)},
        original_completed_study_evidence_unchanged=True,higher_density_phase_started=False,phase600_started=False,
        priority='Complete this prototype before another architecture; distinguish practical value from target qualification')
    write(destinations[1],summary)
    lines=['# Trained recurrent sampler: complete prototype results','',
        'The prototype is trained and independently tested. It is a promising complete-state accelerator, but **the active 1.2 ms / 1e-5 targets remain unmet**. No higher-density or 600-step phase has started.','',
        'It predicts pressure, three staggered velocity fields and tracer. Learned query gates and mixing drive fixed field-sampling layers, followed by prescribed forcing and learned viscosity, flux, modal pressure and tied momentum. The 40,017 learned coefficients include 48 newly fitted double transport parameters. Candidate inference invokes no native numerical advance or analytical pressure inverse.','',
        'One managed CUDA fit used 400 Adam and 40 LBFGS calls on 77,998 samples from 16 original training scenes. A separate managed CPU Run restored every artifact exactly and reproduced the complete transport probe bitwise. The final normalized training loss was about 2.99e-24. The restored full four-step implementation passed 64 complete-state comparisons against its tensor graph.','',
        'All costs below use one CPU thread, batch one, core 23. The trained version uses the qualified integer-address compilation optimization; its neural computation is unchanged.','',
        '## Independent complete-block cost at 32 cubed','',
        '| Steps per pass | Median full block ms | Median ms/step | Maximum observed ms/step |','| --- | ---: | ---: | ---: |']
    for row in data['fresh_cost']['summaries']:
        if row['regime']=='32cubed':lines.append(f"| {row['horizons']} | {row['median_block_ms']:.4f} | {row['median_ms_per_generated_state']:.4f} | {row['maximum_ms_per_generated_state']:.4f} |")
    lines+=['','Construction and first warm calls are recorded separately. Complete warm calls include input validation, forcing, all neural stages, every intermediate output and internal state updates. This is an internally recurrent generator: increasing K gives little amortized speed improvement, and no parallel endpoint-decoding benefit is claimed.','',
        '## Independent physical accuracy','',
        'Sixteen new whole scenes were reserved before the fit. We froze the sole trained candidate and deployment sources before capturing their references; all existing shapes and densities of 8 and 16 cells per metre are represented. This confirms new scene parameters/seeds within supported shapes, not unseen-shape generalization.','',
        '| Interval | Domain group | Worst equation | Worst pressure | Worst velocity | Worst tracer |','| --- | --- | ---: | ---: | ---: | ---: |']
    for key,label in [('fresh_blocks','Short K1–4 blocks'),('fresh_rollout','240 own-history steps')]:
        for regime in ('small','32cubed'):
            selected=[row for row in data[key]['summaries']if row['regime']==regime]
            values={field:max(row['worst'][field]for row in selected)for field in ('equation','pressure','velocity','tracer')}
            lines.append('| '+label+' | '+regime+' | '+' | '.join(f'{value:.6g}'for value in values.values())+' |')
    lines+=['','Every intermediate field and actual own-history step was scored. All 320 fresh short-block states passed 1e-5. Through 240 steps, pressure/velocity/tracer stayed below 1e-5, while the equation criterion failed. That criterion compares predicted pressure with the RHS of the independently evolved native reference, so it includes accumulated history differences. It is not a measurement of the pressure branch in isolation.','',
        '## Matched complete application comparison','',
        'Both 32-cubed development scenes, both densities, steady/step 13 forcing-change scenarios, 48 accepted steps per episode, five balanced serial repeats. Four actual K generators and native PCG/spectral controls use the same CPU core and consumer interface. Setup is separate; timing includes model/solver work, state transfers, checks, event handling and final verification/close.','',
        '| Method | Scenario | Completed episodes | Median application ms/accepted step |','| --- | --- | ---: | ---: |']
    for row in app['summaries']:
        value=row['median_application_ms_per_accepted_step'];display='Unavailable'if value is None else f'{value:.4f}'
        lines.append(f"| {row['method']} | {row['scenario']} | {row['completed']}/{row['episodes']} | {display} |")
    lines+=['','Physical application outputs were retained outside the clocks; exact repeated bytes are checked and accuracy is evaluated afterward. Candidate accuracy covers every consumed state, including forcing changes. Native-control accuracy retains its diagnostic-checkpoint scope. The neural consumer performs structural uploads with zero numerical advances; this is not native physical-acceptance certification. Original numerical control gates remain unchanged.','',
        '## Disposition','',
        'Retain this trained prototype as a useful research result. It improves the earlier complete-generator baseline substantially. The absolute 1.2 ms budget and own-history equation criterion still need work; these remain explicit failures rather than a reason to call the model useless. The user-prioritized prototype is complete before any different architecture is started.','',
        '[Machine-readable results and exact evidence](recurrent-sampler-summary-01.json) · [Completion scope](recurrent-sampler-completion-02.md) · [Practical-value triage policy](prototype-triage-policy-03.json)','']
    destinations[0].write_text('\n'.join(lines))
    write(destinations[2],dict(status='active_phase1_trained_prototype_completed_targets_unmet',updated_unix=time.time(),
        previous_goal_turn_classification='progress',current_live_processes=[],new_managed_optimizer_calls=1240,
        complete=['Qualified integer-address execution; trained 48 transport parameters through managed CUDA; exact managed CPU restore',
                  'Every K1/2/3/4 complete short output and240-step own-history state scored on development and16fresh scenes',
                  'Fresh whole-scene confirmation and complete-block timings; matched native application controls and deferred forcing accuracy',
                  'Preserved failed setup attempts and all source/old-study evidence; practical-value triage recorded'],
        next=['Keep this trained prototype available and improve its remaining complete cost and amplified own-history equation drift',
              'Freeze new versions and independent reservations before any follow-up learning/selection',
              'Open higher-density or 600-step stages only after all reference targets pass'],
        full_generator_targets_verified={str(k):False for k in (1,2,3,4)},higher_density_phase_started=False,phase600_started=False,
        source_registrations_verified=checks,original_completed_study_evidence_unchanged=True,
        evidence=evidence,report=str(destinations[0].relative_to(ROOT))))
    print('Published completed trained prototype evaluation; wider target program remains active',flush=True)


if __name__=='__main__':run()
