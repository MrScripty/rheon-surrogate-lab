"""Publish offline error separation and CPU profile without candidate promotion."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'displacement-component-results.json'
    if output.exists():raise ValueError('Fresh component report required')
    raw_path=BASE/'displacement-component-diagnosis-01.json';raw=read(raw_path)
    protocol_path=DOCS/'displacement-component-diagnostic-protocol-02.json';protocol=read(protocol_path)
    qualification_path=BASE/'viscosity-teacher-qualification-02.json';qualified=read(qualification_path)
    if raw['status']!='completed'or len(raw['rows'])!=640 or len(raw['replays'])!=8:
        raise ValueError('All registered train-only controls required')
    if not raw['normal_metrics_bitwise_equal_to_prior']or not raw['candidate_coefficients_unchanged']or raw['optimizer_updates']!=0:
        raise ValueError('Exact unchanged-candidate control required')
    if qualified['status']!='passed'or len(qualified['rows'])!=56 or raw['native_qualification_sha256']!=sha(qualification_path):
        raise ValueError('Complete independent native viscosity qualification required')
    if raw['protocol_sha256']!=sha(protocol_path)or raw['source_sha256']!=protocol['source_sha256']:
        raise ValueError('Registered diagnostic changed')
    for name,digest in raw['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Qualified diagnostic source changed')
    for name,digest in raw['source_evidence'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Native training evidence changed')
    profile_path=BASE/'displacement-CPU-stage-profile-01.json';profile=read(profile_path)
    if profile['status']!='completed_diagnostic_only'or not profile['predictions_unchanged']or not profile['weights_unchanged']:
        raise ValueError('Source-equivalent actual-model profile required')
    for name,digest in profile['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Profile source changed')
    paths=[raw_path,protocol_path,qualification_path,BASE/'viscosity-teacher-build-02/build-receipt.json',
        profile_path,DOCS/'displacement-CPU-stage-profile-protocol-01.json',
        BASE/'displacement-component-diagnostic-batch-01/execution.json',BASE/'displacement-component-diagnostic-batch-02/execution.json',
        BASE/'viscosity-teacher-build-01/build.log',BASE/'displacement-CPU-stage-profile-batch-01/execution.json',
        DOCS/'temporal-displacement-results.json',DOCS/'temporal-displacement-cost-results.json']
    for name in ('displacement-component-diagnostic-batch-02','displacement-CPU-stage-profile-batch-01'):
        batch=read(BASE/name/'execution.json')
        if batch['status']!='completed'or not batch['owned_child_reaped']:raise ValueError('Numerical controllers must complete and reap')
    write(output,dict(status='completed_offline_diagnosis',summaries=raw['summaries'],
        training_replay_128=[{k:v for k,v in r.items()if k!='rows'}for r in raw['replays']],
        CPU_stage_summaries=profile['stage_summaries'],source_evidence={str(p.relative_to(ROOT)):sha(p)for p in paths},
        reporter_sha256=sha(__file__),optimizer_updates=0,candidate_weights_unchanged=True,
        native_controls_excluded_from_inference_and_frontiers=True,full_state_240step_accuracy_claim=False,
        new_CPU_latency_claim=False,reserved_scenes_consumed=False,original_goal_complete=False,
        next_decisions=['Calibrate the neural viscosity layer against native-qualified exact labels in an ordinary managed fit and restore',
            'Qualify a fused implementation of the fixed learned-displacement readout, then measure complete model latency']))
    lines=['# Displacement error separation and CPU profile','',
        'The learned viscosity head is the main remaining fluid-error source in these offline controls. '
        'The normal model reproduces its previous fixed-window metrics bitwise, and all candidate coefficients remain unchanged. '
        'Controls use 16 training scenes and four starts; the two autonomous replays use 128 steps on training scenes. '
        'They do not establish the 240-step goal or a new candidate.','',
        '| Control | Warm worst P | Equation | Velocity | Tracer |','| --- | ---: | ---: | ---: | ---: |']
    for mode in protocol['modes']:
        values={r['field']:r['worst']for r in raw['summaries']if r['phase']=='warm'and r['mode']==mode}
        lines.append('| '+mode+' | '+' | '.join(f'{values[n]:.7g}'for n in ('pressure','equation','velocity','tracer'))+' |')
    lines.extend(['','| Training replay | Control | Worst equation through 128 | Worst velocity | Worst tracer |',
        '| --- | --- | ---: | ---: | ---: |'])
    for row in raw['replays']:
        lines.append(f'| {row["identifier"]} | {row["mode"]} | {row["worst"]["equation"]:.7g} | {row["worst"]["velocity"]:.7g} | {row["worst"]["tracer"]:.7g} |')
    lines.extend(['','The native viscosity primitive and exact dyadic label path pass 56 cases. '
        'The next neural fit can learn viscosity coefficients from those labels without introducing a numerical viscosity routine into candidate inference. '
        'The first helper build failed on a crate-private axis constant before any numeric query; the preserved second version uses the identical public axis enumeration.','',
        'The separate CPU profile leaves predictions and weights unchanged. Inclusive small-grid block medians include:','',
        '| Stage | Inclusive block ms |','| --- | ---: |'])
    for row in profile['stage_summaries']:
        if row['regime']=='small':lines.append(f'| {row["module"]} | {row["median_inclusive_block_ms"]:.4f} |')
    lines.extend(['','Inclusive stages overlap and are not additive. Instrumented timings are not frontier measurements. '
        'The fixed field readout is the largest measured local stage and supports testing a fused implementation with unchanged learned weights. '
        'Actual full-block benchmark remains 6.35 ms, with accuracy failures retained.','',
        '[Raw controls, profile and hashes](displacement-component-results.json) · '
        '[Full-state accuracy](temporal-displacement-results.md) · [Actual CPU costs](temporal-displacement-cost-results.md)',''])
    (DOCS/'displacement-component-results.md').write_text('\n'.join(lines))
    print('Published offline viscosity/transport separation and actual CPU stage profile; no promotion',flush=True)


if __name__=='__main__':run()
