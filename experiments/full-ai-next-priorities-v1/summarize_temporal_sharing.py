"""Publish every sharing model's full-state accuracy and complete-block timing."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


if __name__=='__main__':
    output=DOCS/'temporal-sharing-results.json'
    if output.exists():raise ValueError('Fresh completed sharing comparison required')
    audit=BASE/'temporal-sharing-comparison-audit-01.json'
    if read(audit)['status']!='passed':raise ValueError('Registered controls must pass before interpretation')
    timing=BASE/'temporal-sharing-timing-01/summary.json';timed=read(timing);rows=[];sources={}
    for k in (2,3,4):
        for architecture in ('endpoint','reuse'):
            identifier=f'b{k}-{architecture}'
            path=BASE/'temporal-sharing-development-evaluation-01'/identifier/'summary.json'
            evaluation=read(path);sources[str(path.relative_to(ROOT))]=sha(path)
            diagnostic=BASE/f'temporal-sharing-fitting-diagnosis-{identifier}-01.json'
            sources[str(diagnostic.relative_to(ROOT))]=sha(diagnostic)
            for regime in ('small','32cubed'):
                scenes=[scene for scene in evaluation['scenes']if(max(scene['counts_xyz'])<=16)==(regime=='small')]
                measured=next(r for r in timed['rows']if r['identifier']==identifier and r['variant']=='full'and r['regime']==regime)
                for index,interval in enumerate(('single_block','own_history12','own_history48','own_history240')):
                    worst={name:max(scene['summaries'][index]['worst'][name]for scene in scenes)
                           for name in ('pressure','equation','velocity','tracer')}
                    passed=all(scene['summaries'][index]['all_accuracy_criteria_pass']for scene in scenes)
                    rows.append(dict(identifier=identifier,architecture=architecture,horizons=k,regime=regime,
                        interval=interval,worst=worst,worst_error=max(worst.values()),accuracy_pass=passed,
                        median_full_block_ms=measured['median_block_ms'],
                        median_ms_per_generated_step=measured['median_ms_per_generated_step'],
                        timing_pass=measured['median_block_ms']<2*k,
                        combined_target_pass=regime=='small'and passed and measured['median_block_ms']<2*k))
    finalists=[identifier for identifier in sorted(set(row['identifier']for row in rows))if
               all(row['combined_target_pass']for row in rows if row['identifier']==identifier and row['regime']=='small')]
    write(output,dict(status='completed',rows=rows,small_grid_finalists=finalists,
        all_requested_K_targets_pass=all(any(r['horizons']==k and r['identifier']in finalists for r in rows)for k in (2,3,4)),
        registration_sha256=sha(DOCS/'temporal-sharing-registration-01.json'),audit_sha256=sha(audit),
        timing_sha256=sha(timing),source_evidence=sources,source_sha256=sha(__file__),
        reserved_scenes_consumed=False,application_model_promoted=False,
        next_decision='Replicate qualifying recipes before reserved evaluation'if finalists else
        'No accuracy-qualified sharing finalist; use retained fitting/startup diagnostics before another bounded architecture/data change',
        original_goal_complete=False))
    lines=['# Endpoint and retained-feature full-state comparison','',
        'Both architectures run the spatial encoder once per block. The endpoint model feeds its predicted endpoint to a small intermediate decoder. '
        'The reuse model retains first-prediction features through cheap latent updates. Every intermediate pressure, velocity and tracer state is supervised and evaluated. '
        'All six managed fits and exact CPU restores are complete; controls verify the common data, sampling, objective, feedback curriculum and paired encoder initialization. '
        'Parameter counts and arithmetic work differ, so the screen is not FLOP matched.','',
        'Complete-block timings include physical encoding, prescribed forcing, all internal updates and reconstruction of every output. '
        'K2/3/4 budgets are below4/6/8ms with no separate internal-stage cap. Worst errors below include all scenes and every physical horizon.','',
        '| Model | Grids | Interval | Block ms | ms / step | Pressure | Equation | Velocity | Tracer | Combined pass |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        fields=' | '.join(f"{row['worst'][name]:.6g}"for name in ('pressure','equation','velocity','tracer'))
        lines.append(f"| {row['identifier']} | {row['regime']} | {row['interval']} | {row['median_full_block_ms']:.3f} | {row['median_ms_per_generated_step']:.3f} | {fields} | {row['combined_target_pass']} |")
    lines.extend(['',read(output)['next_decision']+'.',
        'No reserved scenes were consumed. Single-block accuracy and240-step own-history are separate requirements; latency alone does not qualify a model. '
        'Cold setup, per-repeat spread, tensor-only timing, peak memory and sequential one-step controls are retained in the timing evidence.',
        '', '[Results and hashes](temporal-sharing-results.json) · [Control audit](../../../runs/full-ai-next-priorities-v1/temporal-sharing-comparison-audit-01.json) · [Timing evidence](../../../runs/full-ai-next-priorities-v1/temporal-sharing-timing-01/summary.json)',''])
    (DOCS/'temporal-sharing-results.md').write_text('\n'.join(lines))
    print('Published all six sharing comparisons; qualifying small-grid recipes:',finalists,flush=True)
