"""Compare controlled objective followup without substituting mean error for worst."""
from study_support import ROOT, HERE, BASE, DOCS, read, write, sha


if __name__=='__main__':
    output=DOCS/'temporal-global-results.json'
    if output.exists():
        raise ValueError('Fresh completed comparison required')
    audit=BASE/'temporal-global-comparison-audit-01.json'
    if read(audit)['status']!='passed':
        raise ValueError('Comparison controls must verify before interpretation')
    paths=[BASE/'temporal-development-evaluation-01/b1/summary.json',BASE/'temporal-global-development-evaluation-01/b1/summary.json']
    evaluations=[read(path)for path in paths]
    comparisons=[]
    for regime in ('small','32cubed'):
        for index,interval in enumerate(('single_block','own_history12','own_history48','own_history240')):
            values=[]
            for evidence in evaluations:
                scenes=[scene for scene in evidence['scenes']if(max(scene['counts_xyz'])<=16)==(regime=='small')]
                values.append(dict(worst={name:max(scene['summaries'][index]['worst'][name]for scene in scenes)
                    for name in ('pressure','equation','velocity','tracer')},
                    all_accuracy_criteria_pass=all(scene['summaries'][index]['all_accuracy_criteria_pass']for scene in scenes)))
            comparisons.append(dict(regime=regime,interval=interval,original_log_loss=values[0],shared_global=values[1]))
    diagnosis_paths=[BASE/'temporal-accuracy-diagnosis-b1-01.json',BASE/'temporal-global-fitting-diagnosis-b1-01.json']
    diagnoses=[read(path)for path in diagnosis_paths]
    fitting=[]
    for original,candidate in zip(diagnoses[0]['summaries'],diagnoses[1]['summaries'],strict=True):
        keys=('split','phase','field','outputs')
        if any(original[key]!=candidate[key]for key in keys):
            raise ValueError('Fitting diagnosis populations differ')
        fitting.append(dict(**{key:original[key]for key in keys},
            original_median=original['learned_median'],original_worst=original['learned_worst'],
            global_median=candidate['learned_median'],global_worst=candidate['learned_worst']))
    passed=all(row['shared_global']['all_accuracy_criteria_pass']for row in comparisons)
    result=dict(status='completed',comparisons=comparisons,fitting=fitting,all_accuracy_targets_pass=passed,
        controlled_comparison_audit_sha256=sha(audit),source_sha256=sha(__file__),
        source_evidence={str(path.relative_to(ROOT)):sha(path)for path in [*paths,*diagnosis_paths]},
        reserved_scenes_consumed=False,optimizer_updates=0,new_CPU_latency_claim=False,
        next_decision='Qualify bundled extensions and further seeds'if passed else'Accuracy still insufficient; assess fitting and startup coverage before bundled replications',
        original_goal_complete=False,model_promotion=False)
    write(output,result)
    lines=['# Shared-global temporal capacity comparison','',
        'The same data, sampled work, objective and training loop were verified. Capacity changed; the initial prediction function matches the original persistence baseline.',
        'All values below are worst relative errors across development scenes and output steps. No reserved scenes were consumed.','',
        '| Regime | Interval | Field | Original log loss | Shared global |', '| --- | --- | --- | ---: | ---: |']
    for row in comparisons:
        for field in ('pressure','equation','velocity','tracer'):
            lines.append(f"| {row['regime']} | {row['interval']} | {field} | {row['original_log_loss']['worst'][field]:.6g} | {row['shared_global']['worst'][field]:.6g} |")
    lines.extend(['','## Fixed-window fitting diagnosis','',
        'Four predeclared starting times per scene; this is a fitting diagnostic, not the all-window criterion.','',
        '| Split | Phase | Field | Original median | Global median | Original worst | Global worst |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |'])
    for row in fitting:
        lines.append(f"| {row['split']} | {row['phase']} | {row['field']} | {row['original_median']:.6g} | {row['global_median']:.6g} | {row['original_worst']:.6g} | {row['global_worst']:.6g} |")
    lines.extend(['',result['next_decision']+'.',
        'Untrained feasibility profiles are not trained-model timings. No new latency target claim is inferred for this one-step diagnostic.',
        '', '[Comparison controls](../../../runs/full-ai-next-priorities-v1/temporal-global-comparison-audit-01.json) · [All evidence hashes](temporal-global-results.json)',''])
    (DOCS/'temporal-global-results.md').write_text('\n'.join(lines))
    print('Completed shared-global capacity comparison; all accuracy targets pass:',passed,flush=True)
