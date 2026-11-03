"""Audit matched continuations and their complete every-horizon/own240 results."""
from pathlib import Path
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha
from summarize_coupled import aggregate


def run():
    output=DOCS/'temporal-provisional-results.json'
    if output.exists():raise ValueError('Fresh completed continuation report required')
    models=['b2-provisional-control','b2-provisional-aux'];pairs=[BASE/(m+'-pair-01')for m in models]
    receipts=[read(p/'receipt.json')for p in pairs]
    projects=[ROOT/r['source_project']for r in receipts];assemblies=[read(p/'assembly.json')for p in projects]
    configs=[a['config']for a in assemblies]
    trained=[read(p/'train/receipt.json')for p in pairs];restored=[read(p/'restore/receipt.json')for p in pairs]
    histories=[read(p/'restore/history.json')for p in pairs];registration=read(DOCS/'temporal-provisional-registration-01.json')
    checks=dict(completed_pairs=all(r['status']=='completed'for r in receipts),
        actual4000updates=all(r['optimizer_updates']==4000 for r in trained),
        zero_update_CPU_restores=all(r['optimizer_updates']==0 and r['full_cpu_probe_bitwise_verified']and not r['execution']['cuda_initialized']for r in restored),
        exact_artifact_parity=all(all(r['artifact_parity'].values())for r in receipts),
        same_data=assemblies[0]['static_inputs']==assemblies[1]['static_inputs'],
        same_initial_inventory=histories[0]['initial_inventory']==histories[1]['initial_inventory'],
        initialized_from_registered_parent=all(h['initial_inventory']==c['continuation_inventory']for h,c in zip(histories,configs)),
        captured_parent_checkpoint_exact=all(sha(p/'continuation-weights.npz')==registration['continuation_weights_sha256']for p in projects),
        same_sample_stream=histories[0]['sample_order_sha256']==histories[1]['sample_order_sha256'],
        same_counted_state_work=histories[0]['optimizer_work']==histories[1]['optimizer_work'],
        only_identifier_and_auxiliary_weight_differ={k:v for k,v in configs[0].items()if k not in('identifier','auxiliary_provisional_weight')}=={k:v for k,v in configs[1].items()if k not in('identifier','auxiliary_provisional_weight')},
        auxiliary_weights_exact=[c['auxiliary_provisional_weight']for c in configs]==[0.,1.],
        same_inference_source=sha(projects[0]/'bundle_model_coupled.py')==sha(projects[1]/'bundle_model_coupled.py'),
        same_training_code={Path(s['path']).name:s['sha256']for s in assemblies[0]['sources']if s['path'].endswith('.py')}=={Path(s['path']).name:s['sha256']for s in assemblies[1]['sources']if s['path'].endswith('.py')},
        actual_two_block_counts=all(h['optimizer_work']['network_forwards']==8000 and h['optimizer_work']['encoder_calls']==8000
            and h['optimizer_work']['predicted_physical_states']==16000 and h['optimizer_work']['pressure_readout_calls']==16000 for h in histories))
    for index,(p,a,r,h)in enumerate(zip(projects,assemblies,receipts,histories)):
        checks[f'assembly_binding_{index}']=sha(p/'assembly.json')==r['source_assembly_sha256']
        checks[f'config_registration_{index}']=sha(p/'config.json')==registration['config_sha256']['control'if index==0 else'aux']
        checks[f'captured_sources_{index}']=all(sha(ROOT/s['path'])==s['sha256']for s in a['sources'])
        checks[f'frozen_coefficients_{index}']=all(h['initial_inventory'][name]==h['final_inventory'][name]==value for name,value in trained[index]['frozen_parameter_inventory'].items())
        checks[f'exact_source_closure_{index}']=trained[index]['closure']==restored[index]['closure']
    if not all(checks.values()):
        write(BASE/'temporal-provisional-comparison-audit-failure-01.json',dict(status='failed',checks=checks,source_sha256=sha(__file__)))
        raise ValueError('Matched continuation controls failed')
    paths=[BASE/'temporal-provisional-evaluation-01'/m/'summary.json'for m in models]
    baseline_path=BASE/'temporal-coupled-evaluation-01/b2-coupled/summary.json'
    evaluations=[read(baseline_path),*[read(p)for p in paths]]
    if any(v['status']!='completed'or len(v['scenes'])!=8 for v in evaluations):raise ValueError('Every development scene required')
    if len({v['bank_sha256']for v in evaluations})!=1 or len({v['metrics_sha256']for v in evaluations})!=1:
        raise ValueError('Comparison populations or error criteria differ')
    comparisons=[]
    for regime in ('small','32cubed'):
        for index,interval in enumerate(('single_block','own_history12','own_history48','own_history240')):
            values=[aggregate(v,regime,index)for v in evaluations]
            comparisons.append(dict(regime=regime,interval=interval,parent=values[0],control=values[1],auxiliary=values[2]))
    diagnostics=[BASE/f'temporal-provisional-fitting-{arm}-01.json'for arm in ('control','aux')]
    fitting=[read(p)for p in diagnostics]
    if any(d['status']!='completed'or d['optimizer_updates']!=0 or d['reserved_scenes_consumed']for d in fitting):raise ValueError('Unchanged-model fitting diagnostics required')
    finalists=[models[index]for index,key in enumerate(('control','auxiliary'))if all(v[key]['accuracy_pass']for v in comparisons if v['regime']=='small')]
    source_paths=[*paths,baseline_path,*diagnostics,DOCS/'temporal-provisional-registration-01.json',DOCS/'temporal-provisional-admission-01.json']
    for pair,project in zip(pairs,projects):source_paths.extend([pair/'receipt.json',pair/'train/receipt.json',pair/'restore/receipt.json',pair/'restore/history.json',project/'assembly.json'])
    evidence={str(p.relative_to(ROOT)):sha(p)for p in source_paths}
    write(output,dict(status='completed',comparisons=comparisons,control_audit=checks,
        fitting={arm:value['summaries']for arm,value in zip(('control','auxiliary'),fitting)},
        accuracy_qualified_small_grid_recipes=finalists,CPU_latency_claim=False,inference_source_unchanged=True,
        parent_updates=10000,additional_updates_per_arm=4000,independent_pretraining_claim=False,
        source_evidence=evidence,source_sha256=sha(__file__),reserved_scenes_consumed=False,model_promotion=False,original_goal_complete=False,
        next_decision='Measure complete CPU blocks and qualify requested horizon extensions'if finalists else'Inspect actual continuation outcomes and qualified tracer-representation controls before a new full-state comparison'))
    fields=('pressure','equation','velocity','tracer');lines=['# Matched provisional-supervision continuations','',
        'Both continuations complete 4,000 additional updates from the same 10,000-update checkpoint and exact zero-update CPU restores. '
        'The data, sample stream, initial tensors, optimizer schedule, two-block feedback, inference architecture and target work match. '
        'Only the training-only auxiliary weight differs.','',
        '| Regime | Interval | Arm | Complete | Worst P / equation / velocity / tracer | Accuracy pass |',
        '| --- | --- | --- | --- | --- | --- |']
    for v in comparisons:
        for arm in ('control','auxiliary'):
            a=v[arm];error=' / '.join(f'{a["worst"][f]:.7g}'for f in fields)if a['complete']else'Unavailable; incomplete interval'
            lines.append(f'| {v["regime"]} | {v["interval"]} | {arm} | {a["complete"]} | {error} | {a["accuracy_pass"]} |')
    lines.extend(['','Every intermediate output and all eight development scenes are retained, including complete 12/48/240-step or explicitly incomplete own-history outcomes. '
        'Finite-prefix maxima cannot establish full-interval accuracy. No new CPU timing is claimed; the eight reserved scenes remain unopened.','',
        '[Source-bound results and fitting diagnostics](temporal-provisional-results.json) · '
        '[Frozen comparison](temporal-provisional-registration-01.json) · [Transport diagnosis](temporal-coupled-transport-results.md)',''])
    (DOCS/'temporal-provisional-results.md').write_text('\n'.join(lines))
    print('Completed matched continuation audit; accuracy-qualified recipes:',finalists,flush=True)


if __name__=='__main__':run()
