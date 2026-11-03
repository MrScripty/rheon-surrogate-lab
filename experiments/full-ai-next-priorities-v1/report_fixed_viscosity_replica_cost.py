"""Publish every primary/added seed with its actual policy accuracy and CPU cost."""
import statistics
from application_model_evidence import require
from summarize_coupled import aggregate
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'fixed-viscosity-replication-cost-results.json'
    if output.exists():raise ValueError('Fresh primary/replica cost report required')
    directory=BASE/'fixed-viscosity-replication-timing-01'
    timing=read(directory/'summary.json');execution=read(directory/'execution.json')
    reg_path=DOCS/'fixed-viscosity-replication-timing-registration-01.json';reg=read(reg_path)
    registry_path=BASE/'fixed-viscosity-replication-deployments-01/qualified-models.json';registry=read(registry_path)
    if sha(registry_path)!=reg['model_registry_sha256']:raise ValueError('Frozen actual deployment registry changed')
    methods={m['identifier']:m for m in registry['methods']}
    variants=len(methods)*2+sum(m['model']['horizons']==1 for m in methods.values())*len(reg['bundled_horizons'])
    if timing['status']!='completed'or execution['status']!='completed'or len(execution['completed'])!=5*variants:
        raise ValueError('Complete finite primary/replica CPU population required')
    if len(execution['workers'])!=len(methods)or not all(w['reaped']and w['returncode']==0 for w in execution['workers']):
        raise ValueError('Every numerical worker must succeed and reap')
    if timing['timed_requests']!=5*variants*64 or timing['source_sha256']!=sha(HERE/'benchmark_fixed_viscosity_replicas.py')or timing['registration_sha256']!=sha(reg_path):
        raise ValueError('Actual CPU source/population differs')
    paths=[reg_path,registry_path,directory/'summary.json',directory/'execution.json',DOCS/'fixed-viscosity-replication-results.json',DOCS/'fixed-viscosity-replication-selection-01.json']
    for relative,digest in reg['accuracy_evidence_sha256'].items():
        path=ROOT/relative
        if sha(path)!=digest:raise ValueError('Actual bound accuracy/execution evidence changed')
        paths.append(path)
    for response in execution['completed']:
        path=ROOT/response['path'];raw=read(path)
        if sha(path)!=response['sha256']or len(raw['rows'])!=64 or raw['threads']!=1 or raw['affinity']!=[23]or raw['cuda_initialized']:
            raise ValueError('Raw timing population/context changed')
        paths.append(path)
    bank=BASE/'temporal-development-240-01/bank.json';scene_ids=[r['identifier']for r in read(bank)['trajectories']]
    evaluations={};preparation=[];per_grid=[]
    for identifier,method in methods.items():
        require(method,sha(bank),scene_ids)
        evaluations[identifier]=read(ROOT/method['accuracy_summary']['path'])
        ready_path=directory/(identifier+'-qualification.json');ready=read(ready_path);paths.append(ready_path)
        model=method['model'];identity=ready['model'];k=model['horizons']
        if ready['cases']!=64 or identity['weights_sha256']!=model['weights_sha256']or identity['application_policy']!=model['policy']or identity['replication_ordinal']!=method['replication_ordinal']:
            raise ValueError('Timed trained checkpoint/policy/seed differs from accuracy')
        raw=[read(directory/f'{identifier}-full-r{r}.json')for r in range(5)]
        preparation.append(dict(identifier=identifier,replication_ordinal=method['replication_ordinal'],preparation_after_imports_s=ready['setup_s'],
            peak_rss_bytes=max([ready['peak_rss_bytes']]+[r['peak_rss_bytes']for r in raw])))
        for grid in sorted({tuple(row['counts_xyz'])for r in raw for row in r['rows']}):
            values=[statistics.median(row['seconds']for row in r['rows']if tuple(row['counts_xyz'])==grid)*1000 for r in raw]
            per_grid.append(dict(identifier=identifier,counts_xyz=list(grid),median_block_ms=statistics.median(values),
                median_ms_per_generated_step=statistics.median(values)/k,min_repeat_median_ms=min(values),max_repeat_median_ms=max(values)))
    points=[];sequential=[];comparisons=[]
    for row in timing['rows']:
        method=methods[row['identifier']];ordinal=method['replication_ordinal']
        if row['variant'].startswith('sequential'):
            sequential.append(dict(row,replication_ordinal=ordinal,accuracy_claim=False,
                scope='K complete B1calls from one reference start then own predictions; no future reference inputs'))
            continue
        if row['variant']!='full':continue
        k=row['horizons'];evaluation=evaluations[row['identifier']]
        for index,interval in enumerate(('single_block','rollout_12','rollout_48','rollout_240')):
            errors=aggregate(evaluation,row['regime'],index)
            points.append(dict(track='B',identifier=row['identifier'],horizons=k,replication_ordinal=ordinal,
                policy=method['model']['policy'],regime=row['regime'],interval=interval,
                median_ms=row['median_ms_per_generated_step'],full_block_ms=row['median_block_ms'],
                worst_error=max(errors['worst'].values())if errors['complete']else None,field_worst=errors['worst'],
                all_finite_complete=errors['complete'],accuracy_pass=errors['accuracy_pass'],control_only=k==1,
                target_met=k>1 and errors['accuracy_pass']and row['median_block_ms']<2*k,
                min_repeat_ms=row['min_repeat_median_ms']/k,max_repeat_ms=row['max_repeat_median_ms']/k,
                p95_full_block_ms=row['p95_block_ms'],maximum_full_block_ms=row['maximum_block_ms'],
                physical_output_equivalence_inherited=method['model']['policy']=='fast-float64',
                timing_scope='Complete blocks on64reference-history windows; application/event throughput separate',
                accuracy_scope='Actual every equation/P/V/T output and own-history interval on8development scenes'))
    for control in sequential:
        row=next(r for r in timing['rows']if r['variant']=='full'and r['horizons']==control['horizons']and r['regime']==control['regime']
                 and methods[r['identifier']]['replication_ordinal']==control['replication_ordinal'])
        comparisons.append(dict(horizons=row['horizons'],replication_ordinal=control['replication_ordinal'],regime=row['regime'],
            bundled_block_ms=row['median_block_ms'],sequential_B1_block_ms=control['median_block_ms'],
            bundled_speedup=control['median_block_ms']/row['median_block_ms'],
            scope='Matched physical outputs and seed ordinal, common frozen components and36000training targets; FLOPs differ'))
    seed_summary=[]
    for k in reg['bundled_horizons']:
        rows=[p for p in points if p['horizons']==k and p['regime']=='small'and p['interval']=='rollout_240']
        if {r['replication_ordinal']for r in rows}!={1,2,3}:raise ValueError('All three seeds required without reselection')
        seed_summary.append(dict(horizons=k,all_three_accuracy_pass=all(r['accuracy_pass']for r in rows),
            all_three_combined_median_pass=all(r['target_met']for r in rows),seeds=rows))
    write(output,dict(status='completed',points=points,timing=timing['rows'],per_grid=per_grid,
        worker_preparation=preparation,sequential_controls=sequential,joint_vs_sequential=comparisons,seed_summary=seed_summary,
        source_evidence={str(p.relative_to(ROOT)):sha(p)for p in paths},reporter_sha256=sha(__file__),
        CPU_latency_claim=True,sustained_application_throughput_claim=False,primary_seed_reselected=False,
        shared_component_pretraining=True,independent_component_pretraining=False,all_added_seeds_retained=True,
        reserved_scenes_consumed=False,original_goal_complete=False))
    lines=['# Selected temporal recipes: three seeds and actual CPU costs','',
        'All added seeds are retained. Each policy has its own complete accuracy or exact execution proof. '
        'Five balanced serial repeats use one CPU thread, core23 and batch one, with complete/tensor scopes '
        'and matched sequential B1controls. Original primary seed remains fixed.','',
        '| Model | Seed ordinal | Block ms | ms/state | Worst own240 error | Combined median pass |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for p in points:
        if p['regime']!='small'or p['interval']!='rollout_240':continue
        error=f'{p["worst_error"]:.7g}'if p['worst_error']is not None else'Incomplete'
        lines.append(f'| {p["identifier"]} | {p["replication_ordinal"]} | {p["full_block_ms"]:.4f} | {p["median_ms"]:.4f} | {error} | {p["target_met"]if p["horizons"]>1 else"Control"} |')
    lines.extend(['','These are warm mixed-small-grid medians. Per-grid/tails,32³, setup/RSS, reserved generalization '
        'and sustained application/event throughput are separate. Seeds vary the temporal cell and sampling, '
        'with shared frozen component pretraining.','',
        '[Every seed, raw costs, accuracy and sources](fixed-viscosity-replication-cost-results.json)',''])
    (DOCS/'fixed-viscosity-replication-cost-results.md').write_text('\n'.join(lines))
    print('Published every primary/added seed with actual CPU cost and matched sequential controls',flush=True)


if __name__=='__main__':run()
