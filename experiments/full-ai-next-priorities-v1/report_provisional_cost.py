"""Join actual CPU model costs to unchanged full-state accuracy populations."""
import statistics
from summarize_coupled import aggregate
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'temporal-provisional-cost-results.json'
    if output.exists():raise ValueError('Fresh completed CPU cost report required')
    directory=BASE/'temporal-provisional-timing-02';timing_path=directory/'summary.json';timing=read(timing_path)
    execution=read(directory/'execution.json');batch=read(BASE/'temporal-provisional-timing-batch-02/execution.json')
    if timing['status']!='completed'or execution['status']!='completed'or batch['status']!='completed':raise ValueError('Complete serial benchmark and parent batch required')
    if len(execution['completed'])!=30 or not all(w['reaped']and w['returncode']==0 for w in execution['workers'])or not batch['owned_child_reaped']:
        raise ValueError('Every timing response and terminal worker required')
    if timing['timed_requests']!=1920 or not timing['balanced_serial']or timing['threads']!=1:raise ValueError('Registered timing scope differs')
    if timing['source_sha256']!=sha(HERE/'benchmark_provisional_v2.py')or timing['registration_sha256']!=sha(DOCS/'temporal-provisional-timing-registration-02.json'):
        raise ValueError('Timing source or registration changed')
    failed=read(BASE/'temporal-provisional-timing-01/execution.json')
    if failed['completed']or not all(w['reaped']for w in failed['workers']):raise ValueError('Failed setup was not empty/reaped')
    models=['b2-coupled','b2-provisional-control','b2-provisional-aux'];accuracy={};source_paths=[];preparation=[];per_grid=[]
    for model in models:
        path=BASE/('temporal-coupled-evaluation-01'if model=='b2-coupled'else'temporal-provisional-evaluation-01')/model/'summary.json'
        value=read(path)
        if value['status']!='completed'or len(value['scenes'])!=8 or value['bank_sha256']!=timing['bank_sha256']:raise ValueError('Exact model/evaluation population required')
        accuracy[model]=value;source_paths.append(path)
        ready_path=directory/('b'+model+'-qualification.json');ready=read(ready_path)
        weights=BASE/(model+'-pair-01')/'restore/weights.npz'
        if ready['kind']!='ready'or ready['cases']!=64 or ready['model']['weights_sha256']!=sha(weights):raise ValueError('Timed weights/readiness binding differs')
        preparation.append(dict(identifier=model,benchmark_preparation_after_imports_s=ready['setup_s'],benchmark_peak_rss_bytes=ready['peak_rss_bytes']))
        source_paths.append(ready_path)
        results=[]
        for repeat in range(5):
            path=directory/('b'+model+f'-full-r{repeat}.json');raw=read(path);results.append(raw);source_paths.append(path)
            if raw['identifier']!=model or raw['horizons']!=2 or raw['threads']!=1 or raw['cuda_initialized']or raw['affinity']!=[23]:
                raise ValueError('Actual complete-block CPU context differs')
            preparation[-1]['benchmark_peak_rss_bytes']=max(preparation[-1]['benchmark_peak_rss_bytes'],raw['peak_rss_bytes'])
        grids={tuple(r['counts_xyz'])for result in results for r in result['rows']}
        for grid in sorted(grids):
            values=[statistics.median(r['seconds']for r in result['rows']if tuple(r['counts_xyz'])==grid)*1000 for result in results]
            per_grid.append(dict(identifier=model,counts_xyz=list(grid),median_block_ms=statistics.median(values),
                median_ms_per_generated_step=statistics.median(values)/2,min_repeat_median_ms=min(values),max_repeat_median_ms=max(values)))
    points=[]
    for row in timing['rows']:
        if row['variant']!='full':continue
        for index,label in enumerate(('single_block','rollout_12','rollout_48','rollout_240')):
            errors=aggregate(accuracy[row['identifier']],row['regime'],index)
            if not errors['complete']:raise ValueError('Incomplete measured model must retain explicit unavailable worst error')
            points.append(dict(track='B',identifier=row['identifier'],horizons=2,regime=row['regime'],interval=label,
                median_ms=row['median_ms_per_generated_step'],full_block_ms=row['median_block_ms'],worst_error=max(errors['worst'].values()),
                field_worst=errors['worst'],accuracy_pass=errors['accuracy_pass'],target_met=errors['accuracy_pass']and row['median_block_ms']<4,
                min_repeat_ms=row['min_repeat_median_ms']/2,max_repeat_ms=row['max_repeat_median_ms']/2,
                p95_full_block_ms=row['p95_block_ms'],maximum_full_block_ms=row['maximum_block_ms'],
                timing_scope='Complete model blocks from64fixed reference-history windows; not an own-history episode wall time',
                accuracy_scope='Every declared equation/P/V/T output across all8development scenes and matching interval'))
    source_paths.extend([timing_path,directory/'execution.json',BASE/'temporal-provisional-timing-batch-02/execution.json',
        BASE/'temporal-provisional-timing-01/execution.json',BASE/'temporal-provisional-timing-batch-01/execution.json',
        DOCS/'temporal-provisional-timing-registration-02.json',DOCS/'temporal-provisional-results.json'])
    for record in execution['completed']:
        path=ROOT/record['path']
        if sha(path)!=record['sha256']:raise ValueError('Retained raw timing response changed')
        source_paths.append(path)
    evidence={str(p.relative_to(ROOT)):sha(p)for p in source_paths}
    write(output,dict(status='completed',timing=timing['rows'],points=points,per_grid=per_grid,worker_preparation=preparation,
        source_evidence=evidence,source_sha256=sha(__file__),CPU_latency_claim=True,
        full_state_goal_achieved=False,sustained_application_throughput_claim=False,original_goal_complete=False,
        no_model_or_evaluation_repeated_for_logging_recovery=True,reserved_scenes_consumed=False,
        cold_scope='Worker preparation after imports includes model/maps, data/case packing and contract warmup; peakRSS includes benchmark cases',
        warm_scope='Physical current-state encoding, forcing, complete two-state prediction and physical field reconstruction; tensor scope separate'))
    lines=['# Coupled-model CPU cost and accuracy','',
        'Five balanced serial repeats measure actual restored weights, one CPU thread on core23 and batch size one. '
        'Each worker times 48 small-grid and 16 separate32³ requests per scope. '
        'Accuracy uses every output horizon and every original field on all eight development scenes.','',
        '| Model | Small-grid full block ms | Per generated step ms | Block p95 ms | Worst single-block equation/field error | Combined target |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for model in models:
        point=next(p for p in points if p['identifier']==model and p['regime']=='small'and p['interval']=='single_block')
        lines.append(f'| {model} | {point["full_block_ms"]:.4f} | {point["median_ms"]:.4f} | {point["p95_full_block_ms"]:.4f} | {point["worst_error"]:.7g} | {point["target_met"]} |')
    lines.extend(['','The median spans heterogeneous small grids and is not a per-grid or worst-call guarantee. '
        'These costs quantify architecture headroom; accuracy failures remain failures. '
        'The same reference-history block costs accompany longer-interval error rows as a model-cost comparison, not a measured own-history/application throughput result.','',
        '| Model | Grid XYZ | Full block ms | Per generated step ms |','| --- | --- | ---: | ---: |'])
    for r in per_grid:lines.append(f'| {r["identifier"]} | {"×".join(map(str,r["counts_xyz"]))} | {r["median_block_ms"]:.4f} | {r["median_ms_per_generated_step"]:.4f} |')
    lines.extend(['','| Worker | Preparation after imports s | Peak benchmark RSS GiB |','| --- | ---: | ---: |'])
    for r in preparation:lines.append(f'| {r["identifier"]} | {r["benchmark_preparation_after_imports_s"]:.3f} | {r["benchmark_peak_rss_bytes"]/1024**3:.3f} |')
    lines.extend(['','Preparation includes model loading, dense-map construction, case preparation and contract warmup; RSS includes the benchmark cases. '
        'Dense maps remain unsuitable for production-scale domains. '
        'The first attempt recorded zero repeats and reaped its worker after constructor stdout disrupted JSON replies. '
        'The fresh wrapper changes logging only; all fits, accuracy evaluations, prediction math, populations and scopes remain unchanged.','',
        '[Complete timing, accuracy and source evidence](temporal-provisional-cost-results.json) · '
        '[Matched fit outcomes](temporal-provisional-results.md) · [Original goal](plan.md)',''])
    (DOCS/'temporal-provisional-cost-results.md').write_text('\n'.join(lines))
    print('Published measured complete-block CPU costs; accuracy and full-state goal remain separate',flush=True)


if __name__=='__main__':run()
