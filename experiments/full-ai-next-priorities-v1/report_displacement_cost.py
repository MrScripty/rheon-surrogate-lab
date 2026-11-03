"""Join restored full-state accuracy to measured complete-block CPU costs."""
import statistics
from summarize_coupled import aggregate
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'temporal-displacement-cost-results.json'
    if output.exists():raise ValueError('Fresh displacement CPU cost result required')
    directory=BASE/'temporal-displacement-timing-01';timing=read(directory/'summary.json');execution=read(directory/'execution.json')
    registration_path=DOCS/'temporal-displacement-timing-registration-01.json';registration=read(registration_path)
    if timing['status']!='completed'or execution['status']!='completed'or len(execution['completed'])!=30:
        raise ValueError('All registered CPU responses required')
    if not all(w['reaped']and w['returncode']==0 for w in execution['workers']):raise ValueError('Every timing worker must terminate and reap')
    if timing['source_sha256']!=sha(HERE/'benchmark_displacement.py')or timing['registration_sha256']!=sha(registration_path):
        raise ValueError('Measured timing source/registration changed')
    if timing['timed_requests']!=1920 or not timing['balanced_serial']or timing['threads']!=1:raise ValueError('Fixed CPU population/scope changed')
    paths=[directory/'summary.json',directory/'execution.json',registration_path,DOCS/'temporal-displacement-results.json']
    for row in execution['completed']:
        path=ROOT/row['path']
        if sha(path)!=row['sha256']:raise ValueError('Retained raw response changed')
        paths.append(path)
    evaluations={};preparation=[];per_grid=[]
    for model in registration['models']:
        family={'b2-displacement':'temporal-displacement-evaluation-01','b2-coupled':'temporal-coupled-evaluation-01',
            'b2-provisional-aux':'temporal-provisional-evaluation-01'}[model]
        path=BASE/family/model/'summary.json';value=read(path);paths.append(path)
        if value['status']!='completed'or len(value['scenes'])!=8 or value['bank_sha256']!=timing['bank_sha256']:
            raise ValueError('Exact accuracy population required')
        evaluations[model]=value
        ready_path=directory/f'{model}-qualification.json';ready=read(ready_path);paths.append(ready_path)
        if ready['kind']!='ready'or ready['cases']!=64 or ready['model']['weights_sha256']!=sha(BASE/f'{model}-pair-01/restore/weights.npz'):
            raise ValueError('Exact restored CPU weights/readiness required')
        preparation.append(dict(identifier=model,preparation_after_imports_s=ready['setup_s'],peak_rss_bytes=ready['peak_rss_bytes']))
        raw=[]
        for repeat in range(5):
            data=read(directory/f'{model}-full-r{repeat}.json');raw.append(data)
            if data['identifier']!=model or data['horizons']!=2 or data['cuda_initialized']or data['threads']!=1 or data['affinity']!=[23]:
                raise ValueError('Actual CPU execution context changed')
            preparation[-1]['peak_rss_bytes']=max(preparation[-1]['peak_rss_bytes'],data['peak_rss_bytes'])
        for grid in sorted({tuple(row['counts_xyz'])for repeat in raw for row in repeat['rows']}):
            values=[statistics.median(row['seconds']for row in repeat['rows']if tuple(row['counts_xyz'])==grid)*1000 for repeat in raw]
            per_grid.append(dict(identifier=model,counts_xyz=list(grid),median_block_ms=statistics.median(values),
                median_ms_per_generated_step=statistics.median(values)/2,min_repeat_median_ms=min(values),max_repeat_median_ms=max(values)))
    points=[];comparison=[]
    for row in timing['rows']:
        if row['variant']!='full':continue
        for i,interval in enumerate(('single_block','rollout_12','rollout_48','rollout_240')):
            errors=aggregate(evaluations[row['identifier']],row['regime'],i)
            error=max(errors['worst'].values())if errors['complete']else None
            point=dict(track='B',identifier=row['identifier'],horizons=2,regime=row['regime'],interval=interval,
                median_ms=row['median_ms_per_generated_step'],full_block_ms=row['median_block_ms'],worst_error=error,
                field_worst=errors['worst'],observed_prefix_worst=errors['observed_prefix_worst'],all_finite_complete=errors['complete'],
                accuracy_pass=errors['accuracy_pass'],target_met=errors['accuracy_pass']and row['median_block_ms']<4,
                min_repeat_ms=row['min_repeat_median_ms']/2,max_repeat_ms=row['max_repeat_median_ms']/2,
                p95_full_block_ms=row['p95_block_ms'],maximum_full_block_ms=row['maximum_block_ms'],
                timing_scope='Complete model blocks from64fixed reference-history windows; not sustained own-history/application time',
                accuracy_scope='Every equation/P/V/T output on all eight development scenes and declared interval')
            comparison.append(point)
            if row['identifier']=='b2-displacement':points.append(point)
    evidence={str(p.relative_to(ROOT)):sha(p)for p in paths}
    write(output,dict(status='completed',timing=timing['rows'],points=points,comparison=comparison,per_grid=per_grid,worker_preparation=preparation,
        source_evidence=evidence,reporter_sha256=sha(__file__),CPU_latency_claim=True,full_state_goal_achieved=False,
        sustained_application_throughput_claim=False,original_goal_complete=False,reserved_scenes_consumed=False,
        old_frontier_points_not_replaced=True,whole_block_median_is_not_per_grid_or_worst_call_guarantee=True))
    lines=['# Learned displacement CPU cost and accuracy','',
        'Five balanced serial repeats time exact restored weights on CPU23, one thread and batch size one. '
        'The full scope includes state encoding, prescribed forcing, all internal model work and every physical output; '
        'tensor-only timings are retained separately. Each worker uses 48 small-grid and 16 separate 32³ windows.','',
        '| Model | Small-grid block ms | Per generated step ms | Block p95 ms | Worst single-block error | Combined pass |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for model in registration['models']:
        p=next(v for v in comparison if v['identifier']==model and v['regime']=='small'and v['interval']=='single_block')
        lines.append(f'| {model} | {p["full_block_ms"]:.4f} | {p["median_ms"]:.4f} | {p["p95_full_block_ms"]:.4f} | {p["worst_error"]} | {p["target_met"]} |')
    lines.extend(['','The aggregate median spans different grids. Per-grid values, tails, individual repeats, setup and RSS '
        'remain in the evidence. Setup includes model/map construction, case preparation and warmup after numerical imports; '
        'it is separate from warm inference. These fixed-window model costs are not measured sustained own-history or application throughput. '
        'The same costs accompany long-interval accuracy rows as a comparison, with incomplete rollouts retaining unavailable complete worst errors.','',
        '[All timings, accuracy joins and source evidence](temporal-displacement-cost-results.json) · '
        '[Full-state accuracy and custody](temporal-displacement-results.md) · [Original objectives](plan.md)',''])
    (DOCS/'temporal-displacement-cost-results.md').write_text('\n'.join(lines))
    print('Published actual displacement CPU costs; sustained application and complete goal remain separate',flush=True)


if __name__=='__main__':run()
