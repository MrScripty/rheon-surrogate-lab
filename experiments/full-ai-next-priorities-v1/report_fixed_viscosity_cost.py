"""Join measured full-block costs to exact every-field/own240 accuracy."""
import statistics
from summarize_coupled import aggregate
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'fixed-viscosity-cost-results.json'
    if output.exists():raise ValueError('Fresh family CPU cost report required')
    directory=BASE/'fixed-viscosity-timing-01';timing=read(directory/'summary.json');execution=read(directory/'execution.json')
    reg_path=DOCS/'fixed-viscosity-timing-registration-01.json';reg=read(reg_path)
    if timing['status']!='completed'or execution['status']!='completed'or len(execution['completed'])!=55:
        raise ValueError('All55finite timing responses required')
    if not all(w['reaped']and w['returncode']==0 for w in execution['workers']):raise ValueError('All timing workers must reap')
    if timing['timed_requests']!=3520 or timing['source_sha256']!=sha(HERE/'benchmark_fixed_viscosity_family.py')or timing['registration_sha256']!=sha(reg_path):
        raise ValueError('Registered source/population differs')
    family_path=DOCS/'fixed-viscosity-family-results.json';family=read(family_path)
    if family['status']!='completed':raise ValueError('Complete source-bound accuracy/control audit required')
    paths=[directory/'summary.json',directory/'execution.json',reg_path,family_path]
    for response in execution['completed']:
        path=ROOT/response['path']
        if sha(path)!=response['sha256']:raise ValueError('Timing response changed')
        raw=read(path)
        if raw['threads']!=1 or raw['affinity']!=[23]or raw['cuda_initialized']or len(raw['rows'])!=64:
            raise ValueError('Actual CPU timing context differs')
        paths.append(path)
    evaluations={};preparation=[];per_grid=[]
    for k in (1,2,3,4):
        identifier=f'b{k}-fixed-viscosity-fast';base_identifier=identifier.removesuffix('-fast')
        evaluation_path=BASE/'fixed-viscosity-evaluation-01'/base_identifier/'summary.json'
        evaluation=read(evaluation_path);evaluations[identifier]=evaluation;paths.append(evaluation_path)
        if evaluation['bank_sha256']!=timing['bank_sha256']:raise ValueError('Accuracy/timing populations differ')
        proof_path=BASE/f'fixed-viscosity-execution-qualification-k{k}-01.json';proof=read(proof_path);paths.append(proof_path)
        if proof['status']!='passed'or not proof['all_physical_bytes_exact']or proof['outputs']!=(121*k+240)*8:
            raise ValueError('Every physical execution output must qualify')
        ready_path=directory/(identifier+'-qualification.json');ready=read(ready_path);paths.append(ready_path)
        if ready['kind']!='ready'or ready['cases']!=64 or ready['model']['weights_sha256']!=evaluation['model']['weights_sha256']:
            raise ValueError('Timed weights differ from evaluated weights')
        if ready['model']['execution_variant']!='fused-readout-fast-preparation-v1':raise ValueError('Timed execution differs from qualification')
        values=[read(directory/f'{identifier}-full-r{r}.json')for r in range(5)]
        preparation.append(dict(identifier=identifier,preparation_after_imports_s=ready['setup_s'],
            peak_rss_bytes=max([ready['peak_rss_bytes']]+[v['peak_rss_bytes']for v in values])))
        for grid in sorted({tuple(r['counts_xyz'])for v in values for r in v['rows']}):
            medians=[statistics.median(r['seconds']for r in v['rows']if tuple(r['counts_xyz'])==grid)*1000 for v in values]
            per_grid.append(dict(identifier=identifier,counts_xyz=list(grid),median_block_ms=statistics.median(medians),
                median_ms_per_generated_step=statistics.median(medians)/k,min_repeat_median_ms=min(medians),max_repeat_median_ms=max(medians)))
    points=[];sequential=[];comparisons=[]
    for row in timing['rows']:
        if row['variant'].startswith('sequential'):
            sequential.append(dict(row,accuracy_claim=False,scope='K independent B1 calls from one reference current state followed by own predictions; every physical output included'))
            continue
        if row['variant']!='full':continue
        k=row['horizons'];evaluation=evaluations[row['identifier']]
        for index,interval in enumerate(('single_block','rollout_12','rollout_48','rollout_240')):
            errors=aggregate(evaluation,row['regime'],index)
            points.append(dict(track='B',identifier=row['identifier'],horizons=k,regime=row['regime'],interval=interval,
                median_ms=row['median_ms_per_generated_step'],full_block_ms=row['median_block_ms'],
                worst_error=max(errors['worst'].values())if errors['complete']else None,field_worst=errors['worst'],
                all_finite_complete=errors['complete'],accuracy_pass=errors['accuracy_pass'],control_only=k==1,
                target_met=k>1 and errors['accuracy_pass']and row['median_block_ms']<2*k,
                min_repeat_ms=row['min_repeat_median_ms']/k,max_repeat_ms=row['max_repeat_median_ms']/k,
                p95_full_block_ms=row['p95_block_ms'],maximum_full_block_ms=row['maximum_block_ms'],
                exact_fast_execution_outputs=(121*k+240)*8,
                timing_scope='Complete model blocks on64fixed reference-history windows; sustained application/event cost separate',
                accuracy_scope='Every equation/P/V/T criterion and all outputs on8development scenes through declared own-history interval'))
    for control in sequential:
        row=next(r for r in timing['rows']if r['variant']=='full'and r['horizons']==control['horizons']and r['regime']==control['regime'])
        comparisons.append(dict(horizons=row['horizons'],regime=row['regime'],bundled_block_ms=row['median_block_ms'],
            sequential_B1_block_ms=control['median_block_ms'],bundled_speedup=control['median_block_ms']/row['median_block_ms'],
            scope='Matched complete physical-output counts; separate managed models with common36000physical training targets; no equal-FLOP claim'))
    qualified=[p['identifier']for p in points if p['regime']=='small'and p['interval']=='rollout_240'and p['target_met']]
    write(output,dict(status='completed',points=points,timing=timing['rows'],per_grid=per_grid,worker_preparation=preparation,
        sequential_controls=sequential,joint_vs_sequential=comparisons,small_grid_combined_qualified=qualified,
        source_evidence={str(p.relative_to(ROOT)):sha(p)for p in paths},reporter_sha256=sha(__file__),
        CPU_latency_claim=True,sustained_application_throughput_claim=False,
        replica_and_reserved_gates_pending=True,reserved_scenes_consumed=False,original_goal_complete=False))
    lines=['# Corrected neural family: measured CPU cost and accuracy','',
        'Five balanced serial CPU repeats use one thread, batch size one and core23. Complete costs include '
        'current-state encoding, prescribed forcing, every intermediate prediction and physical reconstruction. '
        'Each fast model preserves every evaluated physical output byte for byte against its exact managed CPU restore.','',
        '| Model | Small-grid block ms | Per step ms | Block p95 ms | Worst own240 error | Combined target |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for k in (1,2,3,4):
        p=next(p for p in points if p['horizons']==k and p['regime']=='small'and p['interval']=='rollout_240')
        error=f'{p["worst_error"]:.7g}'if p['worst_error']is not None else'Incomplete interval'
        lines.append(f'| {p["identifier"]} | {p["full_block_ms"]:.4f} | {p["median_ms"]:.4f} | {p["p95_full_block_ms"]:.4f} | {error} | {p["target_met"]if k>1 else"Control"} |')
    lines.extend(['','Sequential B1 comparisons use complete2/3/4-call blocks from reference current state followed by '
        'predicted inputs; no future reference state enters the control. Those latency controls do not inherit '
        'single-step reference-history accuracy. The full raw controls and bundled speed ratios are reported separately.','',
        'The median mixes three small geometries; repeat spread, latency tails, each grid,32³, worker preparation '
        'and peak RSS remain separately visible. These fixed-window costs do not establish sustained application '
        'or interaction throughput. Replica and reserved-scene gates remain open.','',
        '[Raw CPU costs and exact accuracy joins](fixed-viscosity-cost-results.json) · '
        '[Every-field accuracy and work audit](fixed-viscosity-family-results.md)',''])
    (DOCS/'fixed-viscosity-cost-results.md').write_text('\n'.join(lines))
    print('Measured corrected-family costs; small-grid combined qualifiers:',qualified,flush=True)


if __name__=='__main__':run()
