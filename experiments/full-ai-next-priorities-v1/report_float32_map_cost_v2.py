"""Bind repaired/fused CPU costs to exact all-field and own240 accuracy."""
import statistics
from summarize_coupled import aggregate
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'float32-map-cost-results.json'
    if output.exists():raise ValueError('Fresh neural viscosity CPU report required')
    directory=BASE/'float32-map-timing-01';timing=read(directory/'summary.json');execution=read(directory/'execution.json')
    batch_path=BASE/'float32-map-timing-batch-01/execution.json';batch=read(batch_path)
    reg_path=DOCS/'float32-map-timing-registration-01.json';reg=read(reg_path)
    if timing['status']!='completed'or execution['status']!='completed'or not batch['owned_child_reaped']or batch.get('active')is not None:
        raise ValueError('Complete finite serial CPU campaign required')
    if batch['status']=='stopped':
        commands=batch['completed']
        if len(commands)!=3 or not all(c['reaped']for c in commands)or any(c['returncode']for c in commands[:2]):
            raise ValueError('Only the registered publication-order failure may be recovered')
        if commands[2]['id']!='publish-actual-F32-map-cost-and-accuracy'or commands[2]['returncode']!=1:
            raise ValueError('Unexpected terminal publication failure')
        failure=ROOT/commands[2]['log']
        if sha(failure)!=commands[2]['log_sha256']or 'ValueError: Complete finite serial CPU campaign required'not in failure.read_text():
            raise ValueError('Expected retained circular publication check required')
    elif batch['status']!='completed':raise ValueError('Original timing controller must be terminal and reaped')
    admission_path=BASE/'float32-map-publication-admission-01.json';admission=read(admission_path)
    if admission['original_batch_sha256']!=sha(batch_path)or admission['numerical_work_replayed']is not False:
        raise ValueError('Read-only publication admission required')
    if len(execution['completed'])!=20 or not all(w['reaped']and w['returncode']==0 for w in execution['workers']):
        raise ValueError('All timing responses and worker reaping required')
    if timing['timed_requests']!=1280 or timing['source_sha256']!=sha(HERE/'benchmark_float32_map.py')or timing['registration_sha256']!=sha(reg_path):
        raise ValueError('Registered source or timing population differs')
    qualification_path=BASE/'fast-preparation-model-qualification-01.json';qualification=read(qualification_path)
    if qualification['status']!='passed'or qualification['outputs']!=3856 or not qualification['all_physical_bytes_exact']:
        raise ValueError('Full-population exact fused execution required')
    parent_path=BASE/'neural-viscosity-evaluation-01/b2-displacement-viscosity/summary.json'
    repaired_path=BASE/'float32-map-evaluation-01/summary.json'
    parent=read(parent_path);repaired=read(repaired_path)
    if parent['bank_sha256']!=timing['bank_sha256']or repaired['bank_sha256']!=timing['bank_sha256']:
        raise ValueError('Actual accuracy population differs')
    paths=[admission_path,DOCS/'float32-map-publication-repair-registration-01.json',directory/'summary.json',directory/'execution.json',batch_path,reg_path,qualification_path,parent_path,repaired_path,
        DOCS/'neural-viscosity-results.json',DOCS/'float32-learned-map-results.json',BASE/'b2-displacement-viscosity-assembly-01/receipt.json']
    for row in execution['completed']:
        p=ROOT/row['path']
        if sha(p)!=row['sha256']:raise ValueError('Raw timing response changed')
        paths.append(p)
    preparation=[];per_grid=[]
    for model in reg['models']:
        ready_path=directory/f'{model}-qualification.json';ready=read(ready_path);paths.append(ready_path)
        expected=parent['model']['weights_sha256']if model=='b2-displacement-viscosity-fast'else repaired['model']['weights_sha256']
        if ready['kind']!='ready'or ready['cases']!=64 or ready['model']['weights_sha256']!=expected:
            raise ValueError('Timed neural weights differ from evaluated weights')
        if model.endswith('-fused')and ready['model'].get('execution_variant')!='fused-fixed-field-readout-v1':
            raise ValueError('Timed fused policy differs from exact qualification')
        if model.endswith('-fast')and ready['model'].get('execution_variant')!='fused-readout-fast-preparation-v1':
            raise ValueError('Timed fast preparation differs from qualified policy')
        if model.endswith('-f32map')and ready['model'].get('execution_variant')!='fast-preparation-float32-learned-map-v1':
            raise ValueError('Timed float32 arithmetic differs from evaluated policy')
        preparation.append(dict(identifier=model,preparation_after_imports_s=ready['setup_s'],peak_rss_bytes=ready['peak_rss_bytes']))
        raw=[]
        for repeat in range(5):
            value=read(directory/f'{model}-full-r{repeat}.json');raw.append(value)
            if value['identifier']!=model or value['threads']!=1 or value['affinity']!=[23]or value['cuda_initialized']:
                raise ValueError('Actual CPU timing context differs')
            preparation[-1]['peak_rss_bytes']=max(preparation[-1]['peak_rss_bytes'],value['peak_rss_bytes'])
        for grid in sorted({tuple(r['counts_xyz'])for repeat in raw for r in repeat['rows']}):
            values=[statistics.median(r['seconds']for r in repeat['rows']if tuple(r['counts_xyz'])==grid)*1000 for repeat in raw]
            per_grid.append(dict(identifier=model,counts_xyz=list(grid),median_block_ms=statistics.median(values),
                median_ms_per_generated_step=statistics.median(values)/2,min_repeat_median_ms=min(values),max_repeat_median_ms=max(values)))
    comparison=[];points=[]
    for row in timing['rows']:
        if row['variant']!='full':continue
        accuracy=parent if row['identifier']=='b2-displacement-viscosity-fast'else repaired
        for index,label in enumerate(('single_block','rollout_12','rollout_48','rollout_240')):
            errors=aggregate(accuracy,row['regime'],index)
            point=dict(track='B',identifier=row['identifier'],horizons=2,regime=row['regime'],interval=label,
                median_ms=row['median_ms_per_generated_step'],full_block_ms=row['median_block_ms'],
                worst_error=max(errors['worst'].values())if errors['complete']else None,field_worst=errors['worst'],
                all_finite_complete=errors['complete'],accuracy_pass=errors['accuracy_pass'],
                target_met=errors['accuracy_pass']and row['median_block_ms']<4,
                min_repeat_ms=row['min_repeat_median_ms']/2,max_repeat_ms=row['max_repeat_median_ms']/2,
                p95_full_block_ms=row['p95_block_ms'],maximum_full_block_ms=row['maximum_block_ms'],
                timing_scope='Complete model blocks from64fixed reference-history windows; sustained own-history/application cost separate',
                accuracy_scope='All equation/P/V/T outputs and declared own-history interval on8development scenes',
                exact_fused_accuracy_inherited=row['identifier']=='b2-displacement-viscosity-fast')
            comparison.append(point)
            if row['identifier']=='b2-displacement-viscosity-fast-f32map':points.append(point)
    write(output,dict(status='completed',retained_publication_order_failure=batch['status']=='stopped',numerical_work_replayed=False,points=points,comparison=comparison,timing=timing['rows'],per_grid=per_grid,worker_preparation=preparation,
        source_evidence={str(p.relative_to(ROOT)):sha(p)for p in paths},reporter_sha256=sha(__file__),CPU_latency_claim=True,
        full_model_physical_bytes_exact_for_fused=False,float32_accuracy_recomputed=True,sustained_application_throughput_claim=False,
        K3_K4_and_replicas_pending=True,reserved_scenes_consumed=False,original_goal_complete=False))
    lines=['# Float32 learned-map CPU cost and actual accuracy','',
        'Five balanced serial CPU repeats use exact restored/composed neural weights, one thread, batch size one and core23. '
        'Full costs include physical encoding, known forcing, all intermediate predictions and physical field reconstruction. '
        'The float64 baseline retains its qualified accuracy; float32 map arithmetic receives its own complete accuracy evaluation.','',
        '| Model | Small-grid block ms | Per generated step ms | Block p95 ms | Worst own240 error | Combined own240 pass |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for model in reg['models']:
        point=next(p for p in comparison if p['identifier']==model and p['regime']=='small'and p['interval']=='rollout_240')
        lines.append(f'| {model} | {point["full_block_ms"]:.4f} | {point["median_ms"]:.4f} | {point["p95_full_block_ms"]:.4f} | {point["worst_error"]:.7g} | {point["target_met"]} |')
    lines.extend(['','The median mixes three small geometries and is not a worst-call or per-grid guarantee. '
        '32³, per-grid latency, tails, preparation and peak RSS are reported separately in the evidence. '
        'Preparation includes model/maps, case preparation and warmup after imports. These fixed-window costs do not establish '
        'sustained own-history/application throughput. K3/K4, replicas and reserved scenes remain separate requirements.','',
        'The first publisher expected its own parent to have completed. Its failed read-only attempt is retained; '
        'this dependent publisher validates the original timing responses and worker reaping without repeating numerical work.','',
        '[Raw timings, exact accuracy joins and sources](float32-map-cost-results.json) · '
        '[Recomputed float32 accuracy](float32-learned-map-results.md)',''])
    (DOCS/'float32-map-cost-results.md').write_text('\n'.join(lines))
    print('Published float32 learned-map CPU costs and actual accuracy; remaining goal gates stay explicit',flush=True)


if __name__=='__main__':run()
