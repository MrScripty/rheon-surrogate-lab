"""Join corrected K2 policy timings to their own complete accuracy evidence."""
import statistics
from summarize_coupled import aggregate
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'fixed-viscosity-k2-precision-results.json'
    if output.exists():raise ValueError('Fresh corrected K2 precision report required')
    directory=BASE/'fixed-viscosity-k2-precision-timing-01'
    timing=read(directory/'summary.json');execution=read(directory/'execution.json')
    path=DOCS/'fixed-viscosity-k2-precision-timing-registration-01.json';reg=read(path)
    if timing['status']!='completed'or execution['status']!='completed'or len(execution['completed'])!=20:
        raise ValueError('All twenty timing responses required')
    if len(execution['workers'])!=2 or not all(w['reaped']and w['returncode']==0 for w in execution['workers']):
        raise ValueError('Both timing workers must terminate successfully and reap')
    if timing['source_sha256']!=sha(HERE/'benchmark_fixed_viscosity_k2_precision.py')or timing['registration_sha256']!=sha(path)or timing['timed_requests']!=1280:
        raise ValueError('Actual benchmark source/population differs')
    for name,digest in reg['loader_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Frozen deployment/report source changed')
    paths=[path,directory/'summary.json',directory/'execution.json']
    for relative,digest in reg['accuracy_evidence_sha256'].items():
        source=ROOT/relative
        if sha(source)!=digest:raise ValueError('Actual accuracy evidence changed')
        paths.append(source)
    for response in execution['completed']:
        source=ROOT/response['path'];raw=read(source)
        if sha(source)!=response['sha256']or raw['threads']!=1 or raw['affinity']!=[23]or raw['cuda_initialized']or len(raw['rows'])!=64:
            raise ValueError('Raw CPU response/context changed')
        paths.append(source)
    registry_path=BASE/'fixed-viscosity-k2-precision-inputs-01/models.json';registry=read(registry_path)
    if sha(registry_path)!=reg['model_registry_sha256']:raise ValueError('Concrete deployment registry changed')
    paths.append(registry_path);points=[];per_grid=[];preparation=[]
    for model in registry['models']:
        identifier=model['identifier'];f32=model['policy']=='fast-float32-map'
        evaluation=read(BASE/('fixed-viscosity-k2-float32-evaluation-01/summary.json'if f32 else'fixed-viscosity-evaluation-01/b2-fixed-viscosity/summary.json'))
        if evaluation['bank_sha256']!=timing['bank_sha256']or evaluation['model']['weights_sha256']!=model['weights_sha256']:
            raise ValueError('Accuracy belongs to different weights or population')
        if f32 and (evaluation['model'].get('application_policy')!=model['policy']or evaluation['model'].get('application_loader_sha256')!=sha(HERE/'application_model_support.py')):
            raise ValueError('Float32 policy requires its own actual full evaluation')
        ready_path=directory/(identifier+'-qualification.json');ready=read(ready_path);paths.append(ready_path)
        if ready['cases']!=64 or ready['model']['weights_sha256']!=model['weights_sha256']or ready['model']['application_policy']!=model['policy']:
            raise ValueError('Timed checkpoint/policy differs from actual accuracy')
        raw=[read(directory/f'{identifier}-full-r{r}.json')for r in range(5)]
        preparation.append(dict(identifier=identifier,preparation_after_imports_s=ready['setup_s'],
            peak_rss_bytes=max([ready['peak_rss_bytes']]+[r['peak_rss_bytes']for r in raw])))
        for grid in sorted({tuple(row['counts_xyz'])for r in raw for row in r['rows']}):
            values=[statistics.median(row['seconds']for row in r['rows']if tuple(row['counts_xyz'])==grid)*1000 for r in raw]
            per_grid.append(dict(identifier=identifier,counts_xyz=list(grid),median_block_ms=statistics.median(values),
                median_ms_per_generated_step=statistics.median(values)/2,min_repeat_median_ms=min(values),max_repeat_median_ms=max(values)))
        for row in timing['rows']:
            if row['identifier']!=identifier or row['variant']!='full':continue
            for index,interval in enumerate(('single_block','rollout_12','rollout_48','rollout_240')):
                errors=aggregate(evaluation,row['regime'],index)
                points.append(dict(track='B',identifier=identifier,horizons=2,policy=model['policy'],
                    regime=row['regime'],interval=interval,median_ms=row['median_ms_per_generated_step'],
                    full_block_ms=row['median_block_ms'],worst_error=max(errors['worst'].values())if errors['complete']else None,
                    field_worst=errors['worst'],all_finite_complete=errors['complete'],accuracy_pass=errors['accuracy_pass'],
                    target_met=errors['accuracy_pass']and row['median_block_ms']<4,
                    min_repeat_ms=row['min_repeat_median_ms']/2,max_repeat_ms=row['max_repeat_median_ms']/2,
                    p95_full_block_ms=row['p95_block_ms'],maximum_full_block_ms=row['maximum_block_ms'],
                    timing_scope='Complete blocks on64reference-history windows; sustained application costs separate',
                    accuracy_scope='Actual every-field/output and own-history intervals on8development scenes',
                    physical_output_equivalence_inherited=not f32))
    write(output,dict(status='completed',points=points,timing=timing['rows'],per_grid=per_grid,
        worker_preparation=preparation,source_evidence={str(p.relative_to(ROOT)):sha(p)for p in paths},
        reporter_sha256=sha(__file__),CPU_latency_claim=True,sustained_application_throughput_claim=False,
        replica_and_reserved_gates_pending=True,reserved_scenes_consumed=False,original_goal_complete=False))
    lines=['# Corrected two-step model: paired execution precision comparison','',
        'Same managed neural weights, one CPU thread, core23, batch one and five balanced serial repeats. '
        'Complete cost includes input/forcing preparation and every intermediate physical field. '
        'Float32 accuracy is recomputed on all eight scenes and through240 own-history steps.','',
        '| Policy | Block ms | Per step ms | Worst own240 error | Combined median pass |',
        '| --- | ---: | ---: | ---: | --- |']
    for model in registry['models']:
        point=next(p for p in points if p['identifier']==model['identifier']and p['regime']=='small'and p['interval']=='rollout_240')
        lines.append(f'| {model["policy"]} | {point["full_block_ms"]:.4f} | {point["median_ms"]:.4f} | {point["worst_error"]:.7g} | {point["target_met"]} |')
    lines.extend(['','This warm mixed-grid median is not a per-grid or worst-call guarantee. '
        'Tails,32³, setup/RSS, seed replications, reserved scenes and full application/event costs remain separate.',
        '', '[Raw costs, errors, per-grid measurements and sources](fixed-viscosity-k2-precision-results.json)',''])
    (DOCS/'fixed-viscosity-k2-precision-results.md').write_text('\n'.join(lines))
    print('Published actual corrected K2 precision accuracy/cost comparison',flush=True)


if __name__=='__main__':run()
