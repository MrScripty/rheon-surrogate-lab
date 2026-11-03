"""Publish every-output float32-map accuracy without inheriting parent accuracy."""
from summarize_coupled import aggregate,FIELDS
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'float32-learned-map-results.json'
    if output.exists():raise ValueError('Fresh float32-map result required')
    protocol_path=DOCS/'float32-learned-map-protocol-01.json';protocol=read(protocol_path)
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered execution or reporting source changed')
    path=BASE/'float32-map-evaluation-01/summary.json';value=read(path)
    parent_path=BASE/'neural-viscosity-evaluation-01/b2-displacement-viscosity/summary.json';parent=read(parent_path)
    wrapper_path=path.parent/'wrapper-receipt.json';wrapper=read(wrapper_path)
    if value['status']!='completed'or len(value['scenes'])!=8 or value['bank_sha256']!=parent['bank_sha256']or value['metrics_sha256']!=parent['metrics_sha256']:
        raise ValueError('All unchanged scenes and original accuracy rules required')
    if value['model']['weights_sha256']!=parent['model']['weights_sha256']or not wrapper['training_teacher_disabled']or wrapper['core_summary_sha256']!=sha(path):
        raise ValueError('Exact managed neural weights and isolated inference required')
    if value['model']['execution_variant']!='fast-preparation-float32-learned-map-v1':raise ValueError('Actual arithmetic policy differs')
    paths=[path,parent_path,wrapper_path,protocol_path];rows=[]
    for scene in value['scenes']:
        p=path.parent/(scene['identifier']+'.json');raw=read(p);paths.append(p)
        single=[r for r in raw['rows']if r['mode']=='single_block']
        if len(single)!=242 or {(r['start'],r['horizon'])for r in single}!={(s,h)for s in range(121)for h in (1,2)}:
            raise ValueError('Every intermediate reference-history output required')
        if scene['first_unavailable_step']is None and [r['step']for r in raw['rows']if r['mode']=='own_history']!=list(range(1,241)):
            raise ValueError('Every own240 state required')
    for regime in ('small','32cubed'):
        for index,interval in enumerate(('single_block','rollout_12','rollout_48','rollout_240')):
            rows.append(dict(regime=regime,interval=interval,parent=aggregate(parent,regime,index),float32_map=aggregate(value,regime,index)))
    passed=all(r['float32_map']['accuracy_pass']for r in rows)
    write(output,dict(status='completed',identifier=value['model']['identifier'],rows=rows,model=value['model'],
        accuracy_qualified_all_eight_scenes=passed,CPU_latency_claim=False,combined_target_claim=False,
        original_neural_tensors_unchanged=True,physical_output_bytes_exact_claim=False,
        source_evidence={str(p.relative_to(ROOT)):sha(p)for p in paths},reporter_sha256=sha(__file__),
        reserved_scenes_consumed=False,optimizer_updates=0,original_goal_complete=False))
    lines=['# Float32 learned pressure-map execution','',
        'The learned neural tensors are unchanged. The derived neural unit-response map and requests use float32 '
        'matrix arithmetic on the three small geometries;32³ retains its original component. This changes rounding '
        'and receives a complete new accuracy evaluation. No inverse, solver, correction or parameter fitting is added.','',
        '| Grid group | Interval | Worst P / equation / velocity / tracer | Accuracy pass |',
        '| --- | --- | --- | --- |']
    for row in rows:
        arm=row['float32_map'];text=' / '.join(f'{arm["worst"][f]:.7g}'for f in FIELDS)if arm['complete']else'Incomplete interval'
        lines.append(f'| {row["regime"]} | {row["interval"]} | {text} | {arm["accuracy_pass"]} |')
    lines.extend(['','No CPU speed claim is made before a separate balanced comparison. The original float64 '
        'preparation maps remain allocated, so halving the map arithmetic does not establish halved process memory. '
        'Every field, horizon, own240 step and original near-zero criterion remains required.','',
        '[Actual errors and source evidence](float32-learned-map-results.json)',''])
    (DOCS/'float32-learned-map-results.md').write_text('\n'.join(lines))
    print('Actual float32-map accuracy qualifies all8scenes:',passed,flush=True)


if __name__=='__main__':run()
