"""Audit fresh reserved evidence for every frozen horizon/seed without reselection."""
from summarize_coupled import aggregate
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'temporal-reserved-accuracy-results.json'
    if output.exists():raise ValueError('Fresh reserved accuracy report required')
    batch_path=BASE/'temporal-final-reserved-evaluation-batch-01/execution.json';batch=read(batch_path)
    if batch['status']!='completed'or not batch['owned_child_reaped']or len(batch['completed'])!=10:
        raise ValueError('All ten frozen-model evaluation owners must complete and reap')
    frozen_path=DOCS/'temporal-final-selection-01.json';frozen=read(frozen_path)
    for name,digest in frozen['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Pre-capture frozen inference/evaluation source changed')
    registry_path=ROOT/frozen['model_registry'];registry=read(registry_path)
    if sha(registry_path)!=frozen['model_registry_sha256']:raise ValueError('Frozen complete model registry changed')
    bank_path=BASE/'temporal-reserved-240-01/bank.json';bank=read(bank_path)
    if bank['status']!='completed'or bank['frozen_selection_sha256']!=sha(frozen_path)or len(bank['trajectories'])!=8:
        raise ValueError('Original complete reserved population must follow selection freeze')
    reservation=read(DOCS/'temporal-scene-reservation.json')
    expected={r['identifier']:r for r in reservation['rows']if r['split']=='reserved'}
    if set(expected)!={r['identifier']for r in bank['trajectories']}:raise ValueError('Original reserved scene identities changed')
    sources=[batch_path,frozen_path,registry_path,bank_path,DOCS/'temporal-scene-reservation.json']
    archives=0
    for row in bank['trajectories']:
        receipt_path=ROOT/row['receipt'];receipt=read(receipt_path)
        if sha(receipt_path)!=row['receipt_sha256']or receipt['status']!='completed'or receipt['split']!='reserved':
            raise ValueError('Authoritative reserved receipt changed')
        scene={k:v for k,v in expected[row['identifier']].items()if k not in ('identifier','split')}
        if receipt['scene']!=scene or len(receipt['states'])!=241:raise ValueError('All original scene parameters/frames required')
        custody=receipt['native_process']
        if not custody['reaped']or custody['returncode']!=0 or not custody['graceful_protocol_complete']:
            raise ValueError('Every native reference child must complete/reap')
        for step,state in enumerate(receipt['states']):
            if state['step']!=step or state['time_s']!=step/60 or sha(receipt_path.parent/state['path'])!=state['sha256']:
                raise ValueError('Consecutive complete reserved state archive changed')
            archives+=1
        sources.append(receipt_path)
    rows=[];models=[]
    for method in registry['methods']:
        identifier=method['identifier'];k=method['model']['horizons']
        path=BASE/'temporal-reserved-evaluation-01'/identifier/'summary.json';value=read(path)
        wrapper_path=path.parent/'wrapper-receipt.json';wrapper=read(wrapper_path)
        if value['status']!='completed'or value['bank_sha256']!=sha(bank_path)or len(value['scenes'])!=8 or {s['identifier']for s in value['scenes']}!=set(expected):
            raise ValueError('Each frozen model must evaluate all eight reserved scenes')
        if value['model']['weights_sha256']!=method['model']['weights_sha256']or value['model']['application_policy']!=method['model']['policy']:
            raise ValueError('Reserved outputs belong to a different checkpoint or arithmetic policy')
        if wrapper['core_summary_sha256']!=sha(path)or not wrapper['accuracy_recomputed_for_actual_deployment']or not wrapper['training_teacher_disabled']:
            raise ValueError('Actual reserved policy evaluation and teacher isolation required')
        for scene in value['scenes']:
            raw_path=path.parent/(scene['identifier']+'.json');raw=read(raw_path);sources.append(raw_path)
            single=[r for r in raw['rows']if r['mode']=='single_block']
            if len(single)!=121*k or {(r['start'],r['horizon'])for r in single}!={(s,h)for s in range(121)for h in range(1,k+1)}:
                raise ValueError('Every intermediate reserved prediction must be scored')
            if scene['first_unavailable_step']is None and [r['step']for r in raw['rows']if r['mode']=='own_history']!=list(range(1,241)):
                raise ValueError('Every own240 physical state required for complete intervals')
        local=[]
        for regime in ('small','32cubed'):
            for index,interval in enumerate(('single_block','rollout_12','rollout_48','rollout_240')):
                entry=dict(identifier=identifier,horizons=k,replication_ordinal=method['replication_ordinal'],
                    policy=method['model']['policy'],regime=regime,interval=interval,**aggregate(value,regime,index))
                rows.append(entry);local.append(entry)
        models.append(dict(identifier=identifier,horizons=k,replication_ordinal=method['replication_ordinal'],
            all_eight_scene_accuracy_pass=all(r['accuracy_pass']for r in local),
            original_primary=method['replication_ordinal']==1,accuracy_summary=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))))
        sources.extend([path,wrapper_path])
    write(output,dict(status='completed',rows=rows,models=models,source_evidence={str(p.relative_to(ROOT)):sha(p)for p in sources},
        reporter_sha256=sha(__file__),original_scene_reservation_preserved=True,verified_state_archives=archives,
        original_primary_seed_retained=True,all_added_seeds_retained=True,all_requested_K_retained=True,
        all_intermediate_outputs_required=True,own_history_steps=240,CPU_latency_claim=False,
        reserved_scenes_consumed=True,training_on_reserved=False,original_goal_complete=False))
    lines=['# Frozen temporal models: original reserved-scene accuracy','',
        'All ten frozen checkpoints/policies are evaluated, retaining every additional seed and all requested horizons. '
        'Eight whole scenes were reserved before capture. Each reference contains the initial state and240consecutive '
        'native steps; all archives and native child custody are verified. No model or primary seed is reselected.','',
        '| Model | Seed | Grid group | Interval | Complete | Worst error | Accuracy pass |',
        '| --- | ---: | --- | --- | --- | ---: | --- |']
    for row in rows:
        worst=f'{max(row["worst"].values()):.7g}'if row['complete']else'Incomplete'
        lines.append(f'| {row["identifier"]} | {row["replication_ordinal"]} | {row["regime"]} | {row["interval"]} | {row["complete"]} | {worst} | {row["accuracy_pass"]} |')
    lines.extend(['','Every equation/pressure/velocity/tracer output uses the original near-zero and4e-3criteria. '
        'This report establishes actual reserved accuracy; fresh CPU and sustained application measurements remain separate.',
        '', '[All models, fields, intervals and source evidence](temporal-reserved-accuracy-results.json)',''])
    (DOCS/'temporal-reserved-accuracy-results.md').write_text('\n'.join(lines))
    print('Audited all10frozen models on8original reserved scenes and1928state archives',flush=True)


if __name__=='__main__':run()
