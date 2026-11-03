"""Summarize actual displacement constructors without forecasting benchmark claims."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'temporal-displacement-qualification-results.json'
    if output.exists():raise ValueError('Fresh actual-constructor report required')
    protocol_path=DOCS/'temporal-displacement-qualification-protocol-01.json';protocol=read(protocol_path)
    evidence={str(protocol_path.relative_to(ROOT)):sha(protocol_path)};summaries=[]
    for device in ('cpu','cuda'):
        path=BASE/f'temporal-displacement-{device}-qualification-01.json';value=read(path)
        if value['status']!='passed'or len(value['rows'])!=64 or value['optimizer_updates']!=0:
            raise ValueError('Complete actual-constructor qualification required')
        if value['protocol_sha256']!=sha(protocol_path)or value['source_sha256']!=protocol['source_sha256']:
            raise ValueError('Qualified protocol or sources changed')
        for name,digest in value['source_sha256'].items():
            if sha(HERE/name)!=digest:raise ValueError('Qualified implementation changed')
        for row in value['rows']:
            if row['counts_per_block']!={name:1 if name=='encoder'else row['horizons']for name in ('encoder','local','displacement','pressure','flux','momentum')}:
                raise ValueError('Actual sharing counts changed')
            if not row['deterministic_two_block_gradients']or not row['training_teacher_disabled_during_inference']:
                raise ValueError('Own-history gradients or inference isolation missing')
        summaries.append(dict(device=device,cases=64,parameters=value['aggregate']['2']['parameters'],
            trainable_parameters=value['aggregate']['2']['trainable_parameters'],
            maximum_CPU_device_absolute_difference=max(row['CPU_device_max_abs']for row in value['rows']),
            every_K_all_trainable_paths_active=all(v['all_trainable_tensors_active']for v in value['aggregate'].values())))
        evidence[str(path.relative_to(ROOT))]=sha(path)
    write(output,dict(status='completed_zero_update_constructor_qualification',devices=summaries,source_evidence=evidence,
        reporter_sha256=sha(__file__),optimizer_updates=0,initial_diagnostics_training_windows_only=True,
        full_state_accuracy_claim=False,CPU_latency_claim=False,candidate_frontier_point=False,reserved_scenes_consumed=False,
        next_gate='Bounded managed K2 fit, exact CPU restore, all development horizons and own240; then actual CPU timing'))
    (DOCS/'temporal-displacement-qualification-results.md').write_text(
        '# Actual displacement model qualification\n\n'
        'CPU and CUDA each pass all 64 registered constructor cases: K=1/2/3/4, '
        'four geometries, both spatial densities and cold/warm states. The model '
        'contains 100,456 parameters, with 6,843 trainable parameters.\n\n'
        'The checks exercise the actual trainer and objective through two consecutive '
        'blocks of predicted history, repeated gradients, device parity, frozen learned '
        'coefficients and active trainable paths. Each block executes one shared encoder '
        'and K local/displacement/pressure/flux/momentum readouts. Training-label code is '
        'disabled during inference checks. Passive tracer, zero/constant fields, donor '
        'bounds and entering-velocity staging pass.\n\n'
        'These are zero-update implementation checks. The diagnostic gradient tests '
        'activate separate parameter copies; initial error records use selected training '
        'windows. Neither qualifies held-out accuracy, own240 behavior or CPU inference '
        'latency. Full-state fitting, exact CPU restoration and the complete development '
        'evaluation are the next gates.\n\n'
        '[Evidence and source hashes](temporal-displacement-qualification-results.json) · '
        '[Frozen protocol](temporal-displacement-qualification-protocol-01.json) · '
        '[Implemented architecture](temporal-displacement-plan.md)\n')
    print('Published complete CPU/CUDA displacement constructor qualification; no forecast target claim',flush=True)


if __name__=='__main__':run()
