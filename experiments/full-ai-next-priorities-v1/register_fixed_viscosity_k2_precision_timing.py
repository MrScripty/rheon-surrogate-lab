"""Freeze the paired cost comparison after actual new-weight precision accuracy."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'fixed-viscosity-k2-precision-timing-registration-01.json'
    if output.exists():raise ValueError('Fresh corrected K2 precision timing registration required')
    batch=read(BASE/'fixed-viscosity-k2-precision-evaluation-batch-01/execution.json')
    if batch['status']!='completed'or not batch['owned_child_reaped']:
        raise ValueError('Actual precision evaluation must complete and reap')
    evaluation=BASE/'fixed-viscosity-k2-float32-evaluation-01/summary.json'
    value=read(evaluation)
    if value['status']!='completed'or len(value['scenes'])!=8:
        raise ValueError('Complete eight-scene actual precision evaluation required')
    protocol=read(DOCS/'fixed-viscosity-k2-float32-evaluation-protocol-01.json')
    sources=dict(protocol['source_sha256'])
    for name in ('register_fixed_viscosity_k2_precision_timing.py','report_fixed_viscosity_k2_precision.py'):
        sources[name]=sha(HERE/name)
    for name,digest in sources.items():
        if sha(HERE/name)!=digest:raise ValueError('Registered precision source changed')
    registry=BASE/'fixed-viscosity-k2-precision-inputs-01/models.json';models=read(registry)['models']
    evidence=[evaluation,evaluation.parent/'wrapper-receipt.json',
        BASE/'fixed-viscosity-evaluation-01/b2-fixed-viscosity/summary.json',
        BASE/'fixed-viscosity-execution-qualification-k2-01.json']
    write(output,dict(status='registered_before_paired_timing',models=[m['identifier']for m in models],
        source_sha256=sha(HERE/'benchmark_fixed_viscosity_k2_precision.py'),loader_sha256=sources,
        model_registry_sha256=sha(registry),bank_sha256=protocol['bank_sha256'],
        execution_assets_sha256=protocol['execution_assets_sha256'],
        accuracy_evidence_sha256={str(p.relative_to(ROOT)):sha(p)for p in evidence},
        threads=1,batch_size=1,cpu=23,repeats=5,balanced_serial=True,
        full_and_tensor_separate=True,accuracy_failure_does_not_prevent_cost_measurement=True,
        optimizer_updates=0,reserved_scenes_consumed=False))
    print('Registered five-repeat actual corrected K2float64/float32 comparison',flush=True)


if __name__=='__main__':run()
