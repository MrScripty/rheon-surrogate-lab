"""Freeze actual primary/replica policy evidence before their CPU comparison."""
import copy
from application_model_evidence import require
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'fixed-viscosity-replication-timing-registration-01.json'
    if output.exists():raise ValueError('Fresh actual replica timing registration required')
    batch_path=BASE/'fixed-viscosity-replication-execution-batch-01/execution.json';batch=read(batch_path)
    if batch['status']!='completed'or not batch['owned_child_reaped']:
        raise ValueError('All actual replica execution/policy evaluations must complete and reap')
    inputs=BASE/'fixed-viscosity-replication-deployments-01';original=read(inputs/'models.json')
    registry=copy.deepcopy(original);bank=BASE/'temporal-development-240-01/bank.json'
    scene_ids=[r['identifier']for r in read(bank)['trajectories']]
    evidence={str(batch_path.relative_to(ROOT)):sha(batch_path)}
    for method in registry['methods']:
        for key in ('accuracy_summary','execution_qualification'):
            if key=='execution_qualification'and method['model']['policy']=='fast-float32-map':
                method.pop(key,None);continue
            path=ROOT/method[key]['path'];method[key]['sha256']=sha(path)
            evidence[str(path.relative_to(ROOT))]=sha(path)
        method['actual_qualification']=require(method,sha(bank),scene_ids)
        if method['model']['policy']=='fast-float32-map':
            wrapper=ROOT/method['accuracy_summary']['path'];wrapper=wrapper.parent/'wrapper-receipt.json'
            value=read(wrapper)
            if value['core_summary_sha256']!=method['accuracy_summary']['sha256']or not value['accuracy_recomputed_for_actual_deployment']:
                raise ValueError('Actual complete float32 evaluation wrapper required')
            evidence[str(wrapper.relative_to(ROOT))]=sha(wrapper)
    qualified_path=inputs/'qualified-models.json'
    if qualified_path.exists():raise ValueError('Fresh fully bound deployment registry required')
    write(qualified_path,registry)
    protocol_path=DOCS/'fixed-viscosity-replication-execution-protocol-01.json';protocol=read(protocol_path)
    sources=dict(protocol['source_sha256'])
    for name in ('register_fixed_viscosity_replica_timing.py','report_fixed_viscosity_replica_cost.py'):
        sources[name]=sha(HERE/name)
    for name,digest in sources.items():
        if sha(HERE/name)!=digest:raise ValueError('Registered replica policy source changed')
    horizons=read(DOCS/'fixed-viscosity-replication-selection-01.json')['recipe_horizons']
    write(output,dict(status='registered_before_primary_replica_CPU_comparison',
        models=[m['identifier']for m in registry['methods']],bundled_horizons=horizons,
        source_sha256=sha(HERE/'benchmark_fixed_viscosity_replicas.py'),loader_sha256=sources,
        execution_assets_sha256=protocol['execution_assets_sha256'],bank_sha256=sha(bank),
        model_registry_sha256=sha(qualified_path),accuracy_evidence_sha256=evidence,
        cpu=23,threads=1,batch_size=1,repeats=5,balanced_serial=True,
        full_and_tensor_separate=True,all_additional_seeds_retained=True,primary_seed_reselected=False,
        failed_seed_accuracy_does_not_prevent_cost_characterization=True,reserved_scenes_consumed=False,
        optimizer_updates=0))
    write(DOCS/'fixed-viscosity-replication-timing-batch-specification-01.json',dict(commands=[
        dict(id='five-balanced-primary-replica-and-sequential-B1-costs',runtime='torch',timeout=7200,
            arguments=['experiments/full-ai-next-priorities-v1/benchmark_fixed_viscosity_replicas.py',
                '--registry',str(qualified_path.relative_to(ROOT)),'--bank',str(bank.relative_to(ROOT)),
                '--output','runs/full-ai-next-priorities-v1/fixed-viscosity-replication-timing-01']),
        dict(id='publish-actual-primary-replica-costs-and-accuracy',runtime='torch',timeout=180,
            arguments=['experiments/full-ai-next-priorities-v1/report_fixed_viscosity_replica_cost.py'])]))
    print('Frozen9actual checkpoints/policies, all seeds and matched sequential controls before CPU timing',flush=True)


if __name__=='__main__':run()
