"""Freeze fresh CPU measurement identities after actual reserved accuracy, retaining failures."""
import copy
from application_model_evidence import require
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'temporal-final-reserved-timing-registration-01.json'
    if output.exists():raise ValueError('Fresh reserved CPU registration required')
    report_path=DOCS/'temporal-reserved-accuracy-results.json';report=read(report_path)
    if report['status']!='completed'or len(report['models'])!=10 or not report['original_primary_seed_retained']:
        raise ValueError('Actual complete original reserved evidence without reselection required')
    for relative,digest in report['source_evidence'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Actual reserved accuracy evidence changed')
    frozen_path=DOCS/'temporal-final-selection-01.json';frozen=read(frozen_path)
    original=read(ROOT/frozen['model_registry']);registry=copy.deepcopy(original)
    bank=BASE/'temporal-reserved-240-01/bank.json';scene_ids=[r['identifier']for r in read(bank)['trajectories']]
    evidence={str(report_path.relative_to(ROOT)):sha(report_path),str(frozen_path.relative_to(ROOT)):sha(frozen_path)}
    for method in registry['methods']:
        identifier=method['identifier'];accuracy=BASE/'temporal-reserved-evaluation-01'/identifier/'summary.json'
        method['original_role']=method['role'];method['role']='control'if method['model']['horizons']==1 else'reserved_characterization'
        method['accuracy_summary']=dict(path=str(accuracy.relative_to(ROOT)),sha256=sha(accuracy))
        method['actual_reserved_qualification']=require(method,sha(bank),scene_ids)
        evidence[str(accuracy.relative_to(ROOT))]=sha(accuracy)
        wrapper=accuracy.parent/'wrapper-receipt.json';value=read(wrapper)
        if value['core_summary_sha256']!=sha(accuracy)or not value['accuracy_recomputed_for_actual_deployment']:
            raise ValueError('Actual complete fresh policy evaluation required')
        evidence[str(wrapper.relative_to(ROOT))]=sha(wrapper)
    registry['failures_retained_without_finalist_reselection']=True
    path=BASE/'temporal-finalists-01/reserved-qualified-models.json'
    if path.exists():raise ValueError('Fresh actual reserved timing registry required')
    write(path,registry)
    sources=dict(frozen['source_sha256'])
    for name in ('register_reserved_temporal_timing.py','benchmark_reserved_temporal.py','report_reserved_temporal_cost.py'):
        if sources.get(name)!=sha(HERE/name):raise ValueError('Pre-capture frozen measurement source changed')
    for name,digest in sources.items():
        if sha(HERE/name)!=digest:raise ValueError('Pre-capture frozen source changed')
    selected=read(DOCS/'fixed-viscosity-replication-selection-01.json')['recipe_horizons']
    extra=sorted({2,3,4}-set(selected))
    write(output,dict(status='registered_before_reserved_CPU_measurement',models=[m['identifier']for m in registry['methods']],
        bundled_horizons=selected,extra_primary_sequential_horizons=extra,
        source_sha256=sha(HERE/'benchmark_reserved_temporal.py'),loader_sha256=sources,
        execution_assets_sha256=frozen['execution_assets_sha256'],bank_sha256=sha(bank),
        model_registry_sha256=sha(path),accuracy_evidence_sha256=evidence,
        cpu=23,threads=1,batch_size=1,repeats=5,balanced_serial=True,full_and_tensor_separate=True,
        primary_seed_reselected=False,all_added_seeds_retained=True,failed_accuracy_models_retained=True,
        reserved_scenes_consumed=True,training_on_reserved=False,optimizer_updates=0))
    write(DOCS/'temporal-final-reserved-timing-batch-specification-01.json',dict(commands=[
        dict(id='five-balanced-all-frozen-models-reserved-CPU-repeats',runtime='torch',timeout=7800,
            arguments=['experiments/full-ai-next-priorities-v1/benchmark_reserved_temporal.py',
                '--registry',str(path.relative_to(ROOT)),'--bank',str(bank.relative_to(ROOT)),
                '--output','runs/full-ai-next-priorities-v1/temporal-reserved-timing-01']),
        dict(id='publish-all-frozen-models-reserved-errors-and-costs',runtime='torch',timeout=180,
            arguments=['experiments/full-ai-next-priorities-v1/report_reserved_temporal_cost.py'])]))
    print('Registered all10frozen models and original reserved scenes for fresh5-repeat CPU timing',flush=True)


if __name__=='__main__':run()
