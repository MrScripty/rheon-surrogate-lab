"""Freeze the corrected K2 arithmetic comparison before evaluating its new outputs."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=BASE/'fixed-viscosity-k2-precision-inputs-01'
    if output.exists():raise ValueError('Fresh corrected K2 precision inputs required')
    for name in ('fixed-viscosity-postprocessing-batch-01','float32-map-publication-batch-01'):
        value=read(BASE/name/'execution.json')
        if value['status']!='completed'or not value['owned_child_reaped']:
            raise ValueError('Previous isolated numerical campaigns must finish and reap')
    old=read(DOCS/'float32-learned-map-protocol-01.json')
    sources={name:digest for name,digest in old['source_sha256'].items()
             if name not in ('float32_map_model_support.py','evaluate_float32_map.py','report_float32_map.py')}
    primary=read(DOCS/'fixed-viscosity-fit-registration-01.json')
    sources.update(primary['source_sha256'])
    for name in ('register_fixed_viscosity_k2_precision.py','application_model_support.py',
                 'fixed_viscosity_evaluation_support.py','evaluate_application_policy.py'):
        sources[name]=sha(HERE/name)
    for name,digest in sources.items():
        if sha(HERE/name)!=digest:raise ValueError('Original qualified arithmetic changed')
    for path,digest in old['execution_assets_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Qualified CPU execution asset changed')
    output.mkdir(parents=True);directory=BASE/'b2-fixed-viscosity-pair-01';models=[]
    for policy in ('fast-float64','fast-float32-map'):
        spec=dict(identifier='b2-fixed-viscosity-'+policy,family='fixed-viscosity',
            directory=str(directory.relative_to(ROOT)),receipt_sha256=sha(directory/'receipt.json'),
            weights_sha256=sha(directory/'restore/weights.npz'),
            normalization_sha256=sha(directory/'restore/normalization.json'),horizons=2,policy=policy)
        path=output/(spec['identifier']+'.json');write(path,spec);models.append(spec)
    write(output/'models.json',dict(models=models,weights_identical=True,optimizer_updates=0))
    spec=output/'b2-fixed-viscosity-fast-float32-map.json'
    protocol=DOCS/'fixed-viscosity-k2-float32-evaluation-protocol-01.json'
    if protocol.exists():raise ValueError('Fresh precision evaluation protocol required')
    write(protocol,dict(status='registered_before_actual_corrected_K2_float32_evaluation',
        source_sha256=sources,execution_assets_sha256=old['execution_assets_sha256'],
        model_specification_sha256=sha(spec),model_registry_sha256=sha(output/'models.json'),
        bank_sha256=sha(BASE/'temporal-development-240-01/bank.json'),
        original_accuracy_rules_unchanged=True,every_intermediate_output_required=True,
        own_history_steps=240,accuracy_limit=.004,optimizer_updates=0,reserved_scenes_consumed=False,
        preceding_paired_cost_sha256=sha(DOCS/'float32-map-cost-results.json'),
        family_cost_sha256=sha(DOCS/'fixed-viscosity-cost-results.json'),
        future_reference_inputs=False,physical_output_equivalence_inherited=False))
    command=dict(id='actual-corrected-K2-float32-every-output-and-own240',runtime='torch',timeout=1800,
        arguments=['experiments/full-ai-next-priorities-v1/evaluate_application_policy.py',
            '--specification',str(spec.relative_to(ROOT)),'--protocol',str(protocol.relative_to(ROOT)),
            '--bank','runs/full-ai-next-priorities-v1/temporal-development-240-01/bank.json',
            '--output','runs/full-ai-next-priorities-v1/fixed-viscosity-k2-float32-evaluation-01'])
    write(DOCS/'fixed-viscosity-k2-precision-evaluation-batch-specification-01.json',dict(commands=[command]))
    print('Registered exact corrected K2float32 evaluation; no fitting or reserved scenes',flush=True)


if __name__=='__main__':run()
