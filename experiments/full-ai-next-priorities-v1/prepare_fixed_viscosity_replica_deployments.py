"""Freeze actual primary/replica deployments after the six managed fits finish."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=BASE/'fixed-viscosity-replication-deployments-01'
    if output.exists():raise ValueError('Fresh concrete replica deployments required')
    batch_path=BASE/'fixed-viscosity-replication-training-batch-01/execution.json';batch=read(batch_path)
    if batch['status']!='completed'or not batch['owned_child_reaped']:
        raise ValueError('Six managed fits/restores/evaluations and final audit must complete and reap')
    report_path=DOCS/'fixed-viscosity-replication-results.json';report=read(report_path)
    if report['status']!='completed':raise ValueError('Completed actual replica custody/work audit required')
    for relative,digest in report['source_evidence'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Actual replica audit evidence changed')
    selection_path=DOCS/'fixed-viscosity-replication-selection-01.json';selection=read(selection_path)
    inputs=read(BASE/'fixed-viscosity-replication-inputs-01/models.json')
    entries=[dict(identifier=f'b{k}-fixed-viscosity',horizons=k,replication_ordinal=1)
             for k in (1,*selection['recipe_horizons'])]+inputs['models']
    output.mkdir(parents=True);methods=[]
    for entry in entries:
        identifier=entry['identifier'];k=entry['horizons'];ordinal=entry['replication_ordinal']
        pair=BASE/(identifier+'-pair-01');receipt=read(pair/'receipt.json')
        policy=selection['execution_policy_by_horizon'][str(k)]
        deployment_id=identifier+'-'+policy
        spec=dict(identifier=deployment_id,family='fixed-viscosity',directory=str(pair.relative_to(ROOT)),
            receipt_sha256=sha(pair/'receipt.json'),weights_sha256=sha(pair/'restore/weights.npz'),
            normalization_sha256=sha(pair/'restore/normalization.json'),horizons=k,policy=policy)
        original=BASE/('fixed-viscosity-evaluation-01'if ordinal==1 else'fixed-viscosity-replication-evaluation-01')/identifier/'summary.json'
        accuracy=original
        if policy=='fast-float32-map':
            accuracy=BASE/('fixed-viscosity-k2-float32-evaluation-01'if ordinal==1 else'fixed-viscosity-replication-policy-evaluation-01/'+identifier)/'summary.json'
        proof=BASE/(f'fixed-viscosity-execution-qualification-k{k}-01.json'if ordinal==1 else'fixed-viscosity-replication-execution-01/'+identifier+'.json')
        spec_path=output/(deployment_id+'.json');write(spec_path,spec)
        method=dict(identifier=deployment_id,kind='forecast',model=spec,replication_ordinal=ordinal,
            role='control'if k==1 else'candidate'if ordinal==1 else'replica_characterization',
            specification_path=str(spec_path.relative_to(ROOT)),specification_sha256=sha(spec_path),
            original_accuracy_summary=dict(path=str(original.relative_to(ROOT)),sha256=sha(original)),
            accuracy_summary=dict(path=str(accuracy.relative_to(ROOT))),execution_qualification=dict(path=str(proof.relative_to(ROOT))))
        if accuracy.exists():method['accuracy_summary']['sha256']=sha(accuracy)
        if proof.exists():method['execution_qualification']['sha256']=sha(proof)
        methods.append(method)
    write(output/'models.json',dict(methods=methods,original_primary_seed_retained=True,
        all_additional_seeds_retained_including_failures=True,shared_component_pretraining=True,
        selection_sha256=sha(selection_path),replica_report_sha256=sha(report_path),training_batch_sha256=sha(batch_path)))
    old=read(DOCS/'fixed-viscosity-execution-protocol-01.json');sources=dict(old['source_sha256'])
    precision=read(DOCS/'fixed-viscosity-k2-float32-evaluation-protocol-01.json');sources.update(precision['source_sha256'])
    for name in ('prepare_fixed_viscosity_replica_deployments.py','qualify_fixed_viscosity_replica_execution.py',
                 'application_model_evidence.py'):
        sources[name]=sha(HERE/name)
    for name,digest in sources.items():
        if sha(HERE/name)!=digest:raise ValueError('Qualified inference/evaluation source changed')
    assets=old['execution_assets_sha256']
    for relative,digest in assets.items():
        if sha(ROOT/relative)!=digest:raise ValueError('Qualified execution asset changed')
    execution_path=DOCS/'fixed-viscosity-replication-execution-protocol-01.json'
    if execution_path.exists():raise ValueError('Fresh replica execution protocol required')
    write(execution_path,dict(status='registered_before_replica_execution_evaluations',source_sha256=sources,
        execution_assets_sha256=assets,fit_registration_sha256=sha(DOCS/'fixed-viscosity-replication-fit-registration-01.json'),
        model_registry_sha256=sha(output/'models.json'),bank_sha256=old['bank_sha256'],
        every_intermediate_output_required=True,own_history_steps=240,optimizer_updates=0,reserved_scenes_consumed=False))
    commands=[]
    for method in methods:
        if method['replication_ordinal']==1:continue
        spec=method['model'];identifier=spec['identifier']
        if spec['policy']=='fast-float64':
            commands.append(dict(id=identifier+'-exact-physical-execution',runtime='torch',timeout=1800,
                arguments=['experiments/full-ai-next-priorities-v1/qualify_fixed_viscosity_replica_execution.py',
                    '--identifier',identifier,'--output',method['execution_qualification']['path']]))
        else:
            protocol_path=DOCS/(identifier+'-evaluation-protocol-01.json')
            write(protocol_path,dict(status='registered_before_actual_replica_float32_accuracy',source_sha256=sources,
                execution_assets_sha256=assets,model_specification_sha256=method['specification_sha256'],
                bank_sha256=old['bank_sha256'],original_accuracy_rules_unchanged=True,
                every_intermediate_output_required=True,own_history_steps=240,optimizer_updates=0,reserved_scenes_consumed=False))
            commands.append(dict(id=identifier+'-actual-float32-every-output-own240',runtime='torch',timeout=1800,
                arguments=['experiments/full-ai-next-priorities-v1/evaluate_application_policy.py',
                    '--specification',method['specification_path'],'--protocol',str(protocol_path.relative_to(ROOT)),
                    '--bank','runs/full-ai-next-priorities-v1/temporal-development-240-01/bank.json',
                    '--output',str((ROOT/method['accuracy_summary']['path']).parent.relative_to(ROOT))]))
    batch_spec=DOCS/'fixed-viscosity-replication-execution-batch-specification-01.json'
    if batch_spec.exists():raise ValueError('Fresh concrete replica execution list required')
    write(batch_spec,dict(commands=commands))
    print('Frozen9actual deployments and6new complete execution/precision qualifications; primary unchanged',flush=True)


if __name__=='__main__':run()
