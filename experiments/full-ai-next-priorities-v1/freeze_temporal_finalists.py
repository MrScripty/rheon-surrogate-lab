"""Freeze all four primary horizons and all selected recipe seeds before reserved data."""
import copy
from application_model_evidence import require
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=BASE/'temporal-finalists-01'
    if output.exists():raise ValueError('Fresh final temporal freeze required')
    owner_path=BASE/'fixed-viscosity-replication-postprocessing-batch-01/execution.json';owner=read(owner_path)
    if owner['status']!='completed'or not owner['owned_child_reaped']:
        raise ValueError('All replica fits, exact restores, actual policy proofs and CPU costs must complete/reap')
    cost_path=DOCS/'fixed-viscosity-replication-cost-results.json';cost=read(cost_path)
    if cost['status']!='completed'or cost['primary_seed_reselected']or not cost['all_added_seeds_retained']:
        raise ValueError('Complete actual all-seed cost evidence without primary reselection required')
    for relative,digest in cost['source_evidence'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Actual primary/replica cost evidence changed')
    registry_path=BASE/'fixed-viscosity-replication-deployments-01/qualified-models.json'
    registry=read(registry_path);methods=copy.deepcopy(registry['methods'])
    primaries=[m for m in methods if m['replication_ordinal']==1]
    existing={m['model']['horizons']for m in primaries}
    for k in {1,2,3,4}-existing:
        identifier=f'b{k}-fixed-viscosity';pair=BASE/(identifier+'-pair-01')
        policy='fast-float64';deployment_id=identifier+'-'+policy
        accuracy=BASE/'fixed-viscosity-evaluation-01'/identifier/'summary.json'
        proof=BASE/f'fixed-viscosity-execution-qualification-k{k}-01.json'
        model=dict(identifier=deployment_id,family='fixed-viscosity',directory=str(pair.relative_to(ROOT)),
            receipt_sha256=sha(pair/'receipt.json'),weights_sha256=sha(pair/'restore/weights.npz'),
            normalization_sha256=sha(pair/'restore/normalization.json'),horizons=k,policy=policy)
        methods.append(dict(identifier=deployment_id,kind='forecast',model=model,replication_ordinal=1,
            role='control'if k==1 else'candidate',accuracy_summary=dict(path=str(accuracy.relative_to(ROOT)),sha256=sha(accuracy)),
            execution_qualification=dict(path=str(proof.relative_to(ROOT)),sha256=sha(proof))))
    if len(methods)!=10 or len({m['identifier']for m in methods})!=10:
        raise ValueError('Four original primary horizons and six added seeds must all be retained')
    bank=BASE/'temporal-development-240-01/bank.json';scene_ids=[r['identifier']for r in read(bank)['trajectories']]
    output.mkdir(parents=True);evidence={str(p.relative_to(ROOT)):sha(p)for p in
        (owner_path,cost_path,registry_path,DOCS/'fixed-viscosity-replication-selection-01.json',
         DOCS/'fixed-viscosity-cost-results.json',DOCS/'fixed-viscosity-k2-precision-results.json')}
    for method in methods:
        method['pre_freeze_development_qualification']=require(method,sha(bank),scene_ids)
        model=method['model'];pair=ROOT/model['directory']
        for path in (pair/'receipt.json',pair/'restore/weights.npz',pair/'restore/normalization.json',ROOT/method['accuracy_summary']['path']):
            evidence[str(path.relative_to(ROOT))]=sha(path)
        if 'execution_qualification'in method:
            path=ROOT/method['execution_qualification']['path'];evidence[str(path.relative_to(ROOT))]=sha(path)
        spec_path=output/(method['identifier']+'.json');write(spec_path,model)
        method.update(specification_path=str(spec_path.relative_to(ROOT)),specification_sha256=sha(spec_path))
    write(output/'models.json',dict(methods=methods,all_requested_K_retained=True,all_added_seeds_retained=True,
        original_primary_seed_retained=True,shared_component_pretraining=True,independent_component_pretraining=False))
    protocols=('fixed-viscosity-replication-execution-protocol-01.json','application-deployment-interface-protocol-01.json',
        'native-application-controls-protocol-01.json','bounded-native-client-protocol-01.json')
    sources={};assets={}
    for name in protocols:
        path=DOCS/name;value=read(path);evidence[str(path.relative_to(ROOT))]=sha(path)
        sources.update(value['source_sha256']);assets.update(value.get('execution_assets_sha256',{}))
    for name in ('freeze_temporal_finalists.py','capture_reserved_temporal.py','evaluate_application_policy.py',
                 'application_model_evidence.py','benchmark_forecast_application.py','report_forecast_application.py',
                 'prepare_reserved_temporal_evaluations.py','report_reserved_temporal.py',
                 'register_reserved_temporal_timing.py','benchmark_reserved_temporal.py','report_reserved_temporal_cost.py'):
        sources[name]=sha(HERE/name)
    for name,digest in sources.items():
        if sha(HERE/name)!=digest:raise ValueError('Pre-freeze qualified source changed')
    for relative,digest in assets.items():
        if sha(ROOT/relative)!=digest:raise ValueError('Pre-freeze qualified execution asset changed')
    frozen=DOCS/'temporal-final-selection-01.json'
    if frozen.exists():raise ValueError('Fresh final selection document required')
    evidence[str((output/'models.json').relative_to(ROOT))]=sha(output/'models.json')
    write(frozen,dict(status='frozen_before_original_reserved_temporal_capture',
        model_registry=str((output/'models.json').relative_to(ROOT)),model_registry_sha256=sha(output/'models.json'),
        source_sha256=sources,execution_assets_sha256=assets,source_evidence=evidence,
        original_scene_reservation_sha256=sha(DOCS/'temporal-scene-reservation.json'),
        all_primary_horizons=[1,2,3,4],primary_seed_reselected=False,all_added_seeds_retained=True,
        shared_component_pretraining=True,independent_component_pretraining=False,
        reserved_scenes_consumed=False,original_goal_complete=False))
    write(DOCS/'temporal-final-reserved-capture-protocol-01.json',dict(status='registered_before_reserved_capture',
        frozen_selection_sha256=sha(frozen),source_sha256=sources,execution_assets_sha256=assets,
        original_scene_reservation_sha256=sha(DOCS/'temporal-scene-reservation.json'),
        scenes=8,steps_per_scene=240,original_reference_gates_unchanged=True,training_material=False))
    primaries=sorted((m for m in methods if m['replication_ordinal']==1),key=lambda m:m['model']['horizons'])
    app_methods=copy.deepcopy(primaries)
    app_methods.extend(dict(identifier=name,kind='control',method=name,role='control')for name in ('native-pcg','spectral','C107'))
    app_registry=output/'application-models.json';write(app_registry,dict(methods=app_methods,primary_seed_reselected=False))
    normalization=BASE/'temporal-training-view-02/normalization.json'
    event_bank=BASE/'application-event-reference-01/bank.json'
    write(DOCS/'temporal-final-application-protocol-01.json',dict(status='registered_before_application_measurements',
        registry_sha256=sha(app_registry),source_sha256=sources,execution_assets_sha256=assets,
        banks=dict(steady=dict(path=str(bank.relative_to(ROOT)),sha256=sha(bank)),
                   forcing=dict(path=str(event_bank.relative_to(ROOT)),sha256=sha(event_bank))),
        normalization=dict(path=str(normalization.relative_to(ROOT)),sha256=sha(normalization)),scene_identifiers=scene_ids,
        primary_seed_reselected=False,all_four_actual_trained_horizons=True,steps=dict(steady=240,forcing=48),
        repeats=5,threads=1,cpu=23,batch_size=1,finite_campaign=True,native_gates_unchanged=True,
        candidate_native_steps=0,application_acceptance_limits='Diagnostic; original research criteria retained'))
    print('Frozen10actual checkpoints/policies,4primary horizons and7application methods before reserved capture',flush=True)


if __name__=='__main__':run()
