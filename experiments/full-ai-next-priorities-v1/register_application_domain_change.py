"""Freeze the supported boundary test without changing the registered campaigns."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'application-domain-change-reference-protocol-01.json'
    if output.exists():raise ValueError('Fresh supported-boundary reference protocol required')
    owner=read(BASE/'temporal-final-confirmation-batch-01/execution.json')
    if owner['status']!='completed'or not owner['owned_child_reaped']:
        raise ValueError('Existing final source/weight freeze and all CPU campaigns must complete/reap first')
    old=read(DOCS/'temporal-final-application-protocol-01.json');sources=dict(old['source_sha256'])
    for name,digest in sources.items():
        if sha(HERE/name)!=digest:raise ValueError('Original qualified model or application source changed')
    for name in ('await_final_application.py','register_application_domain_change.py',
                 'register_application_domain_change_qualification.py','capture_application_domain_change.py',
                 'forecast_domain_change_episode.py','qualify_application_domain_change.py',
                 'report_application_domain_change.py'):
        sources[name]=sha(HERE/name)
    assets=dict(old['execution_assets_sha256'])
    assets['experiments/pressure-methods-v1/numerics.py']=sha(ROOT/'experiments/pressure-methods-v1/numerics.py')
    for relative,digest in assets.items():
        if sha(ROOT/relative)!=digest:raise ValueError('Qualified native/compiled execution asset changed')
    bank=BASE/'temporal-development-240-01/bank.json'
    registry=BASE/'temporal-finalists-01/application-models.json'
    write(output,dict(status='registered_before_supported_boundary_reference_capture',
        source_sha256=sources,execution_assets_sha256=assets,bank_sha256=sha(bank),
        registry_sha256=sha(registry),scene_identifiers=old['scene_identifiers'],
        event_step=13,steps=48,cells_per_metre_before_and_after=[8,16],
        semantics='Instantaneous physical closed-box rescale; logical cell counts and uploaded concentration/velocity arrays unchanged; fluid density stays1000kg/m3',
        field_resampling=False,closed_boundary_type_unchanged=True,
        external_volume_mass_change_accounted_separately=True,
        unsupported_boundary_types_retain_explicit_stop_guard=True,
        training_material=False,performance_claim=False,optimizer_updates=0,reserved_scenes_consumed=False))
    print('Registered supported physical-box change at13; existing sources/checkpoints/policies unchanged',flush=True)


if __name__=='__main__':run()
