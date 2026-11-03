"""Freeze exact selected-seed constructors before qualification and new fits."""
from study_support import HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'fixed-viscosity-replication-qualification-protocol-01.json'
    if output.exists():raise ValueError('Fresh replica constructor registration required')
    inputs=BASE/'fixed-viscosity-replication-inputs-01';prepared=read(inputs/'receipt.json');registry=read(inputs/'models.json')
    if prepared['status']!='prepared'or prepared['optimizer_updates']!=0 or prepared['training_admitted']:
        raise ValueError('Only prepared, untrained replica inputs may be registered')
    if sha(inputs/'models.json')!=prepared['models_sha256']:raise ValueError('Exact selected replica registry changed')
    protocol=read(DOCS/'fixed-viscosity-qualification-protocol-01.json')
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Original qualified training implementation changed')
    sources=dict(protocol['source_sha256'])
    for name in ('prepare_fixed_viscosity_replicas.py','qualify_fixed_viscosity_replicas.py',
                 'register_fixed_viscosity_replica_qualification.py'):
        sources[name]=sha(HERE/name)
    protocol.update(status='registered_before_additional_seed_qualification_and_training',source_sha256=sources,
        inputs_receipt_sha256=sha(inputs/'receipt.json'),model_registry_sha256=sha(inputs/'models.json'),
        models=registry['models'],qualification_cases_per_device=len(registry['models'])*16,
        selected_bundled_horizons=registry['selected_bundled_horizons'],one_step_control_included=True,
        initial_and_sampling_seeds_vary=True,independent_component_pretraining=False,optimizer_updates_executed=0)
    write(output,protocol)
    print('Registered actual selected-seed constructor cases:',protocol['qualification_cases_per_device'],'perdevice; zero updates',flush=True)


if __name__=='__main__':run()
