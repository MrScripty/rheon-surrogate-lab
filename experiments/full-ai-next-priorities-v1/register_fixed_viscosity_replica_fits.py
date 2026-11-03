"""Freeze finite replica fits only after exact CPU/CUDA constructor checks."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'fixed-viscosity-replication-fit-registration-01.json'
    if output.exists():raise ValueError('Fresh replica fit registration required')
    inputs=BASE/'fixed-viscosity-replication-inputs-01';registry=read(inputs/'models.json');prepared=read(inputs/'receipt.json')
    protocol_path=DOCS/'fixed-viscosity-replication-qualification-protocol-01.json';protocol=read(protocol_path)
    if sha(inputs/'models.json')!=protocol['model_registry_sha256']or sha(inputs/'receipt.json')!=protocol['inputs_receipt_sha256']:
        raise ValueError('Selected replication inputs changed')
    bindings={};expected_cases=len(registry['models'])*16
    for device in ('cpu','cuda'):
        path=BASE/f'fixed-viscosity-replication-{device}-qualification-01.json';value=read(path)
        if value['status']!='passed'or len(value['rows'])!=expected_cases or value['optimizer_updates']!=0:
            raise ValueError('All actual replica constructor cases must pass')
        if value['protocol_sha256']!=sha(protocol_path)or value['source_sha256']!=protocol['source_sha256']:
            raise ValueError('Qualified replication protocol/source changed')
        for entry in registry['models']:
            identifier=entry['identifier'];rows=[r for r in value['rows']if r['identifier']==identifier]
            if len(rows)!=16 or not all(r['deterministic_two_block_gradients']and r['training_teacher_disabled_during_inference']for r in rows):
                raise ValueError('Every selected replica needs complete repeated two-block gradients')
            aggregate=value['aggregate'][identifier]
            if not aggregate['all_trainable_tensors_active']or aggregate['trainable_parameters']!=6600:
                raise ValueError('Exact fixed-viscosity trainable inventory required')
        bindings[device]=sha(path)
    sources=dict(protocol['source_sha256'])
    for name in ('register_fixed_viscosity_replica_fits.py','prepare_fixed_viscosity_replica_project.py',
                 'run_fixed_viscosity_replica_pair.py','evaluate_fixed_viscosity_replica.py','fixed_viscosity_evaluation_support.py',
                 'prepare_fixed_viscosity_replica_batch.py','report_fixed_viscosity_replicas.py'):
        sources[name]=sha(HERE/name)
    for name,digest in sources.items():
        if sha(HERE/name)!=digest:raise ValueError('Qualified replica source changed')
    write(output,dict(status='registered_before_additional_seed_optimizer_updates',models=[m['identifier']for m in registry['models']],
        model_registry_sha256=sha(inputs/'models.json'),config_sha256=prepared['config_sha256'],
        qualification_sha256=bindings,qualification_cases_per_device=expected_cases,source_sha256=sources,
        constructor_protocol_sha256=sha(protocol_path),selection_sha256=registry['selection_sha256'],
        bank_sha256=sha(BASE/'temporal-development-240-01/bank.json'),evaluation_implementation_sha256=sha(HERE/'evaluate_coupled.py'),
        shared_frozen_components=True,independent_component_pretraining=False,physical_target_states_per_fit=36000,
        original_plan_sha256=sha(DOCS/'plan.md'),original_registration_sha256=sha(DOCS/'registration.json'),
        every_intermediate_P_V_T_required=True,own_history_steps=240,accuracy_limit=.004,
        primary_seed_reselected=False,reserved_scenes_consumed=False,optimizer_updates_executed=0,original_goal_complete=False))
    print('Registered ordinary managed additional-seed fits:',len(registry['models']),flush=True)


if __name__=='__main__':run()
