"""Freeze four finite managed fits only after actual CPU/CUDA qualification."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'fixed-viscosity-fit-registration-01.json'
    if output.exists():raise ValueError('Fresh fixed-viscosity fit registration required')
    protocol_path=DOCS/'fixed-viscosity-qualification-protocol-01.json';protocol=read(protocol_path)
    bindings={}
    for device in ('cpu','cuda'):
        path=BASE/f'fixed-viscosity-{device}-qualification-01.json';qualified=read(path)
        if qualified['status']!='passed'or len(qualified['rows'])!=64 or qualified['optimizer_updates']!=0:
            raise ValueError('Complete actual constructor qualification required')
        if not qualified['actual_trainer_constructor_exercised']or not qualified['actual_objective_constructor_exercised']:
            raise ValueError('Actual trainer/objective must qualify')
        if qualified['protocol_sha256']!=sha(protocol_path)or qualified['source_sha256']!=protocol['source_sha256']:
            raise ValueError('Qualified protocol changed')
        for name,digest in qualified['source_sha256'].items():
            if sha(HERE/name)!=digest:raise ValueError('Qualified source changed')
        for k in (1,2,3,4):
            if sum(r['horizons']==k for r in qualified['rows'])!=16:
                raise ValueError('Every horizon and grid must qualify')
            q=qualified['aggregate'][str(k)]
            if not q['all_trainable_tensors_active']or q['trainable_parameters']!=6600:
                raise ValueError('Exact fixed-viscosity gradient inventory required')
        bindings[device]=sha(path)
    evaluator=read(BASE/'temporal-coupled-evaluator-qualification-01/receipt.json')
    if evaluator['status']!='passed'or evaluator['evaluator_sha256']!=sha(HERE/'evaluate_coupled.py'):
        raise ValueError('Unchanged complete evaluation core required')
    inputs=BASE/'fixed-viscosity-inputs-01';prepared=read(inputs/'receipt.json')
    if sha(inputs/'receipt.json')!=protocol['inputs_receipt_sha256']:
        raise ValueError('Prepared data/config work changed')
    configs={}
    for k in (1,2,3,4):
        path=inputs/f'k{k}-config.json';config=read(path);configs[str(k)]=sha(path)
        if configs[str(k)]!=prepared['config_sha256'][str(k)]or k*(2*config['updates']-config['single_block_updates'])!=36000:
            raise ValueError('Fixed bounded physical-target work required')
    sources=dict(protocol['source_sha256'])
    for name in ('register_fixed_viscosity_fits.py','prepare_fixed_viscosity_project.py','run_fixed_viscosity_pair.py',
                 'evaluate_fixed_viscosity.py','fixed_viscosity_evaluation_support.py'):
        sources[name]=sha(HERE/name)
    write(output,dict(status='registered_before_new_full_state_optimizer_updates',models=[f'b{k}-fixed-viscosity'for k in (1,2,3,4)],
        config_sha256=configs,qualification_sha256=bindings,source_sha256=sources,work=prepared['work'],
        constructor_protocol_sha256=sha(protocol_path),inputs_receipt_sha256=sha(inputs/'receipt.json'),
        bank_sha256=sha(BASE/'temporal-development-240-01/bank.json'),
        evaluation_implementation_sha256=sha(HERE/'evaluate_coupled.py'),
        finite_batch_specification_sha256=sha(DOCS/'fixed-viscosity-training-batch-specification-01.json'),
        original_plan_sha256=sha(DOCS/'plan.md'),original_registration_sha256=sha(DOCS/'registration.json'),
        managed_CUDA_wall_limit_seconds=1800,physical_target_states_per_fit=36000,matched_FLOPs=False,
        viscosity_calibration_results_sha256=sha(DOCS/'viscosity-calibration-results.json'),
        viscosity_frozen_from_initialization=True,every_intermediate_P_V_T_required=True,own_history_steps=240,
        accuracy_limit=.004,complete_block_budget_ms={'2':4,'3':6,'4':8},no_individual_stage_cap=True,
        reserved_scenes_consumed=False,optimizer_updates_executed=0,goal_complete=False))
    print('Registered four bounded fixed-viscosity fits after complete CPU/CUDA checks',flush=True)


if __name__=='__main__':run()
