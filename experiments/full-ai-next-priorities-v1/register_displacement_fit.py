"""Freeze one bounded full-state fit after actual CPU/CUDA qualifications pass."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'temporal-displacement-fit-registration-01.json'
    if output.exists():raise ValueError('Fresh displacement fit registration required')
    protocol_path=DOCS/'temporal-displacement-qualification-protocol-01.json';protocol=read(protocol_path)
    bindings={}
    for device in ('cpu','cuda'):
        path=BASE/f'temporal-displacement-{device}-qualification-01.json';qualified=read(path)
        if qualified['status']!='passed'or len(qualified['rows'])!=64 or qualified['optimizer_updates']!=0:
            raise ValueError('Complete zero-update actual constructor qualification required')
        if not qualified['actual_trainer_constructor_exercised']or not qualified['actual_objective_constructor_exercised']:
            raise ValueError('Actual training path must qualify')
        if qualified['protocol_sha256']!=sha(protocol_path)or qualified['source_sha256']!=protocol['source_sha256']:
            raise ValueError('Qualified protocol or implementation changed')
        for name,digest in qualified['source_sha256'].items():
            if sha(HERE/name)!=digest:raise ValueError('Qualified source changed')
        if not all(r['deterministic_two_block_gradients']and r['training_teacher_disabled_during_inference']for r in qualified['rows']):
            raise ValueError('Own-history gradients and inference isolation required')
        for k in (1,2,3,4):
            if sum(r['horizons']==k for r in qualified['rows'])!=16 or not qualified['aggregate'][str(k)]['all_trainable_tensors_active']:
                raise ValueError('Every K requires complete gradient and staging checks')
        bindings[device]=sha(path)
    component=read(DOCS/'point-sampling-results.json')
    if component['status']!='completed_component_calibration'or component['native_cases']!=24:
        raise ValueError('Managed native-qualified float32 component required')
    for name,digest in component['source_evidence'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Point component evidence changed')
    evaluator=read(BASE/'temporal-coupled-evaluator-qualification-01/receipt.json')
    if evaluator['status']!='passed'or evaluator['evaluator_sha256']!=sha(HERE/'evaluate_coupled.py'):
        raise ValueError('Unchanged qualified evaluation core required')
    evaluation_protocol=read(DOCS/'temporal-displacement-evaluation-protocol-01.json')
    for name,digest in evaluation_protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered evaluation wrapper or loader changed')
    config=BASE/'temporal-displacement-inputs-01/k2-config.json';settings=read(config)
    if settings['updates']!=10000 or settings['single_block_updates']!=2000 or settings['displacement_auxiliary_weight']!=1.:
        raise ValueError('Registered bounded work and objective required')
    sources=dict(protocol['source_sha256'])
    for name in ('register_displacement_fit.py','prepare_displacement_project.py','run_displacement_pair.py','evaluate_displacement.py','displacement_evaluation_support.py'):
        sources[name]=sha(HERE/name)
    write(output,dict(status='registered_before_full_state_optimizer_updates',model='b2-displacement',updates=10000,
        config_sha256=sha(config),qualification_sha256=bindings,source_sha256=sources,
        constructor_protocol_sha256=sha(protocol_path),point_component_results_sha256=sha(DOCS/'point-sampling-results.json'),
        prior_matched_negative_result_sha256=sha(DOCS/'temporal-provisional-results.json'),
        evaluation_protocol_sha256=sha(DOCS/'temporal-displacement-evaluation-protocol-01.json'),
        evaluation_implementation_sha256=sha(HERE/'evaluate_coupled.py'),
        finite_batch_specification_sha256=sha(DOCS/'temporal-displacement-batch-specification-01.json'),
        original_plan_sha256=sha(DOCS/'plan.md'),original_registration_sha256=sha(DOCS/'registration.json'),
        every_intermediate_P_V_T_required=True,own_history_steps=240,accuracy_limit=.004,
        complete_block_budget_ms=4,no_individual_stage_cap=True,zero_reference_future_inference_inputs=True,
        reserved_scenes_consumed=False,single_factor_attribution=False,optimizer_updates_executed=0,goal_complete=False))
    print('Registered one 10000-update K2 full-state displacement fit after CPU/CUDA qualification',flush=True)


if __name__=='__main__':run()
