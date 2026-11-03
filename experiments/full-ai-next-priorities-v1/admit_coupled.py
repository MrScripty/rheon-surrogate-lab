"""Admit a bounded causal coupled fit after actual constructor qualification."""
from study_support import HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'temporal-coupled-admission-01.json'
    if output.exists():raise ValueError('Fresh coupled admission required')
    prior=read(DOCS/'temporal-decoder-results.json')
    if prior['status']!='completed'or prior['accuracy_qualified_small_grid_recipe']:raise ValueError('Completed prior full-state screen required')
    for device in ('cpu','cuda'):
        component=read(BASE/f'full-learned-pressure-map-{device}-qualification-01.json')
        if component['status']!='passed'or component['optimizer_updates']!=0:raise ValueError('Learned pressure map not qualified')
        for name,digest in component['source_sha256'].items():
            if sha(HERE/name)!=digest:raise ValueError('Qualified pressure component changed')
    audit=read(BASE/'full-learned-pressure-map-operator-audit-01.json')
    if audit['status']!='completed'or audit['observed_growing_residual_modes']:raise ValueError('Residual operator observations do not admit this fit')
    primitive=read(BASE/'tied-momentum-qualification-01.json')
    if primitive['status']!='passed_offline_learned_readout_qualification'or primitive['primitive_maximum_native_relative_difference']>1e-6:
        raise ValueError('Learned flux/momentum sharing primitive not qualified')
    evaluator_path=BASE/'temporal-coupled-evaluator-qualification-01/receipt.json'
    evaluator=read(evaluator_path)
    if evaluator['status']!='passed'or not evaluator['all_eight_finite_scenes_metrics_bitwise_equal']or not evaluator['no_fabricated_outputs_or_fallback']:
        raise ValueError('Qualified complete-output and raw-failure evaluator required')
    if evaluator['evaluator_sha256']!=sha(HERE/'evaluate_coupled.py')or evaluator['protocol_sha256']!=sha(DOCS/'temporal-coupled-evaluation-protocol-01.json'):
        raise ValueError('Qualified evaluator or protocol changed')
    bindings={}
    for device in ('cpu','cuda'):
        path=BASE/f'temporal-coupled-trainer-{device}-qualification-01.json';qualified=read(path)
        if qualified['status']!='passed'or len(qualified['rows'])!=64 or qualified['optimizer_updates']!=0:
            raise ValueError('Complete actual-constructor coupled qualification required')
        if not qualified['actual_trainer_constructor_exercised']or not qualified['actual_objective_constructor_exercised']:
            raise ValueError('Actual managed constructors must qualify')
        for name,digest in qualified['source_sha256'].items():
            if sha(HERE/name)!=digest:raise ValueError('Qualified coupled source changed')
        if qualified['registration_sha256']!=sha(DOCS/'temporal-coupled-registration-01.json'):raise ValueError('Registration changed')
        bindings[device]=sha(path)
    write(output,dict(status='admitted',model='b2-coupled',updates=10000,
        registration_sha256=sha(DOCS/'temporal-coupled-registration-01.json'),
        prior_comparison_sha256=sha(DOCS/'temporal-decoder-results.json'),qualification_sha256=bindings,
        pressure_operator_observations_sha256=sha(BASE/'full-learned-pressure-map-operator-audit-01.json'),
        tied_momentum_qualification_sha256=sha(BASE/'tied-momentum-qualification-01.json'),
        evaluator_qualification_sha256=sha(evaluator_path),
        actual_inherited_training='Frozen pressure network and1000-update learned flux calibration; nonlinear request parameters unused',
        reason='Couple learned provisional transport, flux, pressure and momentum while retaining shared encoding and correct tracer staging',
        universal_stability_guarantee=False,single_factor_attribution=False,reserved_capture_admitted=False,optimizer_updates_executed=0,
        source_sha256=sha(__file__),original_goal_complete=False))
    print('Admitted one bounded K2 coupled full-state fit; all prerequisite numerical and gradient checks passed',flush=True)


if __name__=='__main__':run()
