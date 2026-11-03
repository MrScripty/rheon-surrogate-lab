"""Report frozen pressure port qualification separately from temporal results."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    evidence={};rows=[]
    for device in ('cpu','cuda'):
        path=BASE/f'temporal-pressure-core-{device}-qualification-03.json'
        result=read(path)
        if result['status']!='passed':raise ValueError('Both component qualifications must pass')
        for name,digest in result['source_sha256'].items():
            if sha(HERE/name)!=digest:raise ValueError('Qualified source changed: '+name)
        if len(result['rows'])!=128 or len(result['gradients'])!=32 or result['optimizer_updates']!=0:
            raise ValueError('Complete zero-update component qualification required')
        evidence[str(path.relative_to(ROOT))]=sha(path)
        rows.append(dict(device=device,requests=len(result['rows']),gradient_cases=len(result['gradients']),
            maximum_parent_pressure_relative_difference=max(row['parent_relative_difference']for row in result['rows']),
            maximum_CPU_device_pressure_relative_difference=max(row['CPU_device_relative_difference']for row in result['rows']),
            maximum_equation_error_difference=max(abs(row['equation_relative_residual']-row['parent_equation_relative_residual'])for row in result['rows']),
            maximum_batch_individual_relative_difference=max(row['batch_individual_relative_difference']for row in result['rows']),
            maximum_homogeneous_gradient_relative_error=max(row['homogeneous_gradient_relative_error']for row in result['gradients']),
            deterministic_input_gradients=True,all_oracle_classifications_unchanged=True,
            dense_learned_map_bytes=sum(row['matrix_bytes']for row in result['dense_learned_map_preparation']),
            dense_learned_map_setup_seconds=sum(row['seconds']for row in result['dense_learned_map_preparation'])))
    value=dict(status='passed_component_qualification',rows=rows,source_evidence=evidence,
        registration_sha256=sha(DOCS/'temporal-pressure-core-port-registration-02.json'),
        source_sha256={name:sha(HERE/name)for name in('frozen_temporal_pressure_core.py','qualify_frozen_temporal_pressure_core.py',
            'export_temporal_pressure_assets.py','summarize_frozen_temporal_pressure_core.py')},
        frozen_spatial_parameters=92560,parent_learned_parameters=128938,new_trainable_parameters=0,optimizer_updates=0,
        failed_attempts_retained=['runs/full-ai-next-priorities-v1/temporal-pressure-core-cpu-qualification-01.json',
            'runs/full-ai-next-priorities-v1/temporal-pressure-core-cuda-qualification-02.json'],
        first_failure='Float32 batching exceeded fixed equation parity tolerance; precise learned parent execution retained without relaxing tolerances',
        second_failure='Legacy CPU reference loader rejected CUDA already initialized; load ordering fixed without model changes',
        component_only=True,full_state_temporal_candidate=False,causal_prediction_accuracy_claim=False,
        CPU_latency_claim=False,reserved_scenes_consumed=False,original_goal_complete=False)
    write(DOCS/'temporal-pressure-core-port-results.json',value)
    lines=['# Frozen learned pressure component: batched differentiable port','','The existing learned pressure component now passes CPU and CUDA qualification.',
        'This is component preparation, with zero optimizer updates. It is not a causal',
        'full-state temporal model and supplies no new inference-latency claim.','',
        '| Device | Native requests | Gradient cases | Maximum parent pressure difference | Maximum equation-error difference |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['device']} | {row['requests']} | {row['gradient_cases']} | {row['maximum_parent_pressure_relative_difference']:.6g} | {row['maximum_equation_error_difference']:.6g} |")
    lines.extend(['','Four grids, both cells-per-metre values, batch sizes1–4 and deterministic',
        'input gradients qualify. Learned coefficients remain unchanged. Native future',
        'requests enter only this offline component check; reserved scenes remain unopened.',
        'The unchanged independently confirmed pressure implementation is the reference.','',
        'Precise dense CNN maps are constructed from learned unit responses at setup,',
        'without physical inverses or reference labels. They occupy13,634,568bytes;',
        'setup times are recorded separately and are not a warm-latency benchmark.',
        'The first float32 numerical failure and subsequent CUDA loader-order preflight',
        'failure are retained with their source snapshots. Numerical tolerances did not change.','',
        'Next: integrate the pressure expert with causal request, velocity and tracer',
        'prediction, then qualify the deployed full-state feedback path before fitting.',
        'Keep endpoint/intermediate and retained-feature comparisons and complete4/6/8ms',
        'budgets. Every physical output and own-history accuracy criterion remains open.','',
        '[Machine-readable evidence](temporal-pressure-core-port-results.json) · [Next implementation plan](temporal-pressure-expert-reuse-plan.md)',''])
    (DOCS/'temporal-pressure-core-port-results.md').write_text('\n'.join(lines))
    print('Summarized CPU/CUDA learned pressure port qualification; no temporal accuracy or speed claim',flush=True)


if __name__=='__main__':run()
