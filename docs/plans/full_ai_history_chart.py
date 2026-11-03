"""Read mixed-query pressure/history and data-diversity evaluation snapshots."""
import os
from pathlib import Path

from full_ai_tight_chart import STUDY

DIVERSITY='full-ai-data-diversity-v1'


def enrich_pressure_history(study,base,lab,*,read,qualified,pair,sha,publication,accuracy,execution,paired_timing):
    base,lab=Path(base),Path(lab)
    root=lab/'runs'/STUDY
    report=publication('pressure-history-and-mixed-query-summary-12.json','completed_pressure_history_independent_check_latency_targets_unmet')
    if not report:
        return
    records={m['label']:m for m in study['temporal_models']+study['temporal_executions']}
    fitted=pair('mixed-modal-pair-11',80)
    study['tight_components'].append(dict(fitted,name='Mixed-query modal pressure',component_only=True,
        inherited_optimizer_updates=960,groups=[],source_evaluation=fitted['source']))
    label='K=1 recurrent sampler · mixed pressure direct'
    value,rows=accuracy('mixed-modal-consumed-screen-11','mixed-modal-screen-protocol-11.json',label,1,'history_consumed')
    parent=records['K=1 recurrent sampler']
    direct=dict(parent,label=label,identifier='tight-mixed-direct',record_kind='composition',optimizer_updates=0,
        parent_label=parent['label'],parent_weights_sha256=parent['weights_sha256'],
        bank_sha256=rows[0]['bank_sha256'],pressure_component_weights_sha256=fitted['weights_sha256'],component_optimizer_updates=80,
        evaluation_source=rows[0]['source'],learned_inventory=value['learned_inventory'],
        training_scope='An 80-call managed mixed-query pressure refinement adds 128 training-only queries to 144 original pairs. Inherits the earlier 800+80+80 pressure calls and 440-call sampler fit. This is an assembly; full-state accuracy uses 40 consumed scenes, not a new transport fit.')
    study['temporal_models'].append(direct)
    value,rows=accuracy('mixed-modal-m30-learned-control-11','mixed-modal-m30-learned-control-protocol-11.json',label+' · head m30',1,'history_consumed',status='completed_consumed_mixed_model_precision_or_floor_diagnostic')
    execution(label+' · head m30',direct,value,rows)
    history={}
    original_bank='runs/full-ai-next-priorities-v1/temporal-development-240-01/bank.json'
    for k in (1,2,3,4):
        label=f'K={k} recurrent sampler · pressure history'
        value,rows=accuracy('pressure-history-all-horizon-consumed-12','pressure-history-all-horizon-consumed-protocol-12.json',label,k,'history_consumed',status='completed_consumed_scene_pressure_history_prototype')
        item=dict(direct,label=label,identifier=f'tight-history-k{k}',horizons=k,record_kind='composition' if k==1 else 'execution',
            parent_label=direct['label'] if k==1 else 'K=1 recurrent sampler · pressure history',parent_weights_sha256=parent['weights_sha256'],
            evaluation_source=rows[0]['source'],learned_inventory=value['learned_inventory'],
            training_scope=direct['training_scope']+' Causal current-pressure reuse uses learned flux, tied momentum and learned pressure only; no new optimizer calls. K1–4 package recurrent computation. Independent successor scenes remain a separate population.')
        (study['temporal_models'] if k==1 else study['temporal_executions']).append(item)
        history[k]=item
        # Timing is on the original eight scenes. Derive their accuracy from
        # the complete 40-scene record rather than attach a different population.
        accuracy('pressure-history-all-horizon-consumed-12','pressure-history-all-horizon-consumed-protocol-12.json',label,k,
            status='completed_consumed_scene_pressure_history_prototype',subset_bank=original_bank)
        accuracy('pressure-history-fresh-blocks-12','pressure-history-fresh-evaluation-protocol-12.json',label,k,'history_fresh','blocks')
        accuracy('pressure-history-fresh-rollout-12','pressure-history-fresh-evaluation-protocol-12.json',label,k,'history_fresh')
    for tag,mode in [('direct',direct),('reuse',history[1])]:
        label=mode['label']+' · three-scene control'
        value,rows=accuracy(f'pressure-history-{tag}-learned-12',f'pressure-history-{tag}-learned-protocol-12.json',label,1,'history_diagnostic',status='completed_consumed_scene_pressure_history_prototype')
        execution(label,mode,value,rows)
    for tag in ('p1','p2','p4'):
        proof,_=qualified(root/f'sampler-minmax-{tag}-graph-12.json','passed_minmax_complete_graph_byte_execution',f'sampler-minmax-{tag}-graph-protocol-12.json')
        paired_timing(f'sampler-minmax-{tag}-cost-12.json',f'sampler-minmax-{tag}-cost-protocol-12.json',history,
            'development','completed_paired_minmax_sampler_comparison','frozen_v7',proof)

    # Known prescribed inputs are execution changes. They supply no new fitting
    # or inference timings, and no evidence for the broader high-density phase.
    docs=base/DIVERSITY
    summary_path=docs/'strict-force-model-results-04.json'
    if not summary_path.exists():
        return
    summary=read(summary_path)
    if summary['optimizer_updates'] or not summary['no_new_parameter_fits'] or summary['candidate_numerical_pressure_solver']:
        raise ValueError('Diversity execution report changes training/solver scope')
    for source,expected in summary['source_sha256'].items():
        path=lab/source
        if path.suffix=='.json':read(path,expected)
        elif sha(path)!=expected:raise ValueError('Diversity evidence changed: '+source)
    diversity_root=lab/'runs'/DIVERSITY
    def diversity_accuracy(folder,protocol,label,k,population,status):
        value,rows=accuracy(diversity_root/folder,docs/protocol,label,k,population,status=status)
        if value.get('numerical_solver_in_candidate') or value.get('new_parameter_fitting_executed'):
            raise ValueError('Strict prescribed-input execution is not a candidate evaluation')
        for row in rows:row['study_id']=DIVERSITY
        return value,rows
    policies=[
        ('strict-legacy-force-regression-01','strict-legacy-force-regression-protocol-01.json','strict scalar forcing','history_regression','completed_consumed_scene_strict_force_learned_generator_regression'),
        ('table-force-regression-01','table-force-regression-protocol-01.json','tabulated forcing','history_regression','completed_table_force_learned_generator_evaluation'),
    ]
    for folder,protocol,suffix,population,status in policies:
        for k in (1,2,3,4):
            label=f'K={k} recurrent sampler · pressure history · '+suffix
            value,rows=diversity_accuracy(folder,protocol,label,k,population,status)
            item=execution(label,history[k],value,rows)
            item['study_id']=DIVERSITY
            item['training_scope']+=' Prescribed external input arithmetic only; supported grids at densities 8/16. No higher-density model result.'
            if suffix=='tabulated forcing':
                diversity_accuracy('table-force-independent-evaluation-01','table-force-independent-evaluation-protocol-01.json',label,k,'force_fresh',status)
    label='K=1 recurrent sampler · tabulated forcing · tensor pressure diagnostic'
    value,rows=diversity_accuracy('tensor-table-full240-diagnostic-01','tensor-table-full240-diagnostic-protocol-01.json',label,1,'force_diagnostic','completed_table_force_learned_generator_evaluation')
    item=execution(label,history[1],value,rows)
    item['study_id']=DIVERSITY
    item['training_scope']+=' Two consumed failure-diagnostic scenes only; not new independent confirmation.'

    frequency_path=docs/'reference-frequency-results-02.json'
    frequency=read(frequency_path)
    if (frequency['status']!='completed_reference_input_and_discretization_pilot' or frequency['optimizer_updates']
            or frequency['model_targets_verified'] or frequency['reserved_references_consumed']
            or len(frequency['rows'])!=64 or not all(r['complete'] and r['accepted_output_steps']==60 for r in frequency['rows'])):
        raise ValueError('Incomplete reference-frequency pilot')
    for source,expected in frequency['source_sha256'].items():read(lab/source,expected)
    source=lab/'runs'/DIVERSITY/'continuous-reference-convergence-02/results.json'
    raw=read(source)
    if not raw['reference_only'] or raw['optimizer_updates'] or raw['temporal_convergence_qualified']:
        raise ValueError('Reference sensitivity scope differs')
    study['reference_frequency']=dict(rows=frequency['rows'],source=os.path.relpath(frequency_path,base),
        raw_source=os.path.relpath(source,base),metric=frequency['metric'],limitations=frequency['limitations'],
        reference_only=True,temporal_convergence_qualified=False)

    from full_ai_divisor_chart import enrich_divisor
    enrich_divisor(study,base,lab,read=read,qualified=qualified,sha=sha,accuracy=accuracy,execution=execution)
