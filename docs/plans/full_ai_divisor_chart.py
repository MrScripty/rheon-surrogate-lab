"""Admit the frozen learned-divisor accuracy result and untimed audit data."""
from pathlib import Path
import os

from full_ai_history_chart import DIVERSITY


def enrich_divisor(study,base,lab,*,read,qualified,sha,accuracy,execution):
    base,lab=Path(base),Path(lab)
    docs,root=base/DIVERSITY,lab/'runs'/DIVERSITY
    path=docs/'learned-divisor-model-results-05.json'
    if not path.exists():return
    report=read(path)
    if (report['status']!='independent_current_regime_callable_accuracy_pass_latency_pending'
            or report['new_parameter_fits'] or report['numerical_solver_in_candidate']
            or report['analytical_physical_eigenvalues_in_candidate'] or report['latency_qualified']):
        raise ValueError('Learned-divisor result scope changed')
    candidate=read(lab/report['candidate'],report['candidate_sha256'])
    read(lab/report['evaluator_protocol'],report['evaluator_protocol_sha256'])
    fresh=read(lab/report['result'],report['result_sha256'])
    binding=report['consumed_regression']
    consumed=read(lab/binding['result'],binding['sha256'])
    if (not fresh['independent_confirmation'] or not fresh['all_accuracy_pass']
            or len(fresh['rows'])!=report['complete_states_scored'] or len(consumed['rows'])!=binding['physical_outputs']
            or report['independent_scenes']!=16 or binding['scenes']!=72):
        raise ValueError('Learned-divisor evaluation population differs')
    for source,expected in candidate['source_evidence'].items():
        path=lab/source
        if path.suffix=='.json':read(path,expected)
        elif sha(path)!=expected:raise ValueError('Frozen divisor source changed: '+source)
    records={m['label']:m for m in study['temporal_models']+study['temporal_executions']}
    for k in (1,2,3,4):
        label=f'K={k} recurrent sampler · learned divisor'
        rows=[]
        value=None
        for folder,protocol,population,status in [
            ('learned-divisor-consumed-regression-02','learned-divisor-consumed-regression-protocol-02.json','divisor_consumed','completed_consumed_pressure_rounding_control'),
            ('learned-divisor-independent-evaluation-02','learned-divisor-independent-evaluation-protocol-02.json','divisor_fresh','completed_callable_learned_divisor_evaluation'),
        ]:
            captured,added=accuracy(root/folder,docs/protocol,label,k,population,status=status)
            if (captured.get('numerical_oracle_used') or captured.get('numerical_solver_in_candidate')
                    or captured.get('analytical_physical_eigenvalues_in_candidate') or captured.get('new_parameter_fitting_executed')):
                raise ValueError('Divisor execution substitutes numerical inference or training')
            for row in added:row['study_id']=DIVERSITY
            if population=='divisor_consumed':value,rows=captured,added
        parent=records[f'K={k} recurrent sampler · pressure history · tabulated forcing']
        item=execution(label,parent,value,rows)
        item['study_id']=DIVERSITY
        item['training_scope']='Execution-only pressure representation: tensor basis with division by reciprocals of unchanged managed learned gains. Reuses the mixed-query pressure checkpoint and sampler fit; no new optimizer calls or numerical pressure solver. K1–4 package recurrent outputs. Accuracy is measured over 240 steps on 72 consumed and 16 independent scenes at densities 8/16 in four supported geometries. Latency pending; no inherited timing or higher-density accuracy claim.'
        item['independent_evaluation_source']=os.path.relpath(lab/report['result'],base)
    study['divisor_result']=dict(source=os.path.relpath(docs/'learned-divisor-model-results-05.json',base),
        independent_scenes=16,scored_states=report['complete_states_scored'],new_parameter_fits=0,
        accuracy_pass=report['all_accuracy_pass'],latency_qualified=False,groups=report['grouped_summaries'])

    audit_path=lab/report['untimed_allocation_evidence']
    audit=read(audit_path)
    if (audit['status']!='completed_untimed_tensor_work_audit' or audit['latency_claim']
            or audit['clocks_read'] or audit['optimizer_updates'] or len(audit['rows'])!=8):
        raise ValueError('Tensor-work audit is not an untimed complete result')
    read(docs/'divisor-tensor-work-audit-protocol-01.json',audit['protocol_sha256'])
    for source,expected in audit['source_sha256'].items():
        path=lab/source
        if path.suffix=='.json':read(path,expected)
        elif sha(path)!=expected:raise ValueError('Tensor-work audit source changed')
    rows=[]
    for row in audit['rows']:
        if not row['complete_output_bitwise_parity']:raise ValueError('Instrumented divisor output differs')
        stages={}
        for operation in row['operations']:
            stage=stages.setdefault(operation['stage'],dict(calls=0,returned_new_storage_bytes=0))
            for field in stage:stage[field]+=operation[field]
        rows.append(dict(identifier=row['identifier'],counts_xyz=row['counts_xyz'],horizons=row['horizons'],stages=stages))
    study['divisor_work']=dict(source=os.path.relpath(audit_path,base),rows=rows,
        measurement=audit['measurement'],limitations=audit['limitations'],latency_claim=False,clocks_read=False)

    # This is a fixed receipt-only snapshot, never an invitation to consume
    # untouched references or to claim that successor training is ready.
    path=root/'expanded-readiness-audit-01.json'
    if path.exists():
        audit=read(path)
        if (audit['status']!='completed_receipt_only_expanded_readiness_snapshot'
                or audit['reserved_references_opened'] or audit['field_archives_opened'] or audit['model_fitting_executed']):
            raise ValueError('Expanded reference audit scope differs')
        read(docs/'expanded-readiness-audit-protocol-01.json',audit['protocol_sha256'])
        for source,expected in audit['source_sha256'].items():
            target=lab/source
            if target.suffix=='.json':read(target,expected)
            elif sha(target)!=expected:raise ValueError('Readiness audit source changed')
        for row in audit['rows']:read(lab/row['receipt'],row['receipt_sha256'])
        completed={split:len({r['identifier'] for r in audit['rows'] if r['split']==split}) for split in audit['registered_scene_counts']}
        if any(completed[split]!=audit['unique_completed_scenes_by_split'].get(split,0) for split in completed):
            raise ValueError('Reference receipt counts differ')
        study['expanded_readiness']=dict(source=os.path.relpath(path,base),completed=completed,
            registered=audit['registered_scene_counts'],full_data_training_ready=audit['full_data_training_ready'],
            receipt_rows=len(audit['rows']),field_archives_reverified=False,
            mature_four_step_windows=sum(len(r['mature_K4_windows_at_requested_starts']) for r in audit['rows']))
