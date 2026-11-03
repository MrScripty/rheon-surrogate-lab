"""Admit recorded full-state compositions and execution policies separately.

Execution policies reuse trained weights. Float32 policies require their own
accuracy captures; only byte-qualified policies may inherit another capture.
"""
import hashlib
import os
from pathlib import Path

STUDY='full-ai-next-priorities-v1'


def temporal_accuracy_rows(label,horizons,value,source):
    rows=[]
    for regime in ('small','32cubed'):
        scenes=[s for s in value['scenes'] if ('small' if max(s['counts_xyz'])<=16 else '32cubed')==regime]
        for i,interval in enumerate(['single_block','rollout_12','rollout_48','rollout_240']):
            rows.append(dict(label=label,horizons=horizons,regime=regime,interval=interval,
                field_worst={f:max(s['summaries'][i]['worst'][f] for s in scenes) for f in ['equation','pressure','velocity','tracer']},
                scenes=len(scenes),complete=all(s['summaries'][i]['complete'] for s in scenes),
                accuracy_pass=all(s['summaries'][i]['all_accuracy_criteria_pass'] for s in scenes),source=source))
    return rows


def append_viscosity_executions(models,rows,base,lab,load):
    base,lab=Path(base),Path(lab)
    root=lab/'runs'/STUDY
    executions=[]
    reports={}

    def witnessed(path,status='completed'):
        path=Path(path)
        if path in reports:
            return reports[path]
        value=load(path)
        if (value['status']!=status or value.get('model_promotion') or value.get('reserved_scenes_consumed')
                or value.get('optimizer_updates',0)):
            raise ValueError('Unqualified full-state execution record: '+str(path))
        for source,expected in value.get('source_evidence',{}).items():
            target=lab/source
            if target.suffix=='.json':
                load(target,expected)
            elif hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
                raise ValueError('Execution evidence changed: '+source)
        reports[path]=value
        return value

    def parent(identifier):
        return next(m for m in models+executions if m['identifier']==identifier)

    def add(identifier,label,parent_id,kind,evaluation,qualification=None):
        previous=parent(parent_id)
        value=load(root/evaluation)
        model=value['model']
        if (value['status']!='completed' or value['optimizer_updates'] or value['cuda_initialized']
                or value['candidate_numerical_pressure_solve'] or value['bank_sha256']!=previous['bank_sha256']):
            raise ValueError('Execution accuracy scope differs: '+identifier)
        if kind=='execution' and model['weights_sha256']!=previous['weights_sha256']:
            raise ValueError('Execution changed trained weights: '+identifier)
        if qualification:
            proof=witnessed(root/qualification,status='passed')
            if not proof['all_physical_bytes_exact']:
                raise ValueError('Accuracy inheritance needs byte-exact qualification')
            if proof['model']['weights_sha256']!=model['weights_sha256']:
                raise ValueError('Execution qualification checkpoint differs')
        item=dict(previous,label=label,identifier=identifier,record_kind=kind,
            parent_label=previous['label'],parent_weights_sha256=previous['weights_sha256'],
            weights_sha256=model['weights_sha256'],optimizer_updates=0,
            evaluation_source=os.path.relpath(root/evaluation,base),
            accuracy_inheritance='byte-qualified' if qualification else 'actual execution evaluation',
            qualification_source=os.path.relpath(root/qualification,base) if qualification else None,
            training_scope='Execution/composition with zero new full-state optimizer updates; inherited training and calibrated components remain additional.')
        if kind=='composition':
            models.append(item)
        else:
            executions.append(item)
        rows.extend(temporal_accuracy_rows(label,item['horizons'],value,item['evaluation_source']))
        return item

    if not (base/STUDY/'neural-viscosity-results.json').exists():
        return executions
    report=witnessed(base/STUDY/'neural-viscosity-results.json')
    if not report['all_other_neural_tensors_exact'] or report['assembly_optimizer_updates']:
        raise ValueError('Viscosity composition changes its declared parent')
    evaluation='neural-viscosity-evaluation-01/b2-displacement-viscosity/summary.json'
    value=load(root/evaluation)
    metadata=value['model']
    assembly=load(lab/metadata['assembly']/'receipt.json',metadata['assembly_receipt_sha256'])
    if (assembly['optimizer_updates'] or not assembly['all_other_tensors_bitwise_unchanged']
            or assembly['parents']['parent']['weights_sha256']!=parent('b2-displacement')['weights_sha256']
            or hashlib.sha256((lab/metadata['assembly']/'weights.npz').read_bytes()).hexdigest()!=metadata['weights_sha256']):
        raise ValueError('Unbound calibrated viscosity composition')
    composed=add('b2-displacement-viscosity','K=2 calibrated viscosity','b2-displacement','composition',evaluation)
    composed.update(source=os.path.relpath(lab/metadata['assembly']/'receipt.json',base),
        component_optimizer_updates=report['managed_component_updates'],
        training_scope='Zero full-state assembly updates. Inherits the 10,000-update displacement fit and a separately trained 243-parameter viscosity component (1,000 updates); this is a composition, not an independent fit.')

    specs=[
        ('neural-viscosity-cost-results.json','b2-displacement-viscosity-fused','K=2 viscosity · fused','b2-displacement-viscosity',evaluation,'fused-neural-viscosity-model-qualification-01.json'),
        ('fast-preparation-cost-results.json','b2-displacement-viscosity-fast','K=2 viscosity · fast','b2-displacement-viscosity',evaluation,'fast-preparation-model-qualification-01.json'),
        ('float32-learned-map-results.json','b2-displacement-viscosity-fast-f32map','K=2 viscosity · float32 map','b2-displacement-viscosity','float32-map-evaluation-01/summary.json',None),
    ]
    specs.extend(('fixed-viscosity-cost-results.json',f'b{k}-fixed-viscosity-fast',f'K={k} fixed viscosity · fast',f'b{k}-fixed-viscosity',f'fixed-viscosity-evaluation-01/b{k}-fixed-viscosity/summary.json',f'fixed-viscosity-execution-qualification-k{k}-01.json') for k in (1,2,3,4))
    specs.extend([
        ('fixed-viscosity-k2-precision-results.json','b2-fixed-viscosity-fast-float64','K=2 fixed viscosity · float64 policy','b2-fixed-viscosity','fixed-viscosity-evaluation-01/b2-fixed-viscosity/summary.json','fixed-viscosity-execution-qualification-k2-01.json'),
        ('fixed-viscosity-k2-precision-results.json','b2-fixed-viscosity-fast-float32-map','K=2 fixed viscosity · float32 map','b2-fixed-viscosity','fixed-viscosity-k2-float32-evaluation-01/summary.json',None),
    ])
    for report_name,identifier,label,parent_id,evaluation,qualification in specs:
        path=base/STUDY/report_name
        if not path.exists():
            continue
        proof=witnessed(path)
        source=f'runs/{STUDY}/{evaluation}'
        if source not in proof['source_evidence']:
            raise ValueError('Execution accuracy is not witnessed: '+identifier)
        item=add(identifier,label,parent_id,'execution',evaluation,qualification)
        item['execution_report_source']=STUDY+'/'+report_name
    return executions
