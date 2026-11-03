"""Prepare an authored guide bound to all205current native fit cards.

No project edit, model execution, weight copy or lineage mutation. Summary text
comes from the actual captured reviewed cards, not a second interpretation.
"""
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT = RUN/'authored-model-portfolio-note-01'
REPORT = ROOT/'docs/plans/model-portfolio-migration-v1/reports/authored-model-portfolio-preparation-01.json'
GUIDE = 'notes/00-models.portfolio.json'
LABELS = {'Method':'method','Purpose':'purpose','Experiment':'experiment',
          'Observed outcome':'outcome','Lesson':'lesson','Reproduction':'reproduction'}
TASKS = {
    'pressure_proposal':'Pressure proposal fits',
    'full_state_generator':'Complete simulation generators',
    'full_state_correction_component':'Full-state correction components',
    'pressure_component':'Pressure components',
    'guided_shared_component':'Shared guided components',
    'recurrent_transport_component':'Recurrent transport components',
    'pressure_head_component':'Pressure heads',
    'full_state_readout_component':'Full-state readouts',
    'pressure_request_component':'Pressure request components',
    'transport_component':'Transport components',
    'guided_pressure_component':'Guided pressure components',
    'viscosity_component':'Viscosity components'}
RANGES = [(3,15,'pressure-inputs','C03–C15 · inputs and training'),
          (16,25,'pressure-stability','C16–C25 · residual and stability'),
          (26,37,'pressure-multigrid','C26–C37 · grids and recurrence'),
          (38,53,'pressure-direction','C38–C53 · directional feedback'),
          (54,105,'pressure-variants','C54–C105 · pressure variants'),
          (106,130,'pressure-successors','C106–C130 · successors and repeats')]
COMPARISONS = [('C17','C16'),('C18','C20'),('C19','C17'),('C20','C18'),('C24','C21'),
               ('C25','C23'),('C27','C26'),('C28','C27'),('C29','C28'),('C30','C28'),
               ('C31','C28'),('C32','C28'),('C33','C28'),('C34','C33'),('C36','C33'),('C107','C106')]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def binding(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path.read_bytes()))


def require(value,message):
    if not value:
        raise ValueError(message)


def summary(text,fit_id):
    require(text.startswith('# ') and f'Retrospective source review · {fit_id}\n' in text,
            'Reviewed index entry has no original authored review prefix')
    require(text.count('## Exact source bindings for this review\n') == 1,
            'Expected one original review evidence heading')
    prefix = text.split('## Exact source bindings for this review\n',1)[0]
    matches = list(re.finditer(r'^\*\*(Method|Purpose|Experiment|Observed outcome|Lesson|Reproduction)\.\*\* ',prefix,re.M))
    require([m.group(1) for m in matches] == list(LABELS),'Expected six ordered original review paragraphs')
    result = {}
    for i,match in enumerate(matches):
        end = matches[i+1].start() if i+1 < len(matches) else len(prefix)
        paragraph = prefix[match.end():end].strip()
        require(paragraph and paragraph in text,'Authored summary must be literal captured card text')
        result[LABELS[match.group(1)]] = paragraph
    return result


def prepare():
    require(not OUTPUT.exists() and not REPORT.exists(),'Fresh task-owned guide outputs required')
    header_path = PROJECT/'.tuldok/project.json'
    header_raw = header_path.read_bytes();header = json.loads(header_raw)
    current = {}
    for entry in header['inventory']:
        if entry['family'] != 'project_file':
            continue
        raw = (PROJECT/'.tuldok'/entry['path']).read_bytes()
        require(len(raw) == entry['bytes'] and sha(raw) == entry['sha256'],'Canonical file envelope differs')
        document = json.loads(raw)
        if document['project_id'] == header['project_id']:
            require(document['path'] not in current,'Current authored note path repeats')
            current[document['path']] = (entry,document)
    require(GUIDE not in current and not (PROJECT/GUIDE).exists(),'Do not replace or replay a previously adopted guide')
    _,index_record = current['notes/portfolio-index.json']
    index = json.loads(index_record['current']['text'])
    require(len(index['fits']) == 205 and len(index['aliases']) == 379,'Complete native fit/alias index required')
    aliases = {}
    named = {}
    for alias in index['aliases']:
        fit_id = alias['fit_item_id']
        if fit_id is None:
            continue
        aliases.setdefault(fit_id,set()).add(str(alias['label']))
        if alias['item_id'].startswith('cohorts/') and re.fullmatch(r'C\d{2,3}',str(alias['label'])):
            require(alias['label'] not in named or named[alias['label']] == fit_id,'Named checkpoint fit identity is ambiguous')
            named[alias['label']] = fit_id
    groups = [dict(id=task,label=label,parent=None,
        description='Authored task grouping from the retained fit index; not training ancestry.') for task,label in TASKS.items()]
    groups += [dict(id=key,label=label,parent='pressure_proposal',
        description='Authored checkpoint-range grouping. Actual reviewed purpose, training and lessons are retained on each card.')
        for _,_,key,label in RANGES]
    groups.append(dict(id='pressure-other',label='Pressure continuations and other fits',parent='pressure_proposal',
        description='Other recorded pressure fits; no parentage is inferred from membership.'))
    models,refs,texts = [],[],{}
    for fit in index['fits']:
        require(fit['task'] in TASKS,'Unclassified original task requires an authored grouping decision')
        entry,card = current[fit['card_path']];version = card['current'];text = version['text']
        require(card['kind'] == 'note' and not version.get('deleted') and sha(text.encode()) == version['sha256'] == fit['card_sha256'],
                'Original fit card differs from the exact native index binding')
        names = sorted(aliases.get(fit['item_id'],[]))
        checkpoints = sorted([label for label,identity in named.items() if identity == fit['item_id']],key=lambda label:int(label[1:]))
        group = fit['task']
        if group == 'pressure_proposal':
            group = 'pressure-other'
            for low,high,key,_ in RANGES:
                if checkpoints and low <= int(checkpoints[0][1:]) <= high:
                    group = key;break
        reviewed = fit.get('method_review_status') == 'captured_source_and_receipt_reviewed'
        model = dict(id=fit['item_id'],label=(' / '.join(checkpoints)+' · ' if checkpoints else '')+fit['name'],
            group_id=group,aliases=names,review='source_reviewed' if reviewed else 'review_pending',
            card=dict(file_id=entry['id'],path=card['path'],revision=version['revision'],sha256=version['sha256']),
            summary=summary(text,fit['item_id']) if reviewed else None)
        models.append(model);refs.append(dict(fit_item_id=fit['item_id'],record_path=entry['path'],record_sha256=entry['sha256'],**model['card']))
        texts[fit['item_id']] = text
    by_id = {m['id']:m for m in models}
    relationships = []
    for label,baseline in COMPARISONS:
        require(label in named and baseline in named,'Explicit authored comparison requires both actual named fits')
        model = by_id[named[label]]
        require(model['summary'] is not None,'Authored comparison requires a reviewed source card')
        excerpt = next((model['summary'][field] for field in ('lesson','purpose','experiment','outcome','method')
                        if baseline in model['summary'][field]),None)
        require(excerpt is not None and excerpt in texts[model['id']],f'No literal reviewed comparison context for {label}/{baseline}')
        relationships.append(dict(from_=model['id']))
        relationships[-1] = {'from':model['id'],'to':named[baseline],'kind':'comparison',
            'explanation':f'{label} compared with {baseline}; retrospective source-review context, not a Trial ancestry assertion or new metric qualification.',
            'evidence_model':model['id'],'excerpt':excerpt}
    require(len(models) == len(by_id) == 205 and sum(m['review'] == 'source_reviewed' for m in models) == 69,
            'Complete205-fit /69-reviewed population required')
    guide = dict(format='project_model_portfolio_v1',project_id=header['project_id'],title='Rheon model fit portfolio',
        description='205 recorded fits/components, grouped by task and authored method ranges. 69 have retrospective source-reviewed summaries; the remaining reviews are explicit. '
            'The 524 parameter byte states and 379 dashboard aliases are separately indexed, not 524 independent models. '
            'Read exact model cards for pressure-request versus physical-step scope, training populations, outcomes and reproduction limits. '
            'Original Trial ancestry remains unchanged and unknown where not recorded. Full benchmarking, drift, clean reproduction and portability remain incomplete.',
        authorship='retrospective',groups=groups,models=models,relationships=relationships)
    raw = (json.dumps(guide,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    require(len(raw) <= 1048576 and header_path.read_bytes() == header_raw,'Bounded note and unchanged canonical source required')
    OUTPUT.mkdir();target = OUTPUT/GUIDE;target.parent.mkdir();target.write_bytes(raw)
    preparation = dict(status='prepared_not_published',created_unix=time.time(),source_checkpoint=header['checkpoint'],
        source_project_id=header['project_id'],source_manifest_sha256=sha(header_raw),guide_path=GUIDE,
        guide_sha256=sha(raw),guide_bytes=len(raw),notes=[dict(kind='note',path=GUIDE,expected_revision=0,sha256=sha(raw),bytes=len(raw))],
        model_count=205,source_reviewed=69,review_pending=136,groups=len(groups),authored_comparisons=len(relationships),
        exact_card_bindings=refs,new_model_forwards=0,native_mutations=0,payload_copies=0,full_portfolio_complete=False,
        producer=binding(Path(__file__)))
    (OUTPUT/'preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
    REPORT.write_text(json.dumps(dict(preparation,preparation_path=str((OUTPUT/'preparation.json').relative_to(ROOT))),indent=2)+'\n')
    print('205-fit guide prepared:',len(raw),'bytes,69reviewed summaries,',len(relationships),'source-excerpt comparisons. No native mutation.')


if __name__ == '__main__':
    prepare()
