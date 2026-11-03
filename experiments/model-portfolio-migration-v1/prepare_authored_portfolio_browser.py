"""Prepare actual native organization/filter/card/evidence UI checks after ACK."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/model-portfolio-migration-v1'
JOB=RUN/'native-authored-portfolio-adoption-01'
OUTPUT=JOB/'browser-actions.json'


def prepare():
    if OUTPUT.exists():raise ValueError('Preserve original native browser actions')
    report=json.loads((ROOT/'docs/plans/model-portfolio-migration-v1/reports/native-authored-model-portfolio-adoption-01.json').read_bytes())
    if report['status']!='native_authored_205fit_guide_exact_and_original_scientific_records_unchanged':
        raise ValueError('Native guide publication/custody required')
    prep=RUN/'authored-model-portfolio-note-01'
    guide=json.loads((prep/report['guide']['path']).read_bytes())
    models=guide['models'];named={label:model for model in models for label in model['aliases'] if label in ('C107','C106','C23')}
    if len(named)!=3:raise ValueError('Exact native checkpoint aliases required')
    actions=[]
    def add(kind,**values):actions.append(dict(kind=kind,**values))
    def apply(group='',search='',review=''):
        for name,value in [('group',group),('search',search),('review',review)]:
            add('fill',selector='#project-portfolio-'+name,value=value)
        add('click',selector='#project-portfolio-filter-form button')
    def select(model,screenshot):
        add('click',selector='[data-lineage-node="model/'+model['id']+'"]')
        add('wait_text',selector='#project-portfolio-status',text='Exact retained model summary and captured evidence.')
        add('wait_text',selector='#project-portfolio-model-heading',text=model['label'])
        if model['summary']:
            for field in ('method','purpose','experiment','outcome','lesson','reproduction'):
                add('wait_text',selector='#project-portfolio-detail',text=model['summary'][field])
        else:add('wait_text',selector='#project-portfolio-detail',text='A source-reviewed interpretation has not been recorded for this fit.')
        add('scroll',selector='#project-portfolio-model-heading');add('screenshot',name=screenshot)
    add('assert_text',text=report['checkpoint'])
    add('click',selector='#project-portfolio-notes')
    add('wait_text',selector='#project-portfolio-note-count',text='1 organization notes found')
    add('wait_enabled',selector='#project-portfolio-note')
    add('fill',selector='#project-portfolio-note',value=report['guide']['id'])
    add('click',selector='#project-portfolio-load')
    add('wait_text',selector='#project-portfolio-status',text='Loaded retained authored model organization.')
    add('wait_text',selector='#project-portfolio-count',text='205 models in this guide; 69 source reviewed. 205 match the applied filters. Showing 1–20.')
    add('scroll',selector='#project-portfolio-title');add('screenshot',name='all205-authored-organization.png')
    apply(search='C107',review='source_reviewed')
    matching=[model for model in models if model['review']=='source_reviewed' and any('c107' in value.lower() for value in [model['label'],*model['aliases']])]
    add('wait_text',selector='#project-portfolio-count',text=f'{len(matching)} match the applied filters.')
    select(named['C107'],'C107-readable-summary.png')
    add('click',selector='#project-portfolio-detail button',text='Comparison: '+named['C106']['label'])
    add('wait_text',selector='#project-portfolio-model-heading',text=named['C106']['label'])
    add('wait_text',selector='#project-portfolio-detail',text=named['C106']['summary']['method'])
    add('wait_text',selector='#project-portfolio-detail',text=named['C106']['summary']['outcome'])
    add('scroll',selector='#project-portfolio-model-heading');add('screenshot',name='C106-retained-comparison.png')
    add('click',selector='#project-portfolio-detail details > summary',text='Exact captured card and original receipts')
    add('wait_text',selector='#project-portfolio-detail',text='Original retained fit facts before this review')
    add('screenshot',name='C106-captured-card-disclosure.png')
    apply(group='pressure-stability',review='source_reviewed')
    add('wait_text',selector='#project-portfolio-count',text='10 match the applied filters. Showing 1–10.')
    add('scroll',selector='#project-portfolio-count');add('screenshot',name='residual-stability-group.png')
    select(named['C23'],'C23-manufactured-physical-scope.png')
    apply(group='full_state_generator',review='review_pending')
    pending=[m for m in models if m['group_id']=='full_state_generator' and m['review']=='review_pending']
    if not pending:raise ValueError('An actual unreviewed generator is required for pending-state coverage')
    add('wait_text',selector='#project-portfolio-count',text=f'{len(pending)} match the applied filters.')
    select(pending[0],'pending-generator-evidence.png')
    apply()
    add('wait_text',selector='#project-portfolio-count',text='205 match the applied filters. Showing 1–20.')
    add('click',selector='#project-portfolio-next')
    add('wait_text',selector='#project-portfolio-count',text='Showing 21–40.')
    add('scroll',selector='#project-portfolio-count');add('screenshot',name='model-window-second-page.png')
    add('assert_text',text='Experiment progression')
    OUTPUT.write_text(json.dumps(actions,indent=2,ensure_ascii=False)+'\n')
    print(len(actions),'native organization actions prepared; exact C107/C106/C23 and pending-generator selection.')


if __name__=='__main__':prepare()
