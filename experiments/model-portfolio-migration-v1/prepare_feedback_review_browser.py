"""Prepare actual native UI reads of all ten newly reviewed guide entries."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
JOB = RUN/'method-review-native-adoption-12'
OUTPUT = JOB/'browser-actions.json'


def prepare():
    if OUTPUT.exists():
        raise ValueError('Preserve original browser actions')
    report = json.loads((ROOT/'docs/plans/model-portfolio-migration-v1/reports/native-seventynine-method-review-adoption-01.json').read_bytes())
    if report['status'] != 'native_thirteen_note_revisions_exact_and_scientific_metadata_unchanged':
        raise ValueError('Native acknowledgement and exact custody required')
    guide = json.loads((RUN/'native-feedback-review-publication-01/notes/00-models.portfolio.json').read_bytes())
    labels = ['C38','C39','C40','C41','C42','C43','C45','C46','C47','C48']
    names = {label:model for model in guide['models'] for label in model['aliases'] if label in labels}
    if len(names) != 10:
        raise ValueError('Ten exact newly reviewed aliases required')
    actions = []

    def add(kind, **values):
        actions.append(dict(kind=kind,**values))

    add('assert_text',text=report['checkpoint'])
    add('click',selector='#project-portfolio-notes')
    add('wait_text',selector='#project-portfolio-note-count',text='1 organization notes found')
    add('wait_enabled',selector='#project-portfolio-note')
    add('fill',selector='#project-portfolio-note',value='ccf5de88ce134022bb136c93c6fd3bc9c91c7260b6fa28ec97ac39d74ad0b118')
    add('click',selector='#project-portfolio-load')
    add('wait_text',selector='#project-portfolio-status',text='Loaded retained authored model organization.')
    add('wait_text',selector='#project-portfolio-count',text='205 models in this guide; 79 source reviewed.')
    add('wait_text',selector='#project-model-portfolio',text='revision 2')
    add('scroll',selector='#project-portfolio-title')
    add('screenshot',name='79-reviewed-guide.png')
    for label in labels:
        model = names[label]
        for name,value in [('group',''),('search',label),('review','source_reviewed')]:
            add('fill',selector='#project-portfolio-'+name,value=value)
        add('click',selector='#project-portfolio-filter-form button')
        add('wait_text',selector='#project-portfolio-count',text='1 match the applied filters.')
        add('click',selector='[data-lineage-node="model/'+model['id']+'"]')
        add('wait_text',selector='#project-portfolio-status',text='Exact retained model summary and captured evidence.')
        add('wait_text',selector='#project-portfolio-model-heading',text=model['label'])
        for field in ('method','purpose','experiment','outcome','lesson','reproduction'):
            add('wait_text',selector='#project-portfolio-detail',text=model['summary'][field])
        add('scroll',selector='#project-portfolio-model-heading')
        add('screenshot',name=label+'-reviewed-summary.png')
    # Exercise the new comparison control across filters and the source context.
    target = next(m for m in guide['models'] if 'C47' in m['aliases'])
    add('click',selector='#project-portfolio-detail button',text='Comparison: '+target['label'])
    add('wait_text',selector='#project-portfolio-model-heading',text=target['label'])
    add('wait_text',selector='#project-portfolio-detail',text=target['summary']['lesson'])
    add('click',selector='#project-portfolio-detail details > summary',text='Exact captured card and original receipts')
    add('wait_text',selector='#project-portfolio-detail',text='Original retained fit facts before this review')
    add('scroll',selector='#project-portfolio-model-heading')
    add('screenshot',name='C47-comparison-original-evidence.png')
    add('assert_text',text='Experiment progression')
    OUTPUT.write_text(json.dumps(actions,indent=2,ensure_ascii=False)+'\n')
    print(len(actions),'native UI actions prepared for ten complete summaries and guide revision2.')


if __name__ == '__main__':
    prepare()
