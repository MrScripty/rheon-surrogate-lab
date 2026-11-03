"""Prepare real guide/card UI checks only after the40-note custody audit passes."""
from pathlib import Path

from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
JOB = RUN/'method-review-native-adoption-16'
OUTPUT = JOB/'browser-actions.json'


def run():
    require(not OUTPUT.exists(), 'Preserve original37card UI actions')
    audit = read(ROOT/'docs/plans/model-portfolio-migration-v1/reports/native-205-method-review-adoption-01.json')
    require(audit['status'] == 'native_twentysix_note_revisions_exact_and_scientific_metadata_unchanged'
            and audit['reviewed_fit_cards'] == 205 and audit['changed_note_records'] == 26,
            'Actual native acknowledgement and complete custody required')
    prepared = RUN/'native-complete-source-review-batch-02'
    packet = read(prepared/'preparation.json')
    guide = read(prepared/'notes/00-models.portfolio.json')
    current = ROOT/'projects/rheon-model-portfolio-unified-04/notes/00-models.portfolio.json'
    require(binding(current)['sha256'] == binding(prepared/'notes/00-models.portfolio.json')['sha256'],
            'Actual native guide differs from the qualified proposal')
    ids = {n['fit_item_id'] for n in packet['notes'] if 'fit_item_id' in n}
    require(len(ids) == 23, 'All23new summaries required')
    actions = []

    def add(kind, **values):
        actions.append(dict(kind=kind,**values))

    add('assert_text',text=audit['checkpoint'])
    add('click',selector='#project-portfolio-notes')
    add('wait_text',selector='#project-portfolio-note-count',text='1 organization notes found')
    add('wait_enabled',selector='#project-portfolio-note')
    add('fill',selector='#project-portfolio-note',value='ccf5de88ce134022bb136c93c6fd3bc9c91c7260b6fa28ec97ac39d74ad0b118')
    add('click',selector='#project-portfolio-load')
    add('wait_text',selector='#project-portfolio-status',text='Loaded retained authored model organization.')
    add('wait_text',selector='#project-portfolio-count',text='205 models in this guide; 205 source reviewed.')
    add('wait_text',selector='#project-model-portfolio',text='revision 6')
    add('scroll',selector='#project-portfolio-title')
    add('screenshot',name='205-reviewed-guide.png')
    for model in guide['models']:
        if model['id'] not in ids:
            continue
        query = model['label'].lower()
        matching = [m for m in guide['models'] if m['review'] == 'source_reviewed'
                    and any(query in v.lower() for v in [m['label'],*m['aliases']])]
        require(model in matching and len(matching) <= 20, 'Exact model must be visible on first native filtered page')
        for name,value in [('group',''),('search',model['label']),('review','source_reviewed')]:
            add('fill',selector='#project-portfolio-'+name,value=value)
        add('click',selector='#project-portfolio-filter-form button')
        add('wait_text',selector='#project-portfolio-count',text=f'{len(matching)} match the applied filters.')
        add('click',selector='[data-lineage-node="model/'+model['id']+'"]')
        add('wait_text',selector='#project-portfolio-status',text='Exact retained model summary and captured evidence.')
        add('wait_text',selector='#project-portfolio-model-heading',text=model['label'])
        for field in ('method','purpose','experiment','outcome','lesson','reproduction'):
            add('wait_text',selector='#project-portfolio-detail',text=model['summary'][field])
        add('scroll',selector='#project-portfolio-model-heading')
        add('screenshot',name=model['id']+'-summary.png')
    # Existing comparison controls and original receipt disclosure must coexist
    # with the enlarged guide; this is not a fabricated new ancestry edge.
    model = next(m for m in guide['models'] if 'C48' in m['aliases'])
    target = next(m for m in guide['models'] if 'C47' in m['aliases'])
    for name,value in [('group',''),('search','C48'),('review','source_reviewed')]:
        add('fill',selector='#project-portfolio-'+name,value=value)
    add('click',selector='#project-portfolio-filter-form button')
    add('click',selector='[data-lineage-node="model/'+model['id']+'"]')
    add('wait_text',selector='#project-portfolio-model-heading',text=model['label'])
    add('click',selector='#project-portfolio-detail button',text='Comparison: '+target['label'])
    add('wait_text',selector='#project-portfolio-model-heading',text=target['label'])
    add('wait_text',selector='#project-portfolio-detail',text=target['summary']['lesson'])
    add('click',selector='#project-portfolio-detail details > summary',text='Exact captured card and original receipts')
    add('wait_text',selector='#project-portfolio-detail',text='Original retained fit facts before this review')
    add('scroll',selector='#project-portfolio-model-heading')
    add('screenshot',name='retained-comparison-and-receipts.png')
    add('assert_text',text='Experiment progression')
    OUTPUT.write_bytes(canonical(actions)+b'\n')
    print(len(actions),'actual UI actions prepared; all23new summaries, guide6 and retained comparison controls.')


if __name__ == '__main__':
    run()
