"""Prepare ten reviewed cards and a coherent next native guide revision.

The project remains unchanged. Existing review preparation owns cards/index/
overview; this packet adds a guide and references those outputs without copies.
"""
import copy
from pathlib import Path
import time

from prepare_authored_model_portfolio import summary
from prepare_method_review_revisions import prepare
from qualify_review_preparation import qualify
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
RUN = ROOT/'runs/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
BASE = RUN/'native-method-review-revisions-16'
PACKET = RUN/'native-feedback-review-publication-01'
COMBINED = DOC/'reports/captured-feedback-combined-method-reviews-01.json'
QUALIFIED = DOC/'reports/feedback-method-reviews-native-preparation-01.json'
GUIDE = 'notes/00-models.portfolio.json'
REPORTS = ['captured-pressure-budget-coverage-method-reviews-01.json',
           'captured-targeted-supplement-method-reviews-01.json',
           'captured-direction-pressure-method-reviews-01.json']
PAIRS = [('C38','C33'),('C39','C33'),('C40','C33'),('C41','C40'),('C42','C41'),
         ('C43','C44'),('C45','C44'),('C46','C45'),('C47','C45'),('C48','C47')]


def run():
    require(not any(p.exists() for p in (BASE,PACKET,COMBINED,QUALIFIED)), 'Preserve original publication preparation')
    header_path = PROJECT/'.tuldok/project.json'
    header_binding = binding(header_path)
    header = read(header_path)
    documents = {}
    for entry in header['inventory']:
        if entry['family'] != 'project_file':
            continue
        path = PROJECT/'.tuldok'/entry['path']
        require(binding(path)['sha256'] == entry['sha256'] and path.stat().st_size == entry['bytes'], 'Canonical note differs')
        value = read(path)
        if value['project_id'] == header['project_id']:
            require(value['path'] not in documents, 'Current native note path repeats')
            documents[value['path']] = (entry,value)
    reports = [read(DOC/'reports'/name) for name in REPORTS]
    rows = [row for report in reports for row in report['rows']]
    require(len(rows) == len({r['fit_item_id'] for r in rows}) == 10, 'Exactly ten distinct source reviews required')
    combined = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,
        source_reviewed_fits=10,new_model_forwards=0,native_mutations=0,clean_reproduction_qualified=False,
        overview_addition=''.join(report['overview_addition'] for report in reports),producer=binding(Path(__file__)),
        helpers=[binding(DOC/'reports'/name) for name in REPORTS],
        source_review_reports=[binding(DOC/'reports'/name) for name in REPORTS])
    for report in reports:
        for spec in [report['producer'],*report.get('helpers',[])]:
            require(binding(ROOT/spec['path']) == spec, 'Original source-review producer/helper changed')
    COMBINED.write_bytes(canonical(combined)+b'\n')
    prepare(BASE,COMBINED,project=PROJECT,checkpoint=header['checkpoint'])
    qualify(BASE,QUALIFIED,79)
    prep = read(BASE/'preparation.json')
    changes = {n['path']:n for n in prep['notes']}
    index = read(BASE/'notes/portfolio-index.json')
    entry, original = documents[GUIDE]
    old = read(PROJECT/GUIDE)
    require(original['current']['revision'] == 1
            and binding(PROJECT/GUIDE)['sha256'] == original['current']['sha256'], 'Original guide is not exact revision1')
    guide = copy.deepcopy(old)
    models = {m['id']:m for m in guide['models']}
    fit_rows = {r['item_id']:r for r in index['fits']}
    require(len(models) == len(fit_rows) == 205 and models.keys() == fit_rows.keys(), 'Guide/index fit identities differ')
    for row in rows:
        model = models[row['fit_item_id']]
        note = changes[row['note_path']]
        require(model['review'] == 'review_pending' and model['summary'] is None, 'Cannot replace an existing reviewed summary')
        model['review'] = 'source_reviewed'
        model['summary'] = summary((BASE/row['note_path']).read_text(),row['fit_item_id'])
        model['card'].update(revision=note['revision'],sha256=note['sha256'])
    names = {name:model for model in models.values() for name in model['aliases'] if name.startswith('C')}
    for name, baseline in PAIRS:
        model = names[name]
        excerpt = next((model['summary'][key] for key in ('lesson','purpose','experiment','outcome','method')
                        if baseline in model['summary'][key]),None)
        require(excerpt is not None, 'Comparison requires literal reviewed context')
        guide['relationships'].append(dict(kind='comparison',**{'from':model['id'],'to':names[baseline]['id']},
            explanation=f'{name} compared with {baseline}; retrospective source-review context, not a Trial ancestry assertion or new metric qualification.',
            evidence_model=model['id'],excerpt=excerpt))
    require(sum(m['review'] == 'source_reviewed' for m in guide['models']) == 79
            and len(guide['relationships']) == 26, 'Guide reviewed/comparison totals differ')
    guide['description'] = guide['description'].replace('69 have retrospective','79 have retrospective')
    guide_raw = canonical(guide)+b'\n'
    require(len(guide_raw) <= 1048576 and binding(header_path) == header_binding, 'Guide size/source checkpoint changed')
    PACKET.mkdir()
    target = PACKET/GUIDE
    target.parent.mkdir()
    target.write_bytes(guide_raw)
    guide_note = dict(kind='note',path=GUIDE,expected_revision=1,revision=2,
        previous_sha256=original['current']['sha256'],sha256=binding(target)['sha256'],bytes=len(guide_raw),
        prepared_path=str(target.relative_to(ROOT)))
    notes = [dict(n,prepared_path=str((BASE/n['path']).relative_to(ROOT))) for n in prep['notes']]+[guide_note]
    packet = dict(status='prepared_not_published',created_unix=time.time(),source_project_id=header['project_id'],
        source_checkpoint=header['checkpoint'],source_manifest_sha256=header_binding['sha256'],
        source_project_path=str(PROJECT.relative_to(ROOT)),notes=notes,new_reviewed_fits=10,reviewed_total=79,
        guide_revision=2,models=205,review_pending=126,literal_summary_paragraphs=474,authored_comparisons=26,
        original_receipt_suffixes_preserved=10,new_model_forwards=0,native_mutations=0,payload_copies=0,
        Trial_lineage_modified=False,full_portfolio_complete=False,producer=binding(Path(__file__)),
        evidence=[binding(BASE/'preparation.json'),binding(QUALIFIED),binding(COMBINED),binding(PROJECT/GUIDE)])
    (PACKET/'preparation.json').write_bytes(canonical(packet)+b'\n')
    require(binding(header_path) == header_binding, 'Canonical project changed during preparation')
    print('Prepared13notes:10reviews,index,overview,and coherent79-review guide revision2. No native mutation.')


if __name__ == '__main__':
    run()
