"""Prepare six new reviews and three append-only matched-work qualifications."""
from pathlib import Path
import time

from prepare_method_review_revisions import prepare, encode, sha, verified_review_receipt
from review_detached_C37_fits import ROOT, DOC, PROJECT, binding, read, require

OUTPUT = ROOT / 'runs/model-portfolio-migration-v1/native-method-review-revisions-09'
REVIEW = DOC / 'reports/captured-model-method-reviews-08.json'
HEADING = b'\n\n## Follow-up: matched pressure correction budgets\n\n'


def run():
    require(not OUTPUT.exists() and not REVIEW.exists(), 'Fresh bounded preparation required')
    header_path = PROJECT / '.tuldok/project.json'
    header_raw = header_path.read_bytes()
    header = read(header_path)
    numerical_path = DOC / 'reports/pressure-correction-work-01.json'
    numerical = read(numerical_path)
    require(numerical['status'] == 'all_7680_retained_pressure_policy_fields_independently_rescored',
            'Independent numerical matrix required')
    source_paths = [DOC / f'reports/captured-detached-{parent}-method-reviews-01.json'
                    for parent in ('C37','C44','C53')]
    source_reviews = [read(path) for path in source_paths]
    for source in source_reviews:
        require(binding(ROOT / source['producer']['path'])['sha256'] == source['producer']['sha256'],
                'Original source-review producer changed')
    rows = []
    for family in source_reviews[1:]:
        for source_row in family['rows']:
            row = dict(source_row)
            row['lesson'] += (' The subsequent matched-work CUDA matrix confirms these parent regressions '
                'at every tested one/two/four/eight correction budget. C37 benefits at two/four/eight '
                'corrections, while its worst pressure error worsens at one correction; its saved parent '
                'default uses one and its fine-tunes use two. Keep metric and actual work explicit.')
            row['independent_accuracy'] = binding(numerical_path)
            rows.append(row)
    addition = ('\n\n## Detached fine-tunes at matched correction budgets\n\n'
        'Six further source reviews explain the C44 and C53 U/A/D continuations. '
        'All use the audited eight-pattern fitting family and retain distinct original receipts. '
        'All regress against their own parents on both worst pressure and equation errors at matched '
        'one/two/four/eight correction budgets. C37 improvements persist at two/four/eight corrections, '
        'but its fine-tunes worsen worst pressure error at one correction. Its original default '
        'comparison changed one to two corrections; follow-up text now qualifies all three C37 cards. '
        'The unnamed C53-U fit shares weight bytes and all 128 tested default pressure outputs with '
        'C66, without sharing its fit identity. The matrix retained and independently rescored '
        '7,680 fields. No timing, unseen-scene generalization, physical advance or clean-retraining '
        'claim is implied. GPU pressure-call timings remain separate from full-generator timings.\n')
    review = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,
        overview_addition=addition,producer=binding(Path(__file__)),
        original_source_review_reports=[binding(path) for path in source_paths],
        independent_accuracy=binding(numerical_path),new_model_forwards=0,native_mutations=0,
        clean_reproduction_qualified=False)
    REVIEW.write_bytes(encode(review))
    prepare(OUTPUT,REVIEW,project=PROJECT,checkpoint=header['checkpoint'])
    prepared = read(OUTPUT/'preparation.json')
    index = read(OUTPUT/'notes/portfolio-index.json')
    index_cards = {fit['item_id']:fit for fit in index['fits']}
    captured = {}
    for entry in header['inventory']:
        if entry['family'] == 'project_file':
            document = read(PROJECT/'.tuldok'/entry['path'])
            if document['project_id'] == header['project_id']:
                captured[document['path']] = document
    addenda = []
    for source_row, identifier in zip(source_reviews[0]['rows'],('C37-U','C37-A','C37-D'),strict=True):
        path = source_row['note_path']
        raw = (PROJECT/path).read_bytes()
        current = captured[path]['current']
        require(current['text'].encode()==raw and current['sha256']==sha(raw)
                and HEADING not in raw and raw.count(verified_review_receipt(source_row))==1,
                'Existing reviewed C37 card changed or already has this qualification')
        values = numerical['models'][identifier]['summary']
        parent = numerical['models']['C37']['summary']
        text = ('Authored follow-up from the independently rescored 12-checkpoint CUDA matrix. '
            'The saved C37 parent uses one pressure correction and this fine-tune uses two. '
            'The original default-to-default result therefore combines training and additional work. '
            'At matched two/four/eight corrections this fine-tune improves both worst equation and '
            'mean-free pressure relative L2 errors. At one correction it improves worst equation '
            'error but worsens worst pressure error. These statements concern 128 consumed original '
            'pressure equations, not physical time steps, unseen scenes, drift or timing.\n\n'
            '| Corrections | Parent worst pressure relative L2 | Fine-tune worst pressure relative L2 |\n'
            '| --- | ---: | ---: |\n')
        for count in (1,2,4,8):
            policy = f'fixed_{count}'
            text += f"| {count} | {parent[policy]['worst']['pressure_mean_free_relative_l2']:.17g} | {values[policy]['worst']['pressure_mean_free_relative_l2']:.17g} |\n"
        text += '\nIndependent evidence: ' + binding(numerical_path)['path'] + ' SHA256 ' + binding(numerical_path)['sha256'] + '.\n'
        amended = raw + HEADING + text.encode()
        target = OUTPUT/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(amended)
        note = dict(path=path,kind='note',previous_sha256=sha(raw),expected_revision=current['revision'],
            revision=current['revision']+1,sha256=sha(amended),bytes=len(amended),fit_item_id=source_row['fit_item_id'],
            amendment_kind='append_only_matched_correction_budget_qualification',prior_review_bytes=len(raw),
            original_receipt_sha256=source_row['original_receipt']['sha256'])
        prepared['notes'].append(note)
        index_cards[source_row['fit_item_id']]['card_sha256'] = note['sha256']
        addenda.append(dict(path=path,fit_item_id=source_row['fit_item_id'],
            previous_sha256=sha(raw),independent_accuracy=binding(numerical_path)))
    index['sources'].setdefault('method_review_addenda',[]).append(dict(source=binding(REVIEW),cards=addenda))
    index_raw = encode(index)
    (OUTPUT/'notes/portfolio-index.json').write_bytes(index_raw)
    index_note = next(note for note in prepared['notes'] if note['path']=='notes/portfolio-index.json')
    index_note.update(sha256=sha(index_raw),bytes=len(index_raw))
    prepared.update(append_only_qualifications=3,total_reviewed_cards_after_adoption=32,
        producer=binding(Path(__file__)),append_only_cards=addenda)
    require(len(prepared['notes'])==11 and len({n['path'] for n in prepared['notes']})==11
            and sum(f.get('method_review_status')=='captured_source_and_receipt_reviewed' for f in index['fits'])==32
            and all(note['bytes']<=1048576 for note in prepared['notes']), 'Bounded eleven-note preparation differs')
    require(header_path.read_bytes()==header_raw,'Source project changed during preparation')
    (OUTPUT/'preparation.json').write_bytes(encode(prepared))
    (OUTPUT/'adopt-items.json').write_bytes(encode(dict(items=[
        {k:n[k] for k in ('kind','path','expected_revision')} | dict(expected_sha256=n['sha256'])
        for n in prepared['notes']])))
    print('Eleven notes prepared: six new reviews, three C37 qualifications, index and overview.')


if __name__ == '__main__':
    run()
