"""Verify six current-base parent/early-fit reviews without changing scientific index facts."""
import copy
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'projects/rheon-model-portfolio-unified-04'
PREPARED = ROOT / 'runs/model-portfolio-migration-v1/native-method-review-revisions-11'
REPORT = ROOT / 'docs/plans/model-portfolio-migration-v1/reports/six-parent-early-review-preparation-qualification-02.json'


def read(path):
    return json.loads(path.read_bytes())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run():
    require(not REPORT.exists(),'Preserve the original independent qualification')
    prepared = read(PREPARED/'preparation.json')
    raw_header = (PROJECT/'.tuldok/project.json').read_bytes()
    header = json.loads(raw_header)
    require(sha(raw_header)==prepared['source_manifest_sha256']
            and header['checkpoint']==prepared['source_checkpoint'], 'Original checkpoint differs')
    review_path = ROOT/prepared['review_path']
    require(sha(review_path.read_bytes())==prepared['review_sha256'],'Authored review source changed')
    reviews = {row['note_path']:row for row in read(review_path)['rows']}
    require(len(reviews)==6 and len(prepared['notes'])==8,'Expected six source reviews and eight notes')
    captured = {}
    for entry in header['inventory']:
        if entry['family']=='project_file':
            raw = (PROJECT/'.tuldok'/entry['path']).read_bytes()
            require(sha(raw)==entry['sha256'] and len(raw)==entry['bytes'],'Canonical note differs')
            document = json.loads(raw)
            if document['project_id']==header['project_id']:
                captured[document['path']] = document
    amended = []
    checked_receipts = []
    for note in prepared['notes']:
        old = (PROJECT/note['path']).read_bytes()
        new = (PREPARED/note['path']).read_bytes()
        current = captured[note['path']]['current']
        require(sha(old)==note['previous_sha256']==current['sha256'] and old==current['text'].encode()
                and note['expected_revision']==current['revision'] and note['revision']==current['revision']+1
                and sha(new)==note['sha256'] and len(new)==note['bytes']<=1048576, 'Prepared note precondition/value differs')
        if note['path'] in reviews:
            review = reviews[note['path']]
            receipt = (ROOT/review['original_receipt']['path']).read_bytes()
            require(sha(receipt)==review['original_receipt']['sha256']
                    and old.count(receipt)==new.count(receipt)==1, 'Exact original receipt lost or duplicated')
            marker = b'## Retained facts and comparison evidence\n\n'
            require(old.count(marker)==1 and new.endswith(old.split(marker,1)[1]), 'Original complete receipt/configuration suffix differs')
            checked_receipts.append(review['fit_item_id'])
        elif note.get('amendment_kind'):
            require(note['amendment_kind']=='append_only_matched_correction_budget_qualification'
                    and new.startswith(old) and len(old)==note['prior_review_bytes']
                    and new[len(old):].startswith(b'\n\n## Follow-up: matched pressure correction budgets\n\n'),
                    'Existing reviewed card was replaced rather than qualified')
            amended.append(note['fit_item_id'])
    require(len(checked_receipts)==6 and len(amended)==0,'Expected six original receipts and no changes to other reviewed cards')
    correction = read(ROOT/'docs/plans/model-portfolio-migration-v1/reports/full-state-timing-context-correction-preparation-01.json')
    overview = (PREPARED/'notes/portfolio-overview.md').read_bytes()
    prior_overview = correction['previous_prepared_overview']
    require(sha(overview[:prior_overview['bytes']]) == prior_overview['sha256']
            and b'Both original long-reference captures' in overview[prior_overview['bytes']:]
            and b'**train**' in overview[prior_overview['bytes']:]
            and correction['preparation_sha256'] == sha((PREPARED/'preparation.json').read_bytes()),
            'Explicit split correction must append to the original prepared overview')
    original_index = read(PROJECT/'notes/portfolio-index.json')
    index = read(PREPARED/'notes/portfolio-index.json')
    notes = {note.get('fit_item_id'):note for note in prepared['notes'] if note.get('fit_item_id')}
    reconstructed = copy.deepcopy(index)
    for old, new in zip(original_index['fits'],reconstructed['fits'],strict=True):
        require(old['item_id']==new['item_id'],'Scientific fit order or identity changed')
        if old['item_id'] in notes:
            require(new['card_sha256']==notes[old['item_id']]['sha256'],'Index does not bind the proposed card')
            for key in ('card_sha256','method_review_status','clean_reproduction_qualified'):
                if key in old:
                    new[key]=old[key]
                else:
                    new.pop(key,None)
    for key in ('method_reviews','method_review_batches','method_review_addenda'):
        if key in original_index['sources']:
            reconstructed['sources'][key]=original_index['sources'][key]
        else:
            reconstructed['sources'].pop(key,None)
    require(reconstructed==original_index,'Unrelated scientific index facts changed')
    require(len(index['fits'])==205 and sum(f.get('method_review_status')=='captured_source_and_receipt_reviewed'
            for f in index['fits'])==38,'Expected38reviewed fit cards')
    expected=[{key:note[key] for key in ('kind','path','expected_revision')}|dict(expected_sha256=note['sha256'])
              for note in prepared['notes']]
    require(read(PREPARED/'adopt-items.json')['items']==expected,'Native adoption request differs')
    require((PROJECT/'.tuldok/project.json').read_bytes()==raw_header,'Project changed during qualification')
    report=dict(status='six_parent_and_early_reviews_prepared_with_exact_original_evidence',created_unix=time.time(),
        checkpoint=header['checkpoint'],preparation_sha256=sha((PREPARED/'preparation.json').read_bytes()),
        new_source_reviewed_cards=6,append_only_qualified_cards=0,prepared_notes=8,
        original_receipts_preserved=checked_receipts,prior_card_bytes_preserved=amended,
        all_unrelated_index_facts_exact=True,reviewed_fit_cards_after_adoption=38,
        producer_sha256=sha(Path(__file__).read_bytes()),native_mutations=0,new_model_forwards=0,
        native_adoption_complete=False,full_portfolio_complete=False)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'])


if __name__=='__main__':
    run()
