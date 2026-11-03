"""Check prepared C06--C10 cards against original receipts and physical splits."""
import copy
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/plans/model-portfolio-migration-v1/reports'
PROJECT = ROOT / 'projects/rheon-model-portfolio-unified-04'
PREPARED = ROOT / 'runs/model-portfolio-migration-v1/native-method-review-revisions-12'
OUTPUT = DOC / 'water-learning-reviews-preparation-qualification-01.json'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def run():
    require(not OUTPUT.exists(), 'Preserve original qualification')
    prep = read(PREPARED/'preparation.json')
    header_path = PROJECT/'.tuldok/project.json'
    require(sha(header_path) == prep['source_manifest_sha256'], 'Pinned project changed')
    header = read(header_path)
    require(header['checkpoint'] == prep['source_checkpoint'], 'Wrong source checkpoint')
    review_path = ROOT/prep['review_path']
    require(sha(review_path) == prep['review_sha256'], 'Prepared source reviews changed')
    authored = read(review_path)
    require(sha(ROOT/authored['producer']['path']) == authored['producer']['sha256'], 'Review producer changed')
    reviews = {r['note_path']:r for r in authored['rows']}
    require(len(reviews) == 5 and len(prep['notes']) == 7, 'Expected five cards plus overview/index')
    current = {}
    for entry in header['inventory']:
        if entry['family'] == 'project_file':
            path = PROJECT/'.tuldok'/entry['path']
            require(sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], 'Canonical file changed')
            record = read(path)
            if record['project_id'] == header['project_id']:
                current[record['path']] = record['current']
    originals = {}
    notes = {n['path']:n for n in prep['notes']}
    for note in prep['notes']:
        old = (PROJECT/note['path']).read_bytes()
        new = (PREPARED/note['path']).read_bytes()
        canonical = current[note['path']]
        require(old == canonical['text'].encode() and sha(PROJECT/note['path']) == canonical['sha256'] == note['previous_sha256']
                and note['expected_revision'] == canonical['revision'] and note['revision'] == canonical['revision']+1
                and sha(PREPARED/note['path']) == note['sha256'] and len(new) == note['bytes'] <= 1048576,
                'Prepared note/canonical revision or exact text differs')
        if note['path'] not in reviews:
            continue
        review = reviews[note['path']]
        for field in ('original_receipt','history','source_witness','training_population','original_configuration','original_recipe','retained_accuracy','common_score_source'):
            spec = review[field]
            require(sha(ROOT/spec['path']) == spec['sha256'], 'Original reviewed evidence changed')
        receipt_raw = (ROOT/review['original_receipt']['path']).read_bytes()
        receipt = json.loads(receipt_raw)
        require(old.count(receipt_raw) == new.count(receipt_raw) == 1
                and new.endswith(old.split(b'## Retained facts and comparison evidence\n\n',1)[1]),
                'Original complete receipt/configuration suffix changed')
        population = read(ROOT/review['training_population']['path'])['parents']
        trains = [p for p in population if p['entry']['split'] == 'train']
        validation = [p for p in population if p['entry']['split'] == 'validation']
        require(len(trains) == review['physical_train_trajectories']
                and sum(len(p['frames']) for p in trains) == review['physical_train_frames']
                and len(validation) == 64 and sum(len(p['frames']) for p in validation) == 128,
                'Authored physical training/validation counts differ')
        actual_density = {str(density):sum(1/p['cell_size_m'] == density for p in trains)
                          for density in sorted({1/p['cell_size_m'] for p in trains})}
        require(actual_density == review['training_density_scene_counts'], 'Training/validation densities were conflated')
        score = read(ROOT/review['common_score_source']['path'])['models'][review['named_checkpoint']]
        require(receipt['outputs']['weights.npz']['sha256'] == score['weights_sha256'], 'Different checkpoint reviewed')
        for metric in ('common_equation_relative_l2','pressure_mean_free_relative_l2'):
            require(format(score['worst'][metric],'.17g') in review['outcome'], 'Common numerical outcome was misquoted')
        require(receipt['mode'] == 'train' and receipt['status'] == 'complete'
                and not review['clean_reproduction_qualified'] and review['new_model_forwards'] == 0,
                'Source review cannot qualify clean reproduction/new benchmarking')
        originals[review['fit_item_id']] = review['original_receipt']['sha256']
    old_index = read(PROJECT/'notes/portfolio-index.json')
    new_index = read(PREPARED/'notes/portfolio-index.json')
    reconstructed = copy.deepcopy(new_index)
    for old, new in zip(old_index['fits'],reconstructed['fits'],strict=True):
        require(old['item_id'] == new['item_id'], 'Fit order/identity changed')
        if old['item_id'] in originals:
            require(new['card_sha256'] == notes[new['card_path']]['sha256']
                    and new['method_review_status'] == 'captured_source_and_receipt_reviewed', 'Updated card not bound')
            for key in ('card_sha256','method_review_status','clean_reproduction_qualified'):
                if key in old:
                    new[key] = old[key]
                else:
                    new.pop(key,None)
    for key in ('method_reviews','method_review_batches'):
        if key in old_index['sources']:
            reconstructed['sources'][key] = old_index['sources'][key]
        else:
            reconstructed['sources'].pop(key,None)
    require(reconstructed == old_index, 'Unrelated scientific/index facts changed')
    require(len(new_index['fits']) == 205 and sum(f.get('method_review_status')
            == 'captured_source_and_receipt_reviewed' for f in new_index['fits']) == 43,
            'Expected43prepared reviewed cards; native still38')
    require((PREPARED/'notes/portfolio-overview.md').read_bytes().startswith((PROJECT/'notes/portfolio-overview.md').read_bytes()),
            'Overview must append to previous authored context')
    require(read(PREPARED/'adopt-items.json')['items'] == [
        {k:note[k] for k in ('kind','path','expected_revision')}|dict(expected_sha256=note['sha256']) for note in prep['notes']],
        'Exact guarded adoption request differs')
    require(sha(header_path) == prep['source_manifest_sha256'], 'Project changed during qualification')
    value = dict(status='five_water_learning_reviews_prepared_with_exact_original_evidence',created_unix=time.time(),
        source_checkpoint=header['checkpoint'],original_receipts_exact=originals,
        physical_optimizer_population_independently_checked=True,common_metric_outcomes_exact=True,
        all_unrelated_index_facts_exact=True,prepared_reviewed_fit_cards=43,native_reviewed_fit_cards=38,
        prepared_notes=7,prepared_note_bytes=sum(n['bytes'] for n in prep['notes']),
        native_adoption_complete=False,clean_reproduction_qualified=False,full_portfolio_complete=False,
        new_model_forwards=0,new_reference_generation=False,
        evidence=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in
                  (review_path, PREPARED/'preparation.json',PREPARED/'adopt-items.json')],producer_sha256=sha(Path(__file__)))
    OUTPUT.write_text(json.dumps(value,indent=2)+'\n')
    print(value['status'], flush=True)


if __name__ == '__main__':
    run()
