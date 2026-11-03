"""Independently check ten proposed cards, real fit populations and index isolation."""
import copy
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
PREPARED = ROOT/'runs/model-portfolio-migration-v1/native-method-review-revisions-13'
OUTPUT = ROOT/'docs/plans/model-portfolio-migration-v1/reports/learning-conditioning-reviews-qualification-01.json'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


def run():
    require(not OUTPUT.exists(), 'Preserve original independent qualification')
    prep = read(PREPARED/'preparation.json')
    header = PROJECT/'.tuldok/project.json'
    project_header = read(header)
    require(sha(header) == prep['source_manifest_sha256'] and project_header['checkpoint'] == prep['source_checkpoint'],
            'Pinned source project changed')
    review_path = ROOT/prep['review_path']
    require(sha(review_path) == prep['review_sha256'], 'Authored source reviews changed')
    report = read(review_path)
    for spec in report['source_reports']:
        path = ROOT/spec['path']
        require(sha(path) == spec['sha256'], 'Original review batch changed')
        value = read(path)
        for key in ['producer'] + (['helper'] if 'helper' in value else []):
            require(sha(ROOT/value[key]['path']) == value[key]['sha256'], 'Review source/helper changed')
    rows = report['rows']
    require(len(rows) == 10 and {r['named_checkpoint'] for r in rows} == {f'C{i:02}' for i in range(6,16)},
            'Exact C06--C15 fit population required')
    current = {}
    for record in project_header['inventory']:
        if record['family'] == 'project_file':
            path = PROJECT/'.tuldok'/record['path']
            require(sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], 'Canonical file differs')
            value = read(path)
            if value['project_id'] == project_header['project_id']:
                current[value['path']] = value['current']
    notes = {n['path']:n for n in prep['notes']}
    require(len(notes) == 12, 'Ten cards plus original overview/index required')
    for path, note in notes.items():
        old, new = (PROJECT/path).read_bytes(), (PREPARED/path).read_bytes()
        owned = current[path]
        require(old == owned['text'].encode() and sha(PROJECT/path) == note['previous_sha256'] == owned['sha256']
                and note['expected_revision'] == owned['revision'] and note['revision'] == owned['revision']+1
                and sha(PREPARED/path) == note['sha256'] and len(new) == note['bytes'] <= 1048576,
                'Guarded revision, canonical text or prepared bytes differs')
    counts, initialized = {}, {}
    for row in rows:
        for spec in [row['original_receipt'],row['history'],row['source_witness'],row['training_population'],
                     row['original_configuration'],row['original_recipe'],row['retained_accuracy'],row['common_score_source'],
                     *row['sources'],*row['training_inputs']]:
            require(sha(ROOT/spec['path']) == spec['sha256'], 'Original fit evidence differs')
        receipt_raw = (ROOT/row['original_receipt']['path']).read_bytes()
        receipt = json.loads(receipt_raw)
        old, new = (PROJECT/row['note_path']).read_bytes(), (PREPARED/row['note_path']).read_bytes()
        require(old.count(receipt_raw) == new.count(receipt_raw) == 1
                and new.endswith(old.split(b'## Retained facts and comparison evidence\n\n',1)[1]),
                'Exact original receipt/configuration suffix differs')
        split = read(ROOT/row['training_population']['path'])
        train = [p for p in split['parents'] if p['entry']['split'] == 'train']
        val = [p for p in split['parents'] if p['entry']['split'] == 'validation']
        label = row['named_checkpoint']
        expected = {'C06':(256,512),'C08':(256,512),'C09':(256,256),'C10':(1024,1024)}.get(label,(1,1))
        require((len(train),sum(len(p['frames']) for p in train)) == expected
                and len(val) == 64 and sum(len(p['frames']) for p in val) == 128,
                'Independent actual optimizer/validation population differs')
        densities = sorted({1/p['cell_size_m'] for p in train})
        require(densities == ([8] if label in {'C07','C11','C12','C13','C14','C15'} else [8,16]),
                'Validation densities were conflated with training')
        score = read(ROOT/row['common_score_source']['path'])['models'][label]
        require(receipt['mode'] == 'train' and receipt['status'] == 'complete'
                and receipt['outputs']['weights.npz']['sha256'] == score['weights_sha256'], 'Wrong trained candidate')
        for metric in ('common_equation_relative_l2','pressure_mean_free_relative_l2'):
            require(format(score['worst'][metric],'.17g') in row['outcome'], 'Quoted common error differs')
        initialized[label] = read(ROOT/row['history']['path'])['checkpoints']['checkpoint-000.npz']['inventory']
        counts[label] = dict(train_trajectories=len(train),train_frames=expected[1],training_cells_per_metre=densities)
    require(all(value == initialized['C06'] for value in initialized.values()), 'Claimed shared initial trainable tensors differ')
    old_index, new_index = read(PROJECT/'notes/portfolio-index.json'), read(PREPARED/'notes/portfolio-index.json')
    reconstructed = copy.deepcopy(new_index)
    selected = {r['fit_item_id'] for r in rows}
    for old, new in zip(old_index['fits'],reconstructed['fits'],strict=True):
        require(old['item_id'] == new['item_id'], 'Scientific fit identity/order changed')
        if old['item_id'] in selected:
            require(new['card_sha256'] == notes[new['card_path']]['sha256']
                    and new['method_review_status'] == 'captured_source_and_receipt_reviewed', 'New card is not bound')
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
    require(reconstructed == old_index and len(new_index['fits']) == 205
            and sum(f.get('method_review_status') == 'captured_source_and_receipt_reviewed' for f in new_index['fits']) == 48,
            'Unrelated index/scientific facts changed')
    require((PREPARED/'notes/portfolio-overview.md').read_bytes().startswith((PROJECT/'notes/portfolio-overview.md').read_bytes()),
            'Previous overview context was replaced')
    require(read(PREPARED/'adopt-items.json')['items'] == [{k:n[k] for k in ('kind','path','expected_revision')}
            | dict(expected_sha256=n['sha256']) for n in prep['notes']], 'Native request differs')
    require(sha(header) == prep['source_manifest_sha256'], 'Project changed during qualification')
    value = dict(status='ten_learning_conditioning_reviews_prepared_with_exact_original_evidence',created_unix=time.time(),
        source_checkpoint=prep['source_checkpoint'],actual_optimizer_populations=counts,
        all_original_receipt_configuration_suffixes_exact=True,all_unrelated_index_facts_exact=True,
        all_trainable_initial_inventories_equal=True,prepared_reviewed_fit_cards=48,native_reviewed_fit_cards=38,
        new_reviewed_fit_cards=10,prepared_notes=12,prepared_bytes=sum(n['bytes'] for n in prep['notes']),
        native_adoption_complete=False,clean_retraining_qualified=False,full_portfolio_complete=False,
        new_model_forwards=0,new_reference_generation=False,
        evidence=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in
                  (PREPARED/'preparation.json', PREPARED/'adopt-items.json',review_path)],producer_sha256=sha(Path(__file__)))
    OUTPUT.write_text(json.dumps(value,indent=2)+'\n')
    print(value['status'],flush=True)


if __name__ == '__main__':
    run()
