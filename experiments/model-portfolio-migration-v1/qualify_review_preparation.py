"""Verify one guarded authored-review batch against the native current records.

Family-specific source review owns architectural interpretation. This checks
literal evidence, physical counts, numerical quotes and native revision/index
isolation without importing a model or mutating the project.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1/reports'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


def qualify(prepared, output, expected_total):
    require(prepared.is_relative_to(RUN) and output.is_relative_to(DOC) and not output.exists(),
            'Task-owned preparation and fresh qualification report required')
    prep = read(prepared/'preparation.json')
    header_path = PROJECT/'.tuldok/project.json'
    header = read(header_path)
    require(sha(header_path) == prep['source_manifest_sha256'] and header['checkpoint'] == prep['source_checkpoint'],
            'Pinned canonical checkpoint differs')
    review_path = ROOT/prep['review_path']
    require(sha(review_path) == prep['review_sha256'], 'Source-review report changed')
    review = read(review_path)
    for spec in [review['producer'],*review.get('helpers',[])]:
        require(sha(ROOT/spec['path']) == spec['sha256'], 'Reviewed producer/helper changed')
    rows = review['rows']
    require(len({r['fit_item_id'] for r in rows}) == len(rows) and len(prep['notes']) == len(rows)+2,
            'Distinct reviewed fits plus overview/index required')
    current = {}
    for entry in header['inventory']:
        if entry['family'] == 'project_file' and entry['namespace'] == header['project_id']:
            path = PROJECT/'.tuldok'/entry['path']
            require(sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], 'Canonical note differs')
            document = read(path)
            if document['project_id'] == header['project_id']:
                require(document['path'] not in current, 'Current authored note path repeats')
                current[document['path']] = document['current']
    notes = {n['path']:n for n in prep['notes']}
    for path, note in notes.items():
        raw = (PROJECT/path).read_bytes()
        owned = current[path]
        require(raw == owned['text'].encode() and sha(PROJECT/path) == owned['sha256'] == note['previous_sha256']
                and note['expected_revision'] == owned['revision'] and note['revision'] == owned['revision']+1
                and sha(prepared/path) == note['sha256'] and (prepared/path).stat().st_size == note['bytes'] <= 1048576,
                'Prepared bytes or exact canonical revision differs')
    counts = []
    for row in rows:
        for spec in [row['original_receipt'],row['history'],row['source_witness'],row['training_population'],
                     row['original_configuration'],row['original_recipe'],row['retained_accuracy'],row['common_score_source'],
                     *row['sources'],*row['training_inputs']]:
            require(sha(ROOT/spec['path']) == spec['sha256'], 'Reviewed original evidence changed')
        receipt_raw = (ROOT/row['original_receipt']['path']).read_bytes()
        old, new = (PROJECT/row['note_path']).read_bytes(), (prepared/row['note_path']).read_bytes()
        require(old.count(receipt_raw) == new.count(receipt_raw) == 1
                and new.endswith(old.split(b'## Retained facts and comparison evidence\n\n',1)[1]),
                'Original complete receipt/configuration suffix changed')
        parents = read(ROOT/row['training_population']['path'])['parents']
        actual = {f'actual_{split}_{kind}':sum((1 if kind == 'trajectories' else len(p['frames']))
            for p in parents if p['entry']['split'] == split) for split in ('train','validation') for kind in ('trajectories','frames')}
        require(all(row[k] == v for k,v in actual.items()), 'Physical fitting/validation counts differ')
        score = read(ROOT/row['common_score_source']['path'])['models'][row['named_checkpoint']]
        require(json.loads(receipt_raw)['outputs']['weights.npz']['sha256'] == score['weights_sha256'], 'Another checkpoint was reviewed')
        for metric in ('common_equation_relative_l2','pressure_mean_free_relative_l2'):
            require(format(score['worst'][metric],'.17g') in row['outcome'], 'Common metric was misquoted')
        if 'retained_timing' in row:
            spec = row['retained_timing']
            require(sha(ROOT/spec['path']) == spec['sha256'], 'Retained timing changed')
            timing = read(ROOT/spec['path'])['models'][row['named_checkpoint']]['summary']
            require(all(format(timing[k],'.17g') in row['outcome'] for k in ('median_case_median_ms','worst_case_median_ms')),
                    'Complete-call timing statistic was misquoted')
        counts.append(dict(fit_item_id=row['fit_item_id'],**actual))
    old_index, new_index = read(PROJECT/'notes/portfolio-index.json'), read(prepared/'notes/portfolio-index.json')
    reconstructed = copy.deepcopy(new_index)
    selected = {r['fit_item_id'] for r in rows}
    for old, new in zip(old_index['fits'],reconstructed['fits'],strict=True):
        require(old['item_id'] == new['item_id'], 'Scientific fit identity/order changed')
        if old['item_id'] in selected:
            require(new['card_sha256'] == notes[new['card_path']]['sha256']
                    and new['method_review_status'] == 'captured_source_and_receipt_reviewed', 'Authored card not bound')
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
            and sum(f.get('method_review_status') == 'captured_source_and_receipt_reviewed' for f in new_index['fits']) == expected_total,
            'Unrelated index facts or total reviewed population changed')
    require((prepared/'notes/portfolio-overview.md').read_bytes().startswith((PROJECT/'notes/portfolio-overview.md').read_bytes()),
            'Original overview replaced')
    require(read(prepared/'adopt-items.json')['items'] == [{k:n[k] for k in ('kind','path','expected_revision')}
            | dict(expected_sha256=n['sha256']) for n in prep['notes']], 'Guarded adoption request differs')
    require(sha(header_path) == prep['source_manifest_sha256'], 'Project changed during independent qualification')
    value = dict(status='guarded_review_batch_prepared_with_exact_original_evidence',created_unix=time.time(),
        checkpoint=header['checkpoint'],new_reviewed_fits=len(rows),reviewed_fit_cards_after_adoption=expected_total,
        original_receipts_and_history_suffixes_exact=True,unrelated_index_and_scientific_facts_exact=True,
        physical_counts=counts,common_accuracy_and_timing_quotes_exact=True,native_adoption_complete=False,
        clean_reproduction_qualified=False,new_model_forwards=0,full_portfolio_complete=False,
        evidence=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in
                  (prepared/'preparation.json',prepared/'adopt-items.json',review_path)],producer_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(value,indent=2)+'\n')
    print(value['status'],len(rows),'new fit reviews',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--expected-reviewed-total',type=int,required=True)
    args = parser.parse_args()
    qualify(args.preparation.resolve(),args.report.resolve(),args.expected_reviewed_total)
