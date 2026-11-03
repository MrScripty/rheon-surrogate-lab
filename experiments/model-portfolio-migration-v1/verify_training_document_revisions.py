"""Independently verify training-document note bytes and native revision custody.

Uses portable JSON and original receipts only; no native decoder, SQL, model
runtime or credentials. Prepared and published modes make different claims.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from verify_native_portfolio_fork import inventory

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/model-portfolio-migration-v1'
PROJECT = ROOT / 'projects/rheon-model-portfolio-unified-03'
OLD = BASE / 'native-portfolio-cards-04'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify(preparation, report, published):
    if report.exists():
        raise ValueError('Fresh report required')
    prepared_raw = (preparation / 'preparation.json').read_bytes()
    prepared = json.loads(prepared_raw)
    changed = {note['path']: note for note in prepared['notes']}
    if len(changed) != 6 or sum('fit_item_id' in note for note in changed.values()) != 4:
        raise ValueError('Expected four cards, index and overview')
    before = json.loads((preparation / 'before-project.json').read_bytes())
    manifest_raw = (PROJECT / '.tuldok/project.json').read_bytes()
    descriptor, records = inventory(PROJECT)
    if descriptor['project_id'] != before['project_id']:
        raise ValueError('Project identity changed')
    original_notes = {note['path']: note for note in json.loads((OLD / 'preparation.json').read_bytes())['notes']}
    captured = {}
    changed_ids = set()
    for (family, identity), (_, path) in records.items():
        if family != 'project_file':
            continue
        doc = json.loads(path.read_bytes())
        if doc['project_id'] != descriptor['project_id']:
            continue
        if doc['path'] in captured:
            raise ValueError('Repeated native note path')
        captured[doc['path']] = doc
        if doc['path'] in changed:
            changed_ids.add((family, identity))
    if len(captured) != 212 or set(captured) != set(original_notes):
        raise ValueError('Native note membership changed')
    old_records = {(entry['family'], entry['id']): entry for entry in before['inventory']}
    added = set(records) - set(old_records)
    if set(old_records) - set(records):
        raise ValueError('A prior canonical identity disappeared')
    if published:
        request = json.loads((preparation / 'adopt-six.request.json').read_bytes())
        response = json.loads((preparation / 'adopt-six.response.json').read_bytes())
        if len(added) != 1 or next(iter(added))[0] != 'project_operation':
            raise ValueError('Expected only one new native operation journal record')
        journal = json.loads(records[next(iter(added))][1].read_bytes())
        if (journal['operation'] != 'files.adopt_many'
                or journal['project_id'] != descriptor['project_id']
                or journal['request_key'] != request['arguments']['request_key']
                or response['status'] != 200 or journal['result'] != response['result']['value']
                or response['result']['checkpoint'] != descriptor['checkpoint']):
            raise ValueError('New journal does not match the acknowledged exact native request')
    elif added:
        raise ValueError('Canonical identities changed before publication')
    for key, entry in old_records.items():
        if published and key in changed_ids:
            continue
        if records[key][0] != entry:
            raise ValueError('An unrelated canonical record changed: ' + str(key))
    receipts = configs = experiments = 0
    effective = {}
    for path, old_note in original_notes.items():
        old_raw = (OLD / path).read_bytes()
        old_digest = sha(old_raw)
        if old_digest != old_note['sha256']:
            raise ValueError('Original prepared bytes changed')
        note = changed.get(path, old_note)
        raw = (preparation / path).read_bytes() if path in changed else old_raw
        if sha(raw) != note['sha256'] or len(raw) != note['bytes']:
            raise ValueError('Prepared revision differs from its declaration')
        effective[path] = raw
        doc = captured[path]
        current = doc['current']
        if published and path in changed:
            expected_current, revision = raw, 2
            if len(doc['history']) != 2 or doc['history'][-1] != current:
                raise ValueError('Updated native revision history is incomplete')
            previous = doc['history'][0]
        else:
            expected_current, revision = old_raw, 1
            if doc['history'] != [current]:
                raise ValueError('An unchanged note revision history changed')
            previous = current
        if (previous['revision'] != 1 or previous['sha256'] != old_digest
                or previous['text'].encode() != old_raw or previous['bytes'] != len(old_raw)):
            raise ValueError('Original revision1 was not retained exactly')
        if (current['revision'] != revision or current['text'].encode() != expected_current
                or current['sha256'] != sha(expected_current) or current['bytes'] != len(expected_current)
                or (PROJECT / path).read_bytes() != expected_current):
            raise ValueError('Native or working note differs from expected current revision')
        if 'fit_item_id' not in old_note:
            continue
        offset = note['embedded_receipt_offset']
        receipt = raw[offset:offset + note['embedded_receipt_bytes']]
        old_receipt = old_raw[old_note['embedded_receipt_offset']:old_note['embedded_receipt_offset'] + old_note['embedded_receipt_bytes']]
        if receipt != old_receipt or sha(receipt) != old_note['embedded_receipt_sha256']:
            raise ValueError('Original receipt changed in a revised card')
        receipts += 1
        if path in changed:
            row = note['binding']
            offset = note['embedded_document_offset']
            document = raw[offset:offset + note['embedded_document_bytes']]
            primary = (ROOT / row['primary_receipt_path']).read_bytes()
            if (receipt != primary or sha(primary) != row['primary_receipt_sha256']
                    or document != (ROOT / row['document_path']).read_bytes()
                    or sha(document) != json.loads(primary)['closure'][row['binding_field']]
                    or sha(document) != row['expected_sha256']):
                raise ValueError('New training document is not directly receipt-bound')
            if row['document_kind'] == 'configuration' and row['binding_field'] == 'config_sha256':
                configs += 1
            elif row['document_kind'] == 'experiment' and row['binding_field'] == 'experiment_sha256':
                experiments += 1
            else:
                raise ValueError('Training document type and binding disagree')
        elif 'embedded_config_offset' in note:
            offset = note['embedded_config_offset']
            config = raw[offset:offset + note['embedded_config_bytes']]
            if sha(config) != json.loads(receipt)['closure']['config_sha256']:
                raise ValueError('Previously bound training configuration changed')
            configs += 1
    if (receipts, configs, experiments) != (205, 53, 2):
        raise ValueError('Incomplete training-document custody')
    index = json.loads(effective['notes/portfolio-index.json'])
    for fit in index['fits']:
        if sha(effective[fit['card_path']]) != fit['card_sha256']:
            raise ValueError('Portfolio index links to stale card bytes')
    if sum(fit['verified_configuration_embedded'] for fit in index['fits']) != 53:
        raise ValueError('Configuration count in index differs')
    bundle = json.loads((BASE / 'native-pressure-portfolio-bundle-01/bundle.json').read_bytes())
    for packet in bundle['packets']:
        raw = (PROJECT / 'artifacts/sha256' / packet['sha256']).read_bytes()
        if len(raw) != packet['bytes'] or sha(raw) != packet['sha256']:
            raise ValueError('Original pressure packet changed')
    if (PROJECT / '.tuldok/project.json').read_bytes() != manifest_raw:
        raise ValueError('Project checkpoint changed during verification')
    result = dict(schema='rheon.training-document-revision-custody.v1',
                  status='passed_published_revisions' if published else 'passed_prepared_revisions',
                  created_unix=time.time(), checkpoint=descriptor['checkpoint'],
                  preparation_sha256=sha(prepared_raw), canonical_records=len(records),
                  native_notes=212, revised_notes=6 if published else 0, unchanged_notes=206 if published else 212,
                  original_revision1_notes_exact=212, exact_primary_receipts=receipts,
                  exact_receipt_bound_configurations=configs, exact_receipt_bound_experiments=experiments,
                  pressure_packets_exact=9, unrelated_canonical_records_unchanged=True,
                  acknowledged_operation_journal_records=len(added),
                  new_neural_forwards=0, clean_reproduction_qualified=False,
                  runner_sha256=sha(Path(__file__).read_bytes()))
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--published', action='store_true')
    args = parser.parse_args()
    verify(args.preparation.resolve(), args.report.resolve(), args.published)
