"""Prepare six guarded note revisions from four receipt-bound text documents.

The four fit cards, their index and the overview change together. This preparation
does not publish native records or deserialize model payloads. The original card
preparation remains immutable, and every original receipt remains byte exact.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/model-portfolio-migration-v1'
PROJECT = ROOT / 'projects/rheon-model-portfolio-unified-03'
ORIGINAL = BASE / 'native-portfolio-cards-04'
BINDINGS = ROOT / 'docs/plans/model-portfolio-migration-v1/reports/model-training-document-bindings-01.json'
FIT_IDS = frozenset(('fit-10d3a512b41ab2a6', 'fit-d18b434d52b73791',
                     'fit-0b2b6b6a8c42a47a', 'fit-4ca6c279be34e629'))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def prepare(output, expected_checkpoint):
    if output.exists() or not output.is_relative_to(BASE):
        raise ValueError('Fresh task-owned preparation directory required')
    manifest_raw = (PROJECT / '.tuldok/project.json').read_bytes()
    manifest = json.loads(manifest_raw)
    if (manifest['checkpoint'] != expected_checkpoint
            or manifest['project_id'] != '26122f0d30304598b64dcb814dc411ea'):
        raise ValueError('Explicit starting project/checkpoint differs')
    prepared = json.loads((ORIGINAL / 'preparation.json').read_bytes())
    notes = {note['path']: note for note in prepared['notes']}
    captured = {}
    for entry in manifest['inventory']:
        if entry['family'] != 'project_file':
            continue
        raw = (PROJECT / '.tuldok' / entry['path']).read_bytes()
        if digest(raw) != entry['sha256'] or len(raw) != entry['bytes']:
            raise ValueError('Canonical file record differs from its inventory')
        value = json.loads(raw)
        if value['project_id'] == manifest['project_id']:
            if value['path'] in captured:
                raise ValueError('Repeated captured path')
            captured[value['path']] = value
    if len(notes) != 212 or set(captured) != set(notes):
        raise ValueError('Expected all212 original notes')
    for path, note in notes.items():
        raw = (ORIGINAL / path).read_bytes()
        current = captured[path]['current']
        if (digest(raw) != note['sha256'] or len(raw) != note['bytes']
                or (PROJECT / path).read_bytes() != raw or current['text'].encode() != raw
                or current['revision'] != 1 or current['sha256'] != note['sha256']
                or captured[path]['history'] != [current]):
            raise ValueError('Original or user-edited note changed: ' + path)
    binding_raw = BINDINGS.read_bytes()
    rows = [row for row in json.loads(binding_raw)['rows'] if row['fit_item_id'] in FIT_IDS]
    if len(rows) != 4 or {row['fit_item_id'] for row in rows} != FIT_IDS:
        raise ValueError('Expected exactly four distinct new document bindings')
    updates = []
    output.mkdir()

    def write(path, raw, **extra):
        raw.decode('utf-8')
        if len(raw) > 1048576:
            raise ValueError('Native note exceeds supported limit')
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        entry = dict(path=path, kind='note', expected_revision=1, revision=2,
                     previous_sha256=notes[path]['sha256'], sha256=digest(raw), bytes=len(raw), **extra)
        updates.append(entry)
        return entry

    for row in rows:
        note = next(note for note in notes.values() if note.get('fit_item_id') == row['fit_item_id'])
        raw = (ORIGINAL / note['path']).read_bytes()
        receipt = raw[note['embedded_receipt_offset']:note['embedded_receipt_offset'] + note['embedded_receipt_bytes']]
        primary = (ROOT / row['primary_receipt_path']).read_bytes()
        document = (ROOT / row['document_path']).read_bytes()
        if (receipt != primary or digest(primary) != row['primary_receipt_sha256']
                or digest(document) != row['expected_sha256'] or len(document) != row['document_bytes']
                or row['binding_pointer'] != '/closure/' + row['binding_field']
                or json.loads(receipt)['closure'][row['binding_field']] != digest(document)):
            raise ValueError('Document does not match its exact original receipt')
        start = raw.index(b'## Retained facts and comparison evidence\n\n```json\n')
        start += len(b'## Retained facts and comparison evidence\n\n```json\n')
        end = raw.index(b'```\n\n## Exact original receipt bytes', start)
        facts = json.loads(raw[start:end])
        if facts.pop('bound_training_configuration') is not None:
            raise ValueError('A previously embedded configuration must not be replaced')
        facts['bound_training_document'] = row
        value = json.loads(document)
        fields = value['training'] if row['document_kind'] == 'experiment' else {
            key: value[key] for key in ('identifier', 'widths', 'optimizer', 'maximum_iterations')}
        summary = ('Receipt-bound ' + row['document_kind'] + ' recovered: '
                   + json.dumps(fields, ensure_ascii=False, separators=(',', ':'))
                   + '. These are recorded training settings; architecture source review and clean reproduction remain pending.\n\n').encode()
        heading_end = raw.index(b'\n\n', raw.index(b'Evidence draft')) + 2
        prefix = raw[:heading_end] + summary + raw[heading_end:start] + encode(facts) + raw[end:note['embedded_receipt_offset']]
        payload = prefix + receipt + raw[note['embedded_receipt_offset'] + len(receipt):]
        payload += ('\n## Exact receipt-bound training ' + row['document_kind'] + '\n\n'
                    'Binding in the original receipt: `' + row['binding_pointer'] + '`.\n\n```json\n').encode()
        document_offset = len(payload)
        payload += document + b'\n```\n'
        write(note['path'], payload, fit_item_id=row['fit_item_id'],
              embedded_receipt_offset=len(prefix), embedded_receipt_bytes=len(receipt),
              embedded_receipt_sha256=digest(receipt), embedded_document_offset=document_offset,
              embedded_document_bytes=len(document), embedded_document_sha256=digest(document), binding=row)

    index_path = 'notes/portfolio-index.json'
    index = json.loads((ORIGINAL / index_path).read_bytes())
    changes = {note['fit_item_id']: note for note in updates}
    for fit in index['fits']:
        if fit['item_id'] in changes:
            note = changes[fit['item_id']]
            fit['card_sha256'] = note['sha256']
            fit['verified_configuration_embedded'] = note['binding']['document_kind'] == 'configuration'
            fit['verified_training_document'] = note['binding']
    index['sources']['training_document_bindings'] = dict(path=str(BINDINGS.relative_to(ROOT)), sha256=digest(binding_raw))
    write(index_path, encode(index))
    overview_path = 'notes/portfolio-overview.md'
    overview = (ORIGINAL / overview_path).read_bytes()
    old = b'51 exact training configurations are now recovered and embedded alongside their original receipts. '
    new = (b'53 exact training configurations and two experiment definitions are recovered and embedded alongside their original receipts. '
           b'Each document is verified against the corresponding original receipt hash; experiment and configuration bindings remain distinct. ')
    if overview.count(old) != 1:
        raise ValueError('Original overview wording changed')
    write(overview_path, overview.replace(old, new))
    if (PROJECT / '.tuldok/project.json').read_bytes() != manifest_raw:
        raise ValueError('Project changed during preparation')
    (output / 'before-project.json').write_bytes(manifest_raw)
    result = dict(schema='rheon.training-document-note-revisions.v1', status='prepared_not_published',
                  starting_checkpoint=expected_checkpoint, project_id=manifest['project_id'],
                  previous_preparation=str((ORIGINAL / 'preparation.json').relative_to(ROOT)),
                  bindings_path=str(BINDINGS.relative_to(ROOT)), bindings_sha256=digest(binding_raw),
                  notes=updates, original_receipts_preserved=205, resulting_configurations=53,
                  resulting_experiments=2, new_neural_forwards=0, clean_reproduction_qualified=False)
    (output / 'preparation.json').write_bytes(encode(result))
    (output / 'adopt-body.json').write_bytes(encode(dict(items=[
        {key: note[key] for key in ('kind', 'path', 'expected_revision')} | dict(expected_sha256=note['sha256'])
        for note in updates])))
    print(json.dumps(dict(status=result['status'], changed_notes=len(updates),
                          configurations=53, experiments=2, total_adoption_bytes=sum(note['bytes'] for note in updates))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-checkpoint', required=True)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.expected_checkpoint)
