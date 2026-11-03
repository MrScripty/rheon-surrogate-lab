"""Compare the native captured notes with prepared and original receipt bytes.

Reads portable JSON/files only. Does not use the native file decoder, credentials
or model runtime. The canonical inventory is verified by the existing migration
preservation reader. This is custody evidence, not a UI or reproduction claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from verify_native_portfolio_fork import inventory

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/model-portfolio-migration-v1'


def verify(report):
    if report.exists():
        raise ValueError('Fresh verification report required')
    project = ROOT / 'projects/rheon-model-portfolio-unified-03'
    manifest_before = (project / '.tuldok/project.json').read_bytes()
    descriptor, records = inventory(project)
    prepared = json.loads((BASE / 'native-portfolio-cards-04/preparation.json').read_bytes())
    fits = {item['item_id']: item for item in json.loads((BASE / 'discovery/model-worklist-05.json').read_bytes())['fit_cards']}
    captured = {}
    for (family, identity), (_, path) in records.items():
        if family != 'project_file':
            continue
        value = json.loads(path.read_bytes())
        if value['project_id'] != descriptor['project_id']:
            continue
        if value['path'] in captured:
            raise ValueError('Repeated native note path')
        captured[value['path']] = (identity, value)
    expected = {note['path']: note for note in prepared['notes']}
    if len(expected) != 212 or set(captured) != set(expected):
        raise ValueError('Native note membership differs from all212 prepared notes')
    receipt_count = config_count = 0
    for path, note in expected.items():
        identity, document = captured[path]
        raw = (BASE / 'native-portfolio-cards-04' / path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        current = document['current']
        wanted_id = hashlib.sha256((descriptor['project_id'] + '\0note\0' + path).encode()).hexdigest()
        if (identity != wanted_id or document['kind'] != 'note' or current['revision'] != 1
                or document['history'] != [current] or current['text'].encode('utf-8') != raw
                or current['sha256'] != digest or digest != note['sha256']
                or current['bytes'] != len(raw) or len(raw) != note['bytes']
                or (project / path).read_bytes() != raw):
            raise ValueError('Native captured bytes/history differ: ' + path)
        if 'fit_item_id' in note:
            fit = fits[note['fit_item_id']]
            offset = note['embedded_receipt_offset']
            receipt = raw[offset:offset + note['embedded_receipt_bytes']]
            if (hashlib.sha256(receipt).hexdigest() != fit['receipt_sha256']
                    or not any((ROOT / source).read_bytes() == receipt for source in fit['receipt_paths'] if (ROOT / source).is_file())):
                raise ValueError('Native note differs from its original primary fit receipt')
            receipt_count += 1
            if 'embedded_config_offset' in note:
                offset = note['embedded_config_offset']
                config = raw[offset:offset + note['embedded_config_bytes']]
                if hashlib.sha256(config).hexdigest() != json.loads(receipt)['closure']['config_sha256']:
                    raise ValueError('Native embedded configuration differs from direct receipt binding')
                config_count += 1
    bundle = json.loads((BASE / 'native-pressure-portfolio-bundle-01/bundle.json').read_bytes())
    for packet in bundle['packets']:
        raw = (project / 'artifacts/sha256' / packet['sha256']).read_bytes()
        if len(raw) != packet['bytes'] or hashlib.sha256(raw).hexdigest() != packet['sha256']:
            raise ValueError('Previously admitted pressure packet changed')
    if (project / '.tuldok/project.json').read_bytes() != manifest_before:
        raise ValueError('Canonical checkpoint changed during independent verification')
    if (receipt_count, config_count) != (205, 51):
        raise ValueError('Receipt/configuration support is incomplete')
    result = dict(schema='rheon.native-portfolio-notes-custody.v1', status='passed_all212_captured_notes_exact',
        created_unix=time.time(), project_id=descriptor['project_id'], checkpoint=descriptor['checkpoint'],
        canonical_records=len(records), exact_native_notes=len(captured), exact_primary_receipts=receipt_count,
        exact_receipt_bound_configs=config_count, prior_pressure_packets_exact=len(bundle['packets']),
        canonical_manifest_unchanged=True, new_neural_forwards=0,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        full_UI_cold_portability_and_reproduction_qualified=False)
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    verify(args.report.resolve())
