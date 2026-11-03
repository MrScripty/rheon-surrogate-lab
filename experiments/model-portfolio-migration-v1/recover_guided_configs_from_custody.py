"""Recover two receipt-bound guided configurations from retained Run custody.

Read/copy exact JSON and Python source bytes only. No model archives, tensor
payloads, database credentials, imports of captured code or inference are used.
The current cards04/import stays frozen; these are later curation supplements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT.parent.parent / 'creative-media/Tuldok/data-postgres/workspace-runs'
FITS = {'guided-shared': 'fit-10d3a512b41ab2a6', 'guided-pressure': 'fit-d18b434d52b73791'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def recover(output, report):
    if output.exists() or report.exists():
        raise ValueError('Fresh custody recovery outputs required')
    work = json.loads((ROOT / 'runs/model-portfolio-migration-v1/discovery/model-worklist-05.json').read_bytes())
    fits = {fit['item_id']: fit for fit in work['fit_cards']}
    output.mkdir(parents=True)
    blobs = output / 'exact-text-sources'
    blobs.mkdir()
    rows = []
    for family, identity in FITS.items():
        base = ROOT / 'runs/full-ai-data-diversity-v1' / (family + '-managed-pair-01')
        receipt_raw = (base / 'train/receipt.json').read_bytes()
        if sha(receipt_raw) != fits[identity]['receipt_sha256']:
            raise ValueError('Original fit receipt differs from the inventoried fit')
        receipt = json.loads(receipt_raw)
        admission_raw = (base / 'mutations/04-train-admit.json').read_bytes()
        admitted = json.loads(admission_raw)['result']
        recipe = admitted['material']['recipe']
        source = ORIGINAL / ('.external-' + admitted['id']) / 'source'
        captured = []
        wanted = dict(receipt['closure']['sources'])
        wanted['config.json'] = receipt['closure']['config_sha256']
        for name, expected in wanted.items():
            references = [item for item in recipe['sources']
                if Path(item['path']).name == name and item['sha256'] == expected]
            if len(references) != 1:
                raise ValueError('Retained admission does not bind the exact receipt source: ' + name)
            relative = Path(references[0]['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Captured source path escapes its Run source directory')
            path = source / relative
            raw = path.read_bytes()
            raw.decode('utf-8')
            if sha(raw) != expected:
                raise ValueError('Actual captured source differs from receipt/admission: ' + name)
            target = blobs / (expected + Path(name).suffix)
            if target.exists() and target.read_bytes() != raw:
                raise ValueError('Conflicting recovered source identity')
            target.write_bytes(raw)
            captured.append(dict(name=name, sha256=expected, bytes=len(raw), original_path=str(path),
                recovered_path=str(target.relative_to(ROOT))))
        config = next(item for item in captured if item['name'] == 'config.json')
        rows.append(dict(fit_item_id=identity, fit_name=fits[identity]['name'], original_Run_id=admitted['id'],
            receipt_sha256=sha(receipt_raw), retained_admission_sha256=sha(admission_raw),
            config_sha256=config['sha256'], config_recovered_path=config['recovered_path'],
            exact_sources=captured, state='exact_primary_receipt_and_original_admission_bound_text_recovered'))
    result = dict(schema='rheon.guided-configuration-custody-recovery.v1',
        status='passed_two_configurations_and_original_training_text_sources', created_unix=time.time(), rows=rows,
        configurations_recovered=len(rows), distinct_text_blobs=len(list(blobs.iterdir())),
        structured_method_gap_configurations_now_recovered=53, remaining_without_direct_config_binding=2,
        current_cards04_modified=False, current_native_notes_modified=False, clean_reproduction_qualified=False,
        model_payloads_opened=False, new_neural_forwards=0, source_files_executed=False,
        runner_sha256=sha(Path(__file__).read_bytes()))
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'configurations_recovered', 'distinct_text_blobs')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    recover(args.output.resolve(), args.report.resolve())
