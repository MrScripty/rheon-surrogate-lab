"""Account known original Run identities against the native project history.

Training cards and Run history are different populations. Read already retained
public metadata only; this report does not import history or infer Trial edges.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

from verify_native_portfolio_fork import inventory

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/model-portfolio-migration-v1'
PROJECT = ROOT / 'projects/rheon-model-portfolio-unified-03'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def report_coverage(output):
    if output.exists():
        raise ValueError('Fresh coverage report required')
    manifest_raw = (PROJECT / '.tuldok/project.json').read_bytes()
    descriptor, native = inventory(PROJECT)
    sources = {}

    def load(name, path):
        raw = path.read_bytes()
        sources[name] = dict(path=str(path.relative_to(ROOT)), sha256=sha(raw))
        return json.loads(raw)

    known = load('known_runs', BASE / 'discovery/run-reconciliation-02.json')
    work = load('fit_worklist', BASE / 'discovery/model-worklist-05.json')
    fits = defaultdict(set)
    for fit in work['fit_cards']:
        for observation in fit['recorded_Run_observations']:
            fits[observation['catalog_id'], observation['run_id']].add(fit['item_id'])
    captured = {}
    for (family, identity), (entry, path) in native.items():
        if family != 'run_history':
            continue
        doc = json.loads(path.read_bytes())
        key = doc['origin_catalog_id'], doc['run']['id']
        if key in captured:
            raise ValueError('Repeated original native Run identity')
        captured[key] = dict(canonical_id=identity, canonical_sha256=entry['sha256'],
                             state=doc['run']['state'], mode=doc['run']['mode'])
    rows = []
    seen = set()
    for original in known['records']:
        run = original['run']
        key = original['namespace'], run['id']
        if key in seen:
            raise ValueError('Repeated known Run identity')
        seen.add(key)
        current = captured.get(key)
        if current and (current['state'] != run['state'] or current['mode'] != run['mode']):
            raise ValueError('Captured native Run state or mode differs from original public metadata')
        rows.append(dict(origin_catalog_id=key[0], run_id=key[1], original_epoch=original['epoch'],
                         observed_in=original['observed_in'], state=run['state'], mode=run['mode'],
                         original_graph_id=run['graph_id'], original_graph_sha256=run['graph_sha256'],
                         fit_item_ids=sorted(fits[key]), native_history=current,
                         disposition='native_history_present' if current else 'native_history_not_yet_imported'))
    if set(captured) - seen or len(rows) != 460:
        raise ValueError('Native or known Run population differs from reconciled coverage')
    if (PROJECT / '.tuldok/project.json').read_bytes() != manifest_raw:
        raise ValueError('Project changed during coverage read')
    result = dict(schema='rheon.native-portfolio-history-coverage.v1',
                  status='accounted_current_native_membership_not_full_history_qualification',
                  created_unix=time.time(), checkpoint=descriptor['checkpoint'], sources=sources,
                  known_original_Run_identities=len(rows), native_Run_histories=len(captured),
                  remaining_native_Run_histories=len(rows) - len(captured),
                  remaining_by_epoch=dict(Counter(row['original_epoch'] for row in rows if row['native_history'] is None)),
                  remaining_by_state=dict(Counter(row['state'] for row in rows if row['native_history'] is None)),
                  fits_with_recorded_Run_links=sum(bool(fit['recorded_Run_observations']) for fit in work['fit_cards']),
                  fit_card_count=len(work['fit_cards']), rows=rows, new_neural_forwards=0,
                  limitation='Membership and original public state/mode only. Full history contents, Trial ancestry, payload custody and portability require native extraction and independent original-byte checks.',
                  runner_sha256=sha(Path(__file__).read_bytes()))
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'known_original_Run_identities',
        'native_Run_histories', 'remaining_native_Run_histories', 'remaining_by_epoch', 'remaining_by_state')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report_coverage(args.output.resolve())
