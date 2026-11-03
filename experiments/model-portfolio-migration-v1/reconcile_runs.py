"""Account for current and historical Run records and retained publications.

Preserve former-catalog membership and failures. Matching publication metadata
is not a fresh inference result or transitive-custody authorization.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconcile(current_path, archive_path, artifacts_path, local_path, weights_path, output, summary_path):
    if output.exists() or summary_path.exists():
        raise ValueError('Fresh Run reconciliation outputs required')
    current, archive, artifacts, local, weights = (json.loads(p.read_text()) for p in
                                                 (current_path, archive_path, artifacts_path, local_path, weights_path))
    if not current['catalog_id'] == archive['catalog_id'] == artifacts['catalog_id']:
        raise ValueError('Catalog namespaces differ')
    publications = {r['run_id']: r for r in artifacts['manifests']}
    receipts = defaultdict(list)
    for r in local['receipts']:
        if r.get('sha256'):
            receipts[r['sha256']].append(r['path'])
    known = {r['sha256']: r for r in weights['groups']}
    records, seen, issues = [], {}, []
    for epoch, data in (('current_catalog', current), ('archived_schema8_catalog', archive)):
        for run in data['runs']:
            if run['id'] in seen:
                prior = seen[run['id']]
                if prior['run'] == run:
                    prior['observed_in'].append(epoch)
                    continue
                issues.append(dict(run_id=run['id'], kind='Run_identity_has_different_selected_metadata_between_snapshots'))
            publication = publications.get(run['id'])
            inventory = run.get('artifacts') or []
            if run['state'] == 'completed' and publication is None:
                issues.append(dict(run_id=run['id'], kind='completed_Run_missing_manifest'))
            if publication and inventory != publication['declared_artifacts']:
                issues.append(dict(run_id=run['id'], kind='stored_and_publication_artifact_inventories_differ'))
            outputs = []
            for item in inventory:
                if item.get('role') != 'checkpoint':
                    continue
                group = known.get(item['sha256'])
                outputs.append(dict(name=item['name'], sha256=item['sha256'], bytes=item['bytes'],
                                    local_content_verified=group is not None,
                                    local_paths=group['paths'] if group else []))
                if group and group['bytes'] != item['bytes']:
                    issues.append(dict(run_id=run['id'], kind='checkpoint_size_mismatch', name=item['name']))
            recipe_receipts = []
            for item in inventory:
                if item['name'] == 'receipt.json':
                    recipe_receipts.append(dict(sha256=item['sha256'], local_paths=receipts[item['sha256']]))
            record = dict(namespace=current['catalog_id'], epoch=epoch, observed_in=[epoch], run=run,
                                publication_path=publication['path'] if publication else None,
                                publication_manifest_sha256=publication['sha256'] if publication else None,
                                checkpoint_outputs=outputs, recipe_receipts=recipe_receipts,
                                disposition='accounted_saved_metadata_pending_model_identity_and_fresh_benchmark')
            records.append(record)
            seen.setdefault(run['id'], record)
    for run_id in publications.keys() - seen.keys():
        issues.append(dict(run_id=run_id, kind='publication_without_current_or_archived_Run_record'))
    value = dict(schema='rheon.run-publication-reconciliation.v1', created_unix=time.time(),
                 status='accounted_metadata' if not issues else 'requires_review', records=records, issues=issues,
                 inputs=[dict(path=str(p), sha256=sha(p)) for p in
                         (current_path, archive_path, artifacts_path, local_path, weights_path)],
                 source_sha256=sha(Path(__file__)), inference_calls=0,
                 limitations=['No complete scientific model count or fresh benchmark established.',
                              'Checkpoint outputs include initial and intermediate snapshots; a Run is not necessarily a distinct model.',
                              'Publication inventories are matched; other artifact payloads and transitive input custody require separate verification.'])
    with output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    summary = dict(schema=value['schema'], status=value['status'], output=str(output), output_sha256=sha(output),
                   current_Run_records=len(current['runs']), archived_Run_records=len(archive['runs']),
                   unique_Run_ids=len(seen), publication_manifests=len(publications), issues=issues,
                   matching_public_metadata_in_both_snapshots=sum(len(r['observed_in']) > 1 for r in records),
                   run_states=dict(Counter(r['run']['state'] for r in records)),
                   checkpoint_output_declarations=sum(len(r['checkpoint_outputs']) for r in records),
                   checkpoint_declarations_content_located=sum(o['local_content_verified'] for r in records for o in r['checkpoint_outputs']),
                   recipe_receipt_declarations=sum(len(r['recipe_receipts']) for r in records),
                   recipe_receipt_declarations_located=sum(bool(o['local_paths']) for r in records for o in r['recipe_receipts']),
                   inference_calls=0)
    with summary_path.open('x') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('current', 'archive', 'artifacts', 'local', 'weights', 'output', 'summary'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    reconcile(args.current, args.archive, args.artifacts, args.local, args.weights, args.output, args.summary)
