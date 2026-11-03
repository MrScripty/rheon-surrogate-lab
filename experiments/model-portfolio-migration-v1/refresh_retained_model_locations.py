"""Refresh model locators after verified inactive-catalog compaction.

Historical inventories remain unchanged. Exact bytes, not filename similarity,
select a surviving original alias or the pinned compaction keeper. No payload
is copied, linked, deserialized, executed or regenerated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import time

from compact_retired_catalog import safe
from run_pressure_portfolio_v2 import verify_material

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
OLD = ROOT/'runs/dependency-migration-v1/catalog-current-cold-04'
PLAN = RUN/'discovery/inactive-cold-catalog-compaction-01.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def exact(path, expected, size=None):
    if not stat.S_ISREG(path.lstat().st_mode) or sha(path) != expected:
        raise ValueError('Surviving model evidence differs from its recorded SHA256')
    if size is not None and path.stat().st_size != size:
        raise ValueError('Surviving checkpoint byte count differs')
    return path


def resolve(original, expected, *, old, keeper, files):
    if original.exists():
        return exact(original,expected)
    if not original.is_relative_to(old):
        raise ValueError('Missing model evidence has no explicit compaction mapping')
    row = files.get(str(original.relative_to(old)))
    if row is None or row['keeper_path'] is None or row['sha256'] != expected:
        raise ValueError('Compaction mapping differs from original evidence identity')
    retained = safe(keeper/'catalog',row['keeper_path'])
    exact(retained,expected,row['bytes'])
    if stat.S_IMODE(retained.stat().st_mode) != row['mode']:
        raise ValueError('Compaction keeper file mode differs')
    return retained


def run(output, report):
    if output.exists() or report.exists():
        raise ValueError('Fresh current-locator output and report required')
    marker = json.loads((OLD/'COMPACTED-SOURCE.json').read_bytes())
    plan = json.loads(PLAN.read_bytes())
    if (marker['status'] != 'compacted_requires_materialization'
            or sha(PLAN) != marker['manifest_sha256'] or plan['source'] != str(OLD.relative_to(ROOT))):
        raise ValueError('Completed exact inactive-catalog compaction required')
    keeper = safe(ROOT,plan['keeper'])
    if sha(keeper/'backup-receipt.json') != plan['keeper_receipt_sha256']:
        raise ValueError('Complete keeper receipt changed')
    mapped = {row['path']:row for row in plan['files']}
    original_inputs = [RUN/'discovery/checkpoint-identities-01.json',
                       RUN/'discovery/checkpoint-contexts-01.json',
                       RUN/'discovery/model-worklist-05.json']
    identities, contexts, work = [json.loads(p.read_bytes()) for p in original_inputs]
    groups = {row['sha256']:row for row in identities['groups']}
    if len(groups) != 524 or len(contexts['rows']) != 524 or len(work['fit_cards']) != 205:
        raise ValueError('Original checkpoint/fit population differs')
    checkpoints, evidence_remaps, receipts = [], {}, []
    for context in contexts['rows']:
        expected = context['sha256']
        group = groups[expected]
        available = next((safe(ROOT,p) for p in group['paths'] if safe(ROOT,p).is_file()),None)
        if available is None:
            available = resolve(safe(ROOT,context['retained_path']),expected,old=OLD,keeper=keeper,files=mapped)
        exact(available,expected,group['bytes'])
        checkpoints.append(dict(sha256=expected,bytes=group['bytes'],retained_path=str(available.relative_to(ROOT)),
            previous_retained_path=context['retained_path'],role=context['role']))
        for item in context.get('evidence',[]):
            original = safe(ROOT,item['path'])
            retained = resolve(original,item['sha256'],old=OLD,keeper=keeper,files=mapped)
            if retained != original:
                evidence_remaps[item['path']] = dict(original_path=item['path'],retained_path=str(retained.relative_to(ROOT)),sha256=item['sha256'])
    for fit in work['fit_cards']:
        expected = fit['receipt_sha256']
        available = next((safe(ROOT,p) for p in fit['receipt_paths'] if safe(ROOT,p).is_file()),None)
        if available is None:
            original = next((safe(ROOT,p) for p in fit['receipt_paths'] if safe(ROOT,p).is_relative_to(OLD)),None)
            if original is None:
                raise ValueError('No surviving receipt alias or explicit compaction mapping')
            available = resolve(original,expected,old=OLD,keeper=keeper,files=mapped)
        exact(available,expected)
        receipts.append(dict(fit_item_id=fit['item_id'],sha256=expected,retained_path=str(available.relative_to(ROOT))))
    material = json.loads((RUN/'discovery/pressure-material-01.json').read_bytes())
    verify_material(material)
    value = dict(schema='rheon.retained-model-locations.v1',created_unix=time.time(),
        checkpoints=checkpoints,fit_receipts=receipts,evidence_remaps=list(evidence_remaps.values()),
        inputs=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in original_inputs],
        compaction_manifest=dict(path=str(PLAN.relative_to(ROOT)),sha256=sha(PLAN)),
        original_inventories_changed=False,payload_files_copied=0,models_executed=0,
        scope='Current verified locations of the524known checkpoint byte states and205fit receipts; not complete model identity, reproduction or benchmarking.')
    with output.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n')
    summary = dict(schema=value['schema'],created_unix=time.time(),status='all_checkpoint_states_and_fit_receipts_available_after_compaction',
        result=dict(path=str(output.relative_to(ROOT)),sha256=sha(output)),checkpoint_byte_states=len(checkpoints),
        changed_selected_checkpoint_locations=sum(row['retained_path'] != row['previous_retained_path'] for row in checkpoints),
        exact_fit_receipts=len(receipts),explicit_evidence_remaps=len(evidence_remaps),
        named_pressure_material_source_bindings_exact=len(material['models']),payload_files_copied=0,models_executed=0,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'experiments/model-portfolio-migration-v1/compact_retired_catalog.py']})
    with report.open('x') as stream:
        json.dump(summary,stream,indent=2);stream.write('\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args()
    run(args.output.resolve(),args.report.resolve())
