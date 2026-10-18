"""Qualify one existing zero-update CPU graph; journal mutations without retry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def qualify(config_path, output):
    if output.exists():
        raise ValueError('Fresh qualification directory required; never replay mutations')
    config = json.loads(config_path.read_text())
    sys.path[:0] = [str(ROOT / config['client_source']), str(ROOT / config['owner_adapter_source'])]
    import agent_api
    import project_cli
    from qualify_typed_pressure import OwnerHTTP
    output.mkdir()
    began = time.perf_counter()
    owner = OwnerHTTP(agent_api, config['endpoint'], config['catalog_id'], ROOT / config['session_file'])
    graph = json.loads((ROOT / config['restore_graph_receipt']).read_text())['result']
    # Already observed and retained before mutation; do not re-preflight unnecessarily.
    preflight = json.loads((ROOT / config['restore_preflight_receipt']).read_text())
    material = preflight['rows'][0]['material_sha256']
    original = json.loads((ROOT / config['original_restore_receipt']).read_text())
    if original['graph']['id'] != graph['id'] or original['material_sha256'] != material:
        raise ValueError('Restore graph material differs from the recorded workload')
    stages = []

    def mutation(name, operation, arguments, raw=False):
        path = output / (name + '.json')
        receipt = dict(status='intent', operation=operation, arguments=arguments, automatic_retry=False)
        write(path, receipt)
        started = time.perf_counter()
        result = owner.raw('POST', operation, arguments) if raw else owner.call(operation, arguments)
        receipt.update(status='acknowledged', elapsed_s=time.perf_counter()-started, result=result)
        write(path, receipt)
        stages.append(dict(operation=operation, elapsed_s=receipt['elapsed_s']))
        print(name, round(receipt['elapsed_s'], 3), 's', flush=True)
        return result

    approval = mutation('01-approval', '/api/workspace/runs/approve-external', dict(
        graph_id=graph['id'], graph_revision=graph['revision'], expected_digest=material,
        consumer_actor_id=owner.identity['actor_id'], ttl_seconds=3600), raw=True)
    run = mutation('02-admission', 'runs.submit_external', dict(
        request_key=config['restore_request_key'], graph_id=graph['id'],
        graph_revision=graph['revision'], expected_digest=material, approval_id=approval['id']))
    deadline = time.monotonic() + 600
    while run['state'] not in ('completed', 'failed', 'cancelled', 'interrupted', 'unknown'):
        if time.monotonic() > deadline:
            raise RuntimeError('Inspect known Run; do not resubmit after wait expiration')
        time.sleep(2)
        run = owner.call('runs.get', dict(run_id=run['id']))
        write(output / 'state.json', run)
    write(output / 'terminal.json', run)
    if run['state'] != 'completed' or run['custody_phase'] != 'drained':
        raise RuntimeError('Restore did not complete with proven drained custody')
    completed_s = time.perf_counter() - began

    class Reader:
        def call(self, operation, arguments):
            return {'status': 200, 'result': owner.call(operation, arguments)}

    downloads = []
    target = output / 'artifacts'
    target.mkdir()
    baseline = {row['name']: row for row in original['artifacts']}
    compare = set(config['unchanged_artifacts'])
    for row in run['artifacts']:
        started = time.perf_counter()
        project_cli.download(Reader(), run['id'], row['name'], row['sha256'], target / row['name'])
        actual = hashlib.sha256((target / row['name']).read_bytes()).hexdigest()
        if actual != row['sha256']:
            raise ValueError('Published artifact whole hash differs')
        equal = actual == baseline[row['name']]['sha256']
        if row['name'] in compare and not equal:
            raise ValueError('Restore numerical content differs: ' + row['name'])
        downloads.append(dict(name=row['name'], bytes=row['bytes'], sha256=actual,
            elapsed_s=time.perf_counter()-started, whole_sha256_verified=True, matches_original=equal))
        print('verified', row['name'], round(downloads[-1]['elapsed_s'], 3), 's', flush=True)
    if compare - {row['name'] for row in downloads}:
        raise ValueError('Required comparison artifacts missing')
    summary = json.loads((target / 'summary.json').read_text())
    if summary['mode'] != 'restore' or summary['optimizer_updates'] != 0:
        raise ValueError('Expected zero-update restore')
    receipt = dict(status='passed', run_id=run['id'], material_sha256=material,
        completed_drained=True, optimizer_updates=0, training_started=False,
        completed_elapsed_s=completed_s, total_elapsed_s=time.perf_counter()-began,
        stages=stages, artifacts=downloads, unchanged_artifacts=sorted(compare),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), automatic_retry=False)
    write(output / 'qualification.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('status', 'run_id', 'completed_elapsed_s', 'total_elapsed_s')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    qualify(args.config.resolve(), args.output.resolve())
