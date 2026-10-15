"""Normal Tuldok HTTP execution; the catalog owns runtime, Runs and custody."""
import time
import uuid

from policy import LAB, MIB, canonical, decode, fresh, read, require, sha, write


def execute(client, project, output, observer=None):
    """client supplies authenticated loopback request/download and actor/catalog IDs.

    No credential discovery, private catalog, local optimizer or substitute Run.
    Each exact grant is created only after its concrete preflight succeeds.
    """
    output = fresh(output)
    manifest = decode(read(project/'project.json', MIB)); stages = decode(read(project/'stages.json', MIB))
    opened = client.request('POST', '/api/workspace/projects', {'root': 'lab', 'manifest_path': (project/'project.json').relative_to(LAB).as_posix()})
    require(opened['id'] == manifest['id'] and opened['manifest_sha256'] == sha(read(project/'project.json', MIB)), 'project_snapshot', 'same registered execution project')
    context = client.request('GET', '/api/workspace/context')
    require(context['dataset_id'] == client.catalog_id, 'catalog_identity', 'existing selected Tuldok catalog')
    recipes = {r['id']: r for r in manifest['recipes']}; references = {r['id']: r for r in manifest['data']}
    completed = {}; report = {'catalog_id': client.catalog_id, 'project_id': opened['id'], 'stages': []}
    try:
        for index, stage in enumerate(stages):
            directory = fresh(output/f'stage-{index:02d}'); inputs = {}
            for slot, binding in stage['inputs'].items():
                if isinstance(binding, str):
                    ref = references[binding]; inputs[slot] = {'reference_id': binding, 'expected_sha256': ref['sha256']}
                else:
                    prior = completed[binding['stage']]; item = next(a for a in prior['artifacts'] if a['name'] == binding['artifact'])
                    inputs[slot] = {'kind': 'run_artifact', 'run_id': prior['id'], 'artifact_name': item['name'], 'expected_sha256': item['sha256']}
            recipe = recipes[stage['recipe_id']]
            arguments = {'name': 'Variable-grid '+stage['name'], 'project_id': opened['id'], 'recipe_id': recipe['id'],
                         'expected_sha256': recipe['sha256'], 'inputs': inputs, 'parameters': {}}
            graph = client.request('POST', '/api/workspace/graphs/external', arguments)
            preflight = client.request('POST', f'/api/workspace/graphs/{graph["id"]}/preflight-external', {'revision': graph['revision']})
            write(directory/'preflight.json', preflight)
            require(preflight['valid'], 'preflight', str(preflight.get('errors')))
            approval = client.request('POST', '/api/workspace/runs/approve-external', {'graph_id': graph['id'], 'graph_revision': graph['revision'],
                       'expected_digest': preflight['material_sha256'], 'consumer_actor_id': client.actor_id})
            write(directory/'approval.json', approval)
            submission = {'request_key': 'variable-model-'+uuid.uuid4().hex, 'graph_id': graph['id'], 'graph_revision': graph['revision'],
                          'expected_digest': preflight['material_sha256'], 'approval_id': approval['id']}
            write(directory/'submission.json', submission)
            run = client.request('POST', '/api/workspace/runs/submit-external', submission)
            print('ADMITTED', stage['name'], run['id'], flush=True)
            deadline = time.monotonic()+650
            while run['state'] in ('queued', 'running', 'cancelling'):
                require(time.monotonic() < deadline, 'observation_deadline', 'Run remains Tuldok-owned; inspect before any new execution')
                progress = client.request('GET', f'/api/workspace/runs/{run["id"]}/progress')
                current = progress.get('progress') or {}
                if observer:
                    observer.publish('Variable-grid managed '+stage['name'], f'Run {run["id"]}: {run["state"]}; {current.get("stage", "preparing")}',
                                     current.get('completed', 0), current.get('total'))
                time.sleep(.5)
                run = client.request('GET', f'/api/workspace/runs/{run["id"]}')
            write(directory/'run.json', run)
            require(run['state'] == 'completed' and run['custody_phase'] == 'drained', 'managed_completion', run['id']+' '+run['state'])
            artifacts = fresh(directory/'artifacts')
            for item in run['artifacts']:
                raw = client.download(f'/api/workspace/runs/{run["id"]}/artifacts/{item["name"]}', item['bytes'])
                require(len(raw) == item['bytes'] and sha(raw) == item['sha256'], 'download_hash', item['name'])
                require('/' not in item['name'] and item['name'] not in ('.', '..'), 'artifact_name', 'flat exact output')
                (artifacts/item['name']).write_bytes(raw)
            completed[stage['name']] = run
            report['stages'].append({'name': stage['name'], 'run_id': run['id'], 'state': run['state'], 'custody_phase': run['custody_phase'],
                                     'material_sha256': preflight['material_sha256'], 'artifacts': run['artifacts']})
        for index, stage in enumerate(report['stages']):
            current = client.request('GET', f'/api/workspace/runs/{stage["run_id"]}')
            require(current['state'] == 'completed' and current['custody_phase'] == 'drained', 'reopened_read', 'retained normal Run')
            destination = fresh(output/f'redownloaded-stage-{index:02d}')
            for item in current['artifacts']:
                raw = client.download(f'/api/workspace/runs/{current["id"]}/artifacts/{item["name"]}', item['bytes'])
                require(sha(raw) == item['sha256'] and raw == read(output/f'stage-{index:02d}'/'artifacts'/item['name'], 4*MIB), 'repeat_download', 'whole artifact parity')
                (destination/item['name']).write_bytes(raw)
        report['status'] = 'completed_and_drained'; write(output/'receipt.json', report)
        return report
    except BaseException as error:
        report.update(status='failed', error=str(error)); write(output/'failure.json', report)
        if observer:
            observer.publish('Variable-grid managed execution failed', 'Retained Run/evidence must be inspected; no automatic retry', state='failed')
        raise
