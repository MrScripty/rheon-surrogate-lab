"""One explicit native project owner operation with retained request/response.

Uses the upstream negotiated SDK. Requests are retained before dispatch and are
never automatically retried. Credentials stay in the new private controller.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/model-portfolio-migration-v1/sources/Tuldok'


def run(controller, operation, body, output, replay=None, timeout=60):
    if any(output.with_suffix(suffix).exists() for suffix in ('.request.json', '.response.json', '.error.json')):
        raise ValueError('Existing owner request must be reconciled explicitly; no automatic replay')
    state = json.loads((controller / 'controller.json').read_text())
    if state['status'] != 'ready':
        raise ValueError('Observe the owned controller live before dispatch')
    source = Path(state['source']).resolve()
    if (not source.is_relative_to(ROOT / 'runs/model-portfolio-migration-v1/sources')
            or not (source / 'agent_api.py').is_file()):
        raise ValueError('The owned controller must use a task-owned native source')
    sys.path.insert(0, str(source))
    import agent_api
    client = agent_api.ProjectFilesOwnerClient(state['origin'],
        (controller / 'private/owner-session.txt').read_text().strip(), state['catalog_id'], timeout=timeout)
    client.bootstrap()
    current_envelope = client.call('project_files.status', {})
    if current_envelope['status'] != 200:
        raise ValueError('Native owner status unavailable; no mutation dispatched')
    current = current_envelope['result']
    arguments = json.loads(body.read_text())
    contract = agent_api.operation(operation)
    if replay is not None:
        prior = json.loads(replay.read_text())
        if (prior['operation'] != operation or prior['origin'] != state['origin']
                or prior['controller_catalog_id'] != state['catalog_id']
                or prior['authoring_body_sha256'] != hashlib.sha256(body.read_bytes()).hexdigest()):
            raise ValueError('Explicit replay must retain the exact original operation, authority and authoring body')
        arguments = prior['arguments']
    elif operation != 'project_files.status':
        arguments['binding'] = current['binding']
        if contract['side_effects']:
            arguments['expected_checkpoint'] = current['project']['checkpoint'] if current['project'] else None
            arguments['request_key'] = uuid.uuid4().hex
        else:
            if not current['project']:
                raise ValueError('This read requires an open native project')
            arguments['checkpoint'] = current['project']['checkpoint']
    output.parent.mkdir(parents=True, exist_ok=True)
    request = dict(operation=operation, arguments=arguments,
                   controller_catalog_id=state['catalog_id'], origin=state['origin'],
                   timeout_seconds=timeout,
                   authoring_body_sha256=hashlib.sha256(body.read_bytes()).hexdigest())
    output.with_suffix('.request.json').write_text(json.dumps(request, indent=2) + '\n')
    try:
        value = client.call(operation, arguments)
    except BaseException as error:
        output.with_suffix('.error.json').write_text(json.dumps(dict(error_type=type(error).__name__,
            error=str(error), code=getattr(error, 'code', None), status=getattr(error, 'status', None),
            request_outcome='inspect_original_request_and_owner_before_any_retry'), indent=2) + '\n')
        raise
    output.with_suffix('.response.json').write_text(json.dumps(value, indent=2) + '\n')
    if value['status'] >= 400:
        raise RuntimeError('Native owner returned a retained error response; inspect before retry')
    print(json.dumps(dict(operation=operation, response=str(output.with_suffix('.response.json').relative_to(ROOT)),
                          result_keys=list(value['result']), checkpoint=value['result'].get('checkpoint'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', type=Path, required=True)
    parser.add_argument('--operation', required=True)
    parser.add_argument('--body', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-request', type=Path)
    parser.add_argument('--timeout-seconds', type=int, choices=(60, 300), default=60)
    args = parser.parse_args()
    run(args.controller.resolve(), args.operation, args.body.resolve(), args.output.resolve(),
        args.replay_request.resolve() if args.replay_request else None, args.timeout_seconds)
