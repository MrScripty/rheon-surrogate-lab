"""Admit the prepared nine pressure partitions through the native owner SDK.

One dispatch per retained request; no retries or SQL/store bypass. A failed or
unknown outcome stops the sequence for explicit reconciliation. This records
native import and metadata consumption, not UI or full benchmark acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'runs/model-portfolio-migration-v1/native-pressure-portfolio-bundle-01'
PROJECT = '26122f0d30304598b64dcb814dc411ea'
CHECKPOINT = '28c50a2acb961290aabb57446098252d856309143c2177b1cf7b9e0942b3bc5e'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def run(controller, output, report):
    if output.exists() or report.exists():
        raise ValueError('Fresh outputs required; reconcile existing requests, never restart them')
    state = json.loads((controller / 'controller.json').read_text())
    source = Path(state['source']).resolve()
    if state['status'] != 'ready' or not source.is_relative_to(ROOT / 'runs/model-portfolio-migration-v1/sources'):
        raise ValueError('Ready task-owned native controller required')
    sys.path.insert(0, str(source))
    import agent_api
    bundle = json.loads((BUNDLE / 'bundle.json').read_bytes())
    manifest = json.loads((BUNDLE / 'source-project.json').read_bytes())
    references = {row['sha256']: row for row in manifest['data']}
    for entry in bundle['packets']:
        if hashlib.sha256((BUNDLE / entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Prepared packet changed before admission')
        if entry['sha256'] not in references:
            raise ValueError('Prepared source manifest does not bind every packet')
    output.mkdir(parents=True)
    result = dict(schema='rheon.native-pressure-portfolio-import.v1', status='running',
        created_unix=time.time(), controller=str(controller.relative_to(ROOT)),
        source_commit=state['source_commit'], runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bundle_sha256=hashlib.sha256((BUNDLE / 'bundle.json').read_bytes()).hexdigest(),
        project_id=PROJECT, requests=[], imports=[], metadata_checks=[],
        new_neural_forwards=0, reference_generation=False, UI_qualified=False,
        full_portfolio_migration_complete=False)
    save(report, result)
    client = agent_api.ProjectFilesOwnerClient(state['origin'],
        (controller / 'private/owner-session.txt').read_text().strip(), state['catalog_id'], timeout=300)
    try:
        client.bootstrap()
        observed = client.call('project_files.status', {})
        if (observed['status'] != 200 or observed['result']['state'] != 'open'
                or observed['result']['project']['project_id'] != PROJECT
                or observed['result']['project']['checkpoint'] != CHECKPOINT):
            raise ValueError('Expected saved history project/checkpoint required before admission')
        current = dict(binding=observed['result']['binding'], checkpoint=CHECKPOINT)

        def call(action, body=None):
            operation = 'project_files.' + action
            arguments = dict(body or {}, binding=current['binding'])
            mutation = agent_api.operation(operation)['side_effects']
            if mutation:
                arguments.update(expected_checkpoint=current['checkpoint'], request_key=uuid.uuid4().hex)
            else:
                arguments['checkpoint'] = current['checkpoint']
            stem = f'{len(result["requests"]) + 1:03d}-{action}'
            save(output / (stem + '.request.json'), dict(operation=operation, arguments=arguments,
                origin=state['origin'], controller_catalog_id=state['catalog_id']))
            receipt = dict(operation=operation, request=stem + '.request.json', outcome='dispatched_once')
            result['requests'].append(receipt)
            save(report, result)
            started = time.monotonic()
            response = client.call(operation, arguments)
            save(output / (stem + '.response.json'), response)
            receipt.update(status=response['status'], response=stem + '.response.json',
                           metadata_operation_wall_seconds=time.monotonic() - started)
            save(report, result)
            if response['status'] != 200:
                raise ValueError('Native operation rejected; inspect retained response before continuing')
            value = response['result']
            if value['binding'] != current['binding'] or value['project_id'] != PROJECT:
                raise ValueError('Native owner selection changed during import')
            current['checkpoint'] = value['checkpoint']
            result['last_acknowledged_checkpoint'] = current['checkpoint']
            save(report, result)
            print(json.dumps(dict(operation=action, status=200, completed=len(result['requests']))), flush=True)
            return value['value']

        registered = call('source_register', dict(root='lab', manifest_path=str((BUNDLE / 'source-project.json').relative_to(ROOT))))
        if registered['project_id'] != manifest['id']:
            raise ValueError('Registered source identity differs from the prepared manifest')
        snapshot = {key: registered[key] for key in ('origin_catalog_id', 'project_id', 'manifest_sha256')}
        result['source_snapshot'] = snapshot
        for entry in bundle['packets']:
            admitted = call('measurement_import', dict(snapshot=snapshot,
                reference_id=references[entry['sha256']]['id'], expected_sha256=entry['sha256']))
            if admitted['blob'] != dict(sha256=entry['sha256'], bytes=entry['bytes']):
                raise ValueError('Admitted packet identity differs from prepared source')
            result['imports'].append(dict(path=entry['path'], packet_sha256=entry['sha256'],
                origin_catalog_id=admitted['origin_catalog_id'], subjects=len(entry['subjects']),
                observations=entry['observations'], measured=entry['measured'], unavailable=entry['unavailable']))
            save(report, result)
        sources = call('measurement_sources', dict(offset=0, limit=40))
        if sources['total'] != len(bundle['packets']) or {row['id'] for row in sources['items']} != set(references):
            raise ValueError('Native packet membership differs from complete prepared portfolio')
        for entry, admitted in zip(bundle['packets'], result['imports'], strict=True):
            response = call('measurement_metadata', dict(origin_catalog_id=admitted['origin_catalog_id'],
                packet_sha256=entry['sha256']))['value']
            if (response['subject_count'] != len(entry['subjects']) or response['case_count'] != bundle['cases_per_subject']
                    or response['observation_count'] != entry['observations']):
                raise ValueError('Native metadata loses subject/case/observation membership')
            result['metadata_checks'].append(dict(packet_sha256=entry['sha256'], exact_counts=True))
            save(report, result)
        result.update(status='passed_native_complete_pressure_partition_import_and_metadata',
            subject_count=len(bundle['subjects']), measured_scalars=bundle['measured_scalar_values'],
            explicit_unavailable=bundle['explicit_unavailable_observations'], project_left_open=True,
            finished_unix=time.time())
    except BaseException as error:
        result.update(status='stopped_for_explicit_reconciliation', error_type=type(error).__name__,
            error=str(error), next_action='Reconcile the last retained request; do not rerun this importer')
        raise
    finally:
        save(report, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    run(args.controller.resolve(), args.output.resolve(), args.report.resolve())
