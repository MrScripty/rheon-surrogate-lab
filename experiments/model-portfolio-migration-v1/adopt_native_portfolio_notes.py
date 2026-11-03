"""Adopt remaining prepared notes through bounded native SDK transactions.

Retain every request before dispatch. Never retry or resume an unknown result.
This performs metadata publication only, and leaves the user project open.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
CARDS = ROOT / 'runs/model-portfolio-migration-v1/native-portfolio-cards-04'
PROJECT = '26122f0d30304598b64dcb814dc411ea'
WORKING = ROOT / 'projects/rheon-model-portfolio-unified-03'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def run(controller, output, report, checkpoint, existing_notes, quiet_seconds):
    if output.exists() or report.exists():
        raise ValueError('Fresh outputs required; reconcile retained requests before any further action')
    state = json.loads((controller / 'controller.json').read_bytes())
    source = Path(state['source']).resolve()
    if state['status'] != 'ready' or not source.is_relative_to(ROOT / 'runs/model-portfolio-migration-v1/sources'):
        raise ValueError('Ready task-owned native controller required')
    prepared = json.loads((CARDS / 'preparation.json').read_bytes())
    notes = prepared['notes']
    if len(notes) != 212 or len({note['path'] for note in notes}) != 212:
        raise ValueError('Expected all 212 distinct prepared notes')
    for note in notes:
        relative = Path(note['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'notes':
            raise ValueError('Prepared note path is outside the native notes directory')
        for root in (CARDS, WORKING):
            raw = (root / relative).read_bytes()
            if len(raw) != note['bytes'] or hashlib.sha256(raw).hexdigest() != note['sha256']:
                raise ValueError('Prepared or working note changed before native adoption')
    manifest_raw = (WORKING / '.tuldok/project.json').read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest['checkpoint'] != checkpoint or manifest['project_id'] != PROJECT:
        raise ValueError('Canonical preflight differs from the explicit expected checkpoint')
    captured = {}
    for entry in manifest['inventory']:
        if entry['family'] != 'project_file':
            continue
        raw = (WORKING / '.tuldok' / entry['path']).read_bytes()
        if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Existing canonical note identity changed')
        document = json.loads(raw)
        if document['project_id'] != PROJECT:
            continue
        if document['path'] in captured:
            raise ValueError('Existing captured note path repeats')
        captured[document['path']] = document
    expected = {note['path']: note for note in notes}
    if len(captured) != existing_notes or not set(captured) <= set(expected):
        raise ValueError('Existing native note membership differs from the explicit starting state')
    for path, document in captured.items():
        current = document['current']
        if (current['revision'] != 1 or current['sha256'] != expected[path]['sha256']
                or current['text'].encode() != (CARDS / path).read_bytes()):
            raise ValueError('An existing captured note changed; review it before continuing')
    if (WORKING / '.tuldok/project.json').read_bytes() != manifest_raw:
        raise ValueError('Canonical checkpoint changed during preflight')
    remaining = [note for note in notes if note['path'] not in captured]
    if not remaining or 'notes/portfolio-overview.md' not in captured:
        raise ValueError('Expected a partially adopted portfolio including the original overview')
    groups = []
    for note in remaining:
        if not groups or len(groups[-1]) == 40 or sum(item['bytes'] for item in groups[-1]) + note['bytes'] > 8*1024*1024:
            groups.append([])
        groups[-1].append(note)
    output.mkdir(parents=True)
    result = dict(schema='rheon.native-portfolio-note-adoption.v1', status='running',
        created_unix=time.time(), project_id=PROJECT, source_commit=state['source_commit'],
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preparation_sha256=hashlib.sha256((CARDS / 'preparation.json').read_bytes()).hexdigest(),
        controller=str(controller.relative_to(ROOT)), requests=[], adopted_notes=0,
        starting_checkpoint=checkpoint, starting_native_notes=existing_notes,
        planned_remaining_notes=len(remaining), quiet_seconds_between_batches=quiet_seconds,
        new_neural_forwards=0, full_portfolio_migration_complete=False)
    save(report, result)
    sys.path.insert(0, str(source))
    import agent_api
    client = agent_api.ProjectFilesOwnerClient(state['origin'],
        (controller / 'private/owner-session.txt').read_text().strip(), state['catalog_id'], timeout=300)
    try:
        client.bootstrap()
        observed = client.call('project_files.status', {})
        if (observed['status'] != 200 or observed['result']['state'] != 'open'
                or observed['result']['project']['project_id'] != PROJECT
                or observed['result']['project']['checkpoint'] != checkpoint):
            raise ValueError('Expected saved project/checkpoint required before adopting notes')
        binding = observed['result']['binding']
        result['phase'] = 'interactive_window_before_first_batch'
        save(report, result)
        deadline = time.monotonic() + quiet_seconds
        while time.monotonic() < deadline and not (output / 'stop-after-batch').exists():
            time.sleep(min(1, max(0, deadline - time.monotonic())))
        for index, group in enumerate(groups, 1):
            if (output / 'stop-after-batch').exists():
                result.update(status='stopped_cooperatively_between_acknowledged_batches', project_left_open=True)
                return
            arguments = dict(binding=binding, expected_checkpoint=checkpoint, request_key=uuid.uuid4().hex,
                items=[dict(kind='note', path=note['path'], expected_revision=0,
                    expected_sha256=note['sha256']) for note in group])
            stem = f'{index:03d}-files-adopt'
            save(output / (stem + '.request.json'), dict(operation='project_files.files_adopt',
                arguments=arguments, origin=state['origin'], controller_catalog_id=state['catalog_id']))
            receipt = dict(request=stem + '.request.json', outcome='dispatched_once', notes=len(group))
            result['phase'] = 'native_batch_in_flight'
            result['requests'].append(receipt)
            save(report, result)
            started = time.monotonic()
            response = client.call('project_files.files_adopt', arguments)
            save(output / (stem + '.response.json'), response)
            receipt.update(status=response['status'], response=stem + '.response.json',
                metadata_operation_wall_seconds=time.monotonic() - started)
            save(report, result)
            if response['status'] != 200:
                raise ValueError('Native adoption rejected; inspect retained response')
            envelope = response['result']
            value = envelope['value']
            if (envelope['binding'] != binding or envelope['project_id'] != PROJECT
                    or value['changed_count'] != len(group) or len(value['items']) != len(group)):
                raise ValueError('Native adoption selection or membership changed')
            for actual, expected in zip(value['items'], group, strict=True):
                if any(actual[field] != expected[field] for field in ('path', 'sha256', 'bytes')) or actual['revision'] != 1:
                    raise ValueError('Native captured note differs from the prepared exact bytes')
            checkpoint = envelope['checkpoint']
            result['last_acknowledged_checkpoint'] = checkpoint
            result['adopted_notes'] += len(group)
            receipt['outcome'] = 'acknowledged_exact_group'
            save(report, result)
            print(json.dumps(dict(batch=index, status=200, adopted=result['adopted_notes'])), flush=True)
            if index < len(groups):
                result['phase'] = 'interactive_window_between_batches'
                save(report, result)
                deadline = time.monotonic() + quiet_seconds
                while time.monotonic() < deadline and not (output / 'stop-after-batch').exists():
                    time.sleep(min(1, max(0, deadline - time.monotonic())))
        result.update(status=f'passed_native_{len(remaining)}_notes_adopted', project_left_open=True, finished_unix=time.time(),
            final_cold_and_UI_checks_pending=True)
    except BaseException as error:
        result.update(status='stopped_for_explicit_reconciliation', error_type=type(error).__name__,
            error=str(error), next_action='Reconcile last retained request; never rerun this importer automatically')
        raise
    finally:
        save(report, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--expected-checkpoint', required=True)
    parser.add_argument('--existing-notes', type=int, choices=range(1,212), required=True)
    parser.add_argument('--quiet-seconds-between-batches', type=int, choices=range(60,301), default=60)
    args = parser.parse_args()
    run(args.controller.resolve(), args.output.resolve(), args.report.resolve(),
        args.expected_checkpoint, args.existing_notes, args.quiet_seconds_between_batches)
