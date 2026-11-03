"""Exercise the real measurement read owner on a tiny validated checkpoint.

This isolates unused graph copying, not whole-Open latency or SQL reconstruction.
The installed application's source and scientific project remain untouched.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch


def run(source, output):
    if output.exists():
        raise ValueError('A fresh regression report is required')
    sys.path[:0] = [str(source), str(source/'tests')]
    import project_session
    import project_measurements
    from project_projection import validate_project_records
    from test_project_receipt_validation import candidate, PROJECT

    original = candidate()
    measurement = next(row for row in original if row['family'] == project_measurements.FAMILY)
    snapshot = next(row for row in original if row['family'] == 'project_snapshot')
    hashes = {snapshot['document']['value']['manifest_sha256'], measurement['document']['id']}
    records = [snapshot, *[row for row in original if row['family'] == 'project_artifact' and row['id'] in hashes],
               measurement, dict(family='project_artifact', id='f'*64,
                                 document=dict(format='project_artifact_v1', sha256='f'*64, bytes=7))]
    validate_project_records(records, project_id=PROJECT, artifacts=None)
    expected = {measurement['id']: copy.deepcopy(measurement['document'])}
    fixed = dict(project_id=PROJECT, name='Read-copy regression', checkpoint='e'*64, records=records)
    session = object.__new__(project_session.ProjectSession)
    session._snapshot, session.available, session.lock = copy.deepcopy(fixed), True, threading.RLock()
    session.store = SimpleNamespace(summary=lambda: {key: fixed[key] for key in ('project_id','name','checkpoint')})
    owner = object.__new__(project_measurements.ProjectMeasurements)
    owner.session = session
    copied_unrelated = []
    original_copy = project_session.copy_validated_json

    def observe(value):
        if isinstance(value, dict) and 'records' in value:
            copied_unrelated.extend(row['id'] for row in value['records'] if row['id'] == 'f'*64)
        return original_copy(value)

    started = time.monotonic()
    with patch.object(project_session, 'copy_validated_json', observe):
        observed = owner._documents()
    literal_equal = observed == expected
    observed[measurement['id']]['title'] = 'Caller edit'
    isolated = session._snapshot == fixed
    result = dict(status='passed' if literal_equal and isolated and not copied_unrelated else 'failed',
        source=str(source), source_files={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
            for name in ('project_measurements.py','project_session.py')},
        records=len(records), measurement_documents=len(expected), unrelated_graph_copies=len(copied_unrelated),
        literal_documents_equal=literal_equal, caller_mutation_isolated=isolated,
        elapsed_seconds=time.monotonic()-started,
        scope=__doc__.strip(), SQL_opened=False, project_or_payload_copies=0, whole_Open_latency_qualified=False)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result), flush=True)
    if result['status'] != 'passed':
        raise AssertionError('Measurement read copied an unrelated retained record')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.source.resolve(), arguments.report.resolve())
