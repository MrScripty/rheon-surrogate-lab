"""Measure original versus grouped presence reads on the retained history page.

This isolates artifact presence inspection, not the complete Run API or UI.
No payload hashing, SQL access, simulation, inference or project writes occur.
"""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/model-portfolio-migration-v1/sources/Tuldok-artifact-inspection'
PROJECT = ROOT / 'projects/rheon-model-portfolio-history-rehearsal-01'
REPORT = ROOT / 'docs/plans/model-portfolio-migration-v1/reports/artifact-presence-performance-01.json'
RUN_ID = '74240450dfb642599b9aea0d83517a0f'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if REPORT.exists():
        raise ValueError('Existing measurement must be reconciled, not overwritten')
    sys.path.insert(0, str(SOURCE))
    from project_store import ProjectStore
    from project_artifacts import ProjectArtifacts
    report = dict(schema='rheon.artifact-presence-performance.v1', status='running',
                  claim='Presence inspection contributor for the default 40-payload historical Run page',
                  limitations='Four interleaved same-process samples; no full Run/UI latency or model timing claim',
                  methods=dict(individual='Original inspect loop, one full closure scan per payload',
                               grouped='inspect_many, full closure scans before and after the page'),
                  python=sys.version, platform=platform.platform(), samples=[],
                  runner_sha256=sha(Path(__file__)),
                  source_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
                  source_sha256={name: sha(SOURCE / name) for name in
                                 ('project_store.py', 'project_artifacts.py', 'project_history.py')},
                  neural_inference=False, live_catalog_accessed=False, project_writes=False)
    def save():
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
    save()
    store = artifacts = None
    try:
        store = ProjectStore.open(str(PROJECT))
        before = store.snapshot()
        docs = [v['document'] for v in before['records']
                if v['family'] == 'run_history' and v['document']['run']['id'] == RUN_ID]
        if len(docs) != 1:
            raise ValueError('Exactly one original Run required')
        declarations = [dict(format='project_artifact_v1', **edge['blob'])
                        for edge in docs[0]['payloads'][:40]]
        if len(declarations) != 40:
            raise ValueError('The real default page must contain 40 payloads')
        report.update(checkpoint=before['checkpoint'], records=len(before['records']),
                      run_id=RUN_ID, page_size=len(declarations),
                      declarations_sha256=hashlib.sha256(json.dumps(declarations, sort_keys=True).encode()).hexdigest())
        artifacts = ProjectArtifacts(store)
        expected = None
        for method in ('individual', 'grouped', 'grouped', 'individual'):
            print(json.dumps(dict(starting=method, sample=len(report['samples']))), flush=True)
            with patch.object(store, 'snapshot', wraps=store.snapshot) as checks:
                wall = time.perf_counter()
                cpu = time.process_time()
                values = ([artifacts.inspect(value) for value in declarations]
                          if method == 'individual' else artifacts.inspect_many(declarations))
                sample = dict(method=method, wall_seconds=time.perf_counter() - wall,
                              CPU_seconds=time.process_time() - cpu, closure_scans=checks.call_count)
            if expected is None:
                expected = values
            if values != expected or any(v['observed_sha256'] is not None for v in values):
                raise ValueError('Presence results changed or claimed an unmeasured digest')
            if sample['closure_scans'] != (40 if method == 'individual' else 2):
                raise ValueError('Unexpected closure validation count')
            report['samples'].append(sample)
            save()
            print(json.dumps(sample), flush=True)
        if before != store.snapshot():
            raise ValueError('Canonical project changed during measurement')
        report.update(status='qualified_presence_inspection_contributor', all_results_exact=True,
                      canonical_snapshot_unchanged=True,
                      observation_states={state: sum(v['state'] == state for v in expected)
                                          for state in sorted({v['state'] for v in expected})})
    except BaseException as error:
        report.update(status='failed_or_incomplete', error_type=type(error).__name__)
        raise
    finally:
        if artifacts is not None:
            artifacts.close()
        if store is not None:
            store.close()
        save()


if __name__ == '__main__':
    main()
