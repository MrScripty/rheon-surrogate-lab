"""Measure the real retained artifact inventory with grouped native hash checks.

No SQL, service credentials, simulation or neural runtime. The optional single
verified read measures a copy-path contributor, not complete Save As latency.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def run(source, project, report):
    if report.exists() or not source.is_relative_to(ROOT / 'runs/model-portfolio-migration-v1/sources'):
        raise ValueError('Fresh report and explicit task-owned native source required')
    sys.path.insert(0, str(source))
    from project_artifacts import ProjectArtifacts
    from project_fork_composition import artifact_plan
    from project_store import ProjectStore
    before = (project / '.tuldok/project.json').read_bytes()
    with ProjectStore.open(project) as store, ProjectArtifacts(store) as artifacts:
        snapshot = store.snapshot()
        plan = artifact_plan(snapshot['records'])
        declarations = [item['declaration'] for item in plan]
        started = time.monotonic()
        with patch.object(store, 'snapshot', wraps=store.snapshot) as scans:
            observations = artifacts.status_many(declarations)
        elapsed = time.monotonic() - started
        status_scans = scans.call_count
        missing_unique = [item['declaration']['sha256'] for item, observation in zip(plan, observations)
            if observation['state'] == 'missing' and item['preservation'] == 'unique']
        present = [item for item, observation in zip(declarations, observations) if observation['state'] == 'verified']
        started = time.monotonic()
        with patch.object(store, 'snapshot', wraps=store.snapshot) as read_scans:
            with artifacts.open_verified(present[0], maximum_bytes=present[0]['bytes']):
                pass
        read_elapsed = time.monotonic() - started
    if (project / '.tuldok/project.json').read_bytes() != before:
        raise ValueError('Canonical source changed during qualification')
    if missing_unique or status_scans != 2 or any(item['state'] == 'corrupt' for item in observations):
        raise ValueError('Grouped actual custody verification failed')
    result = dict(schema='rheon.native-fork-status-qualification.v1',
        status='exact_actual_artifact_custody_with_two_source_closure_scans',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
        source_worktree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip()),
        source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ('project_artifacts.py', 'project_fork.py', 'project_history.py', 'project_fork_composition.py')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        project_id=snapshot['project_id'], checkpoint=snapshot['checkpoint'],
        artifact_count=len(plan), states=dict(Counter(item['state'] for item in observations)),
        missing_unique=len(missing_unique), status_closure_scans=status_scans,
        status_wall_seconds=elapsed, single_verified_read_closure_scans=read_scans.call_count,
        single_verified_read_wall_seconds=read_elapsed, single_verified_read_bytes=present[0]['bytes'],
        canonical_source_unchanged=True, new_neural_forwards=0,
        limitation='Read-only custody and one copy-path contributor; not complete native Open/Save As or model timing.')
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.project.resolve(), args.report.resolve())
