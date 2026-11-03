"""Measure material decoding through the real inert history reader.

Uses retained canonical facts with a frozen Session substitute. This isolates
history/material CPU work; it excludes file reading, live Session checks,
authorization, inference and whole-request latency. No project is mutated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/model-portfolio-migration-v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(source, output, fixed, reader='history'):
    if not source.is_relative_to(BASE / 'sources') or output.exists():
        raise ValueError('Task-owned native source and fresh output required')
    project = ROOT / 'projects/rheon-model-portfolio-unified-03'
    original = (project / '.tuldok/project.json').read_bytes()
    manifest = json.loads(original)
    records = []
    for entry in manifest['inventory']:
        raw = (project / '.tuldok' / entry['path']).read_bytes()
        if len(raw) != entry['bytes'] or sha(raw) != entry['sha256']:
            raise ValueError('Retained canonical record differs')
        records.append(dict(family=entry['family'], id=entry['id'], document=json.loads(raw)))
    sys.path.insert(0, str(source))
    import project_material
    from project_history import ProjectHistory, decode_records
    calls = 0
    snapshot_checks = 0
    decoder = project_material.decode_records

    def counted(values):
        nonlocal calls
        calls += 1
        return decoder(values)

    def checked():
        nonlocal snapshot_checks
        snapshot_checks += 1

    project_material.decode_records = counted
    session = SimpleNamespace(snapshot=dict(project_id=manifest['project_id'], records=records),
                              check=checked, artifacts=None,
                              access=SimpleNamespace(require=lambda *a, **kw: None))
    history = ProjectHistory(session)
    started = time.process_time()
    try:
        if fixed:
            facts = decode_records(records, materials=project_material.MaterialSnapshot(records))
        elif reader == 'study':
            from project_study_execution import ProjectStudyExecution
            owner=SimpleNamespace(session=session,materials=project_material.ProjectMaterial(session))
            facts=ProjectStudyExecution._canonical(owner)
        else:
            facts = history._facts()
        elapsed = time.process_time() - started
    finally:
        project_material.decode_records = decoder
    normalized = {name: [[list(key) if isinstance(key, tuple) else key, value]
                         for key, value in sorted(collection.items())]
                  for name, collection in vars(facts).items()}
    encoded = json.dumps(normalized, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if (project / '.tuldok/project.json').read_bytes() != original:
        raise ValueError('Source project changed during the observation')
    if 'torch' in sys.modules:
        raise ValueError('Inert history benchmark imported the neural runtime')
    result = dict(schema='rheon.history-material-decoding-benchmark.v1',
        scope=__doc__.strip(), source=str(source.relative_to(ROOT)),
        source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={name: sha((source / name).read_bytes())
                       for name in ('project_history.py', 'project_material.py', 'project_study_execution.py', 'agent_api.py')},
        source_checkpoint=manifest['checkpoint'], source_manifest_sha256=sha(original),
        fixed_decoder_comparison=fixed, reader=reader, material_decode_calls=calls,
        frozen_session_check_calls=snapshot_checks,
        CPU_seconds=elapsed, result_sha256=sha(encoded), history_Runs=len(facts.runs),
        history_Studies=len(facts.studies), new_neural_forwards=0,
        runner_sha256=sha(Path(__file__).read_bytes()))
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in
                     ('material_decode_calls', 'CPU_seconds', 'result_sha256', 'history_Runs')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixed-decoder-comparison', action='store_true')
    parser.add_argument('--reader', choices=('history','study'), default='history')
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve(), args.fixed_decoder_comparison, args.reader)
