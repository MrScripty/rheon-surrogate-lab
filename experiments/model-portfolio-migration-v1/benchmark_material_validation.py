"""Compare fixed-candidate material lookups on a retained real runtime record.

CPU metadata validation only. No runtime installation, source import, model
deserialization, inference or old-environment filesystem probing is requested.
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


def run(source, output, fixed):
    if not source.is_relative_to(BASE / 'sources') or output.exists():
        raise ValueError('Task-owned source and fresh output required')
    project = ROOT / 'projects/rheon-model-portfolio-unified-03'
    manifest_raw = (project / '.tuldok/project.json').read_bytes()
    manifest = json.loads(manifest_raw)
    entry = next(row for row in manifest['inventory'] if row['family'] == 'runtime_material')
    record_raw = (project / '.tuldok' / entry['path']).read_bytes()
    if len(record_raw) != entry['bytes'] or hashlib.sha256(record_raw).hexdigest() != entry['sha256']:
        raise ValueError('Original native runtime identity differs')
    record = json.loads(record_raw)
    original = record['value']
    records = [dict(family=entry['family'], id=entry['id'], document=record)]
    sys.path.insert(0, str(source))
    import project_material
    ref = original['ref']
    started = time.process_time()
    if fixed:
        view = project_material.MaterialSnapshot(records)
    else:
        view = project_material.ProjectMaterial(SimpleNamespace(
            snapshot=dict(project_id=manifest['project_id'], records=records), check=lambda: None))
    outputs = [view.runtime(ref) for _ in range(4)]
    elapsed = time.process_time() - started
    if not all(value == original for value in outputs):
        raise ValueError('Retained original runtime result changed')
    if (project / '.tuldok/project.json').read_bytes() != manifest_raw:
        raise ValueError('Source project changed during the observation')
    result = dict(schema='rheon.fixed-material-validation-benchmark.v1',
                  scope='CPU metadata validation only; not model inference or whole migration latency.',
                  source=str(source.relative_to(ROOT)),
                  source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                  module_sha256=hashlib.sha256((source / 'project_material.py').read_bytes()).hexdigest(),
                  source_runtime_record_sha256=entry['sha256'], source_checkpoint=manifest['checkpoint'],
                  private_fixed_view=fixed, runtime_lookups=4, CPU_seconds=elapsed,
                  exact_original_runtime_results=True,
                  result_sha256=hashlib.sha256(json.dumps(outputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                  new_neural_forwards=0, runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('scope', 'private_fixed_view', 'CPU_seconds', 'exact_original_runtime_results')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixed-view', action='store_true')
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve(), args.fixed_view)
