"""Measure native metadata-diagnostic construction, never model inference.

The exact retained warning bytes are hashed for comparison between the original
and indexed implementations. Ordinary constructor dependencies are references
only; no SQL, Dataset, source capability, worker or model is executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/model-portfolio-migration-v1'


def run(source, output):
    if not source.is_relative_to(BASE / 'sources') or output.exists():
        raise ValueError('Task-owned source and fresh evidence required')
    source_raw = (source / 'project_extraction.py').read_bytes()
    sys.path.insert(0, str(source))
    from project_extraction import _Capture
    from project_transfer import CapturedArtifacts
    measurements = []
    with tempfile.TemporaryDirectory(prefix='rheon-diagnostic-scaling-') as temporary:
        owner = SimpleNamespace(
            db=SimpleNamespace(dataset_id='a' * 32, lock=threading.RLock(), path=Path(temporary)),
            snapshot=dict(project_id='a' * 32, records=[]), artifacts=CapturedArtifacts())
        for size in (4000, 16000):
            captured = _Capture(owner)
            started = time.process_time()
            for ordinal in range(size):
                captured.issue(('run_history', f'{ordinal:032x}'), 'missing_regenerable',
                               'Historical runtime bytes unavailable', warning=True)
            elapsed = time.process_time() - started
            if len(captured.warnings) != size or captured.blockers:
                raise ValueError('Native diagnostic population changed')
            raw = json.dumps(captured.warnings, ensure_ascii=False, separators=(',', ':')).encode()
            measurements.append(dict(issues=size, CPU_seconds=elapsed,
                                     exact_ordered_diagnostic_sha256=hashlib.sha256(raw).hexdigest()))
        captured = _Capture(owner)
        parent = 'run_history', 'b' * 32
        for detail in ('λ' * 1000 + 'one', 'λ' * 1000 + 'two', 42, '42'):
            captured.issue(parent, 'missing_source', detail, warning=True)
        captured.issue(parent, 'missing_source', '42')
        normalized_raw = json.dumps(dict(warnings=captured.warnings, blockers=captured.blockers),
                                    ensure_ascii=False, separators=(',', ':')).encode()
    if (source / 'project_extraction.py').read_bytes() != source_raw:
        raise ValueError('Observed source changed during the benchmark')
    ratio = measurements[1]['CPU_seconds'] / measurements[0]['CPU_seconds']
    result = dict(schema='rheon.extraction-diagnostic-scaling.v1',
                  scope='Metadata warning construction only; not model inference or project migration latency.',
                  source=str(source.relative_to(ROOT)),
                  source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                  extraction_source_sha256=hashlib.sha256(source_raw).hexdigest(), measurements=measurements,
                  fourfold_population_CPU_growth=ratio, growth_below_eight=ratio < 8,
                  normalized_diagnostic_sha256=hashlib.sha256(normalized_raw).hexdigest(),
                  new_neural_forwards=0, SQL_executed=False,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('scope', 'fourfold_population_CPU_growth', 'growth_below_eight')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())
