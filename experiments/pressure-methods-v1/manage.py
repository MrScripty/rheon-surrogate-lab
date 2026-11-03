"""Use the ordinary managed controller with explicit preview-budget handling.

Project inspection has a cumulative hash budget. A present input omitted from
that preview is independently hashed here, then checked again by the server's
ordinary recipe material/preflight/admission. No Run or training retry occurs.
The exact inherited controller and two changed seams are captured per command.
"""
import argparse
import importlib.util
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha


def execute(args):
    if args.output.exists():
        raise ValueError('Fresh command directory required; inspect interrupted journals')
    config = read(args.config)
    source = ROOT / config['source_lab'] / 'experiments'
    sys.path[:0] = [str(source / 'pressure-throughput-feedback-v1'), str(source / 'dependency-migration-v1')]
    original = source / 'pressure-throughput-feedback-v1/run_study.py'
    text = original.read_text()
    before = "    require(all(r['status']=='available' and r['integrity']=='verified' for r in refs.values()),'all Project inputs must verify')"
    after = "    verify_project_inputs(args, project, refs)"
    if text.count(before) != 1:
        raise ValueError('Captured inspection gate changed')
    text = text.replace(before, after)
    before = "expected_sha256=refs[slot]['observed_sha256']"
    if text.count(before) != 1:
        raise ValueError('Captured input-binding seam changed')
    text = text.replace(before, "expected_sha256=refs[slot].get('observed_sha256', refs[slot]['sha256'])")
    captures = args.output.with_name(args.output.name + '-controller-source')
    if captures.exists():
        raise ValueError('Fresh controller source capture required')
    captures.mkdir()
    captured = captures / 'run_study.py'
    captured.write_text(text)
    spec = importlib.util.spec_from_file_location('run_study', captured)
    study = importlib.util.module_from_spec(spec)
    sys.modules['run_study'] = study
    spec.loader.exec_module(study)
    original_bindings = study.restore_bindings

    def verify_project_inputs(arguments, project, refs):
        checked = []
        for row in refs.values():
            path = (ROOT / row['path']).resolve(strict=True)
            if (row['status'] != 'available' or not path.is_relative_to(ROOT)
                    or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']):
                raise ValueError('Exact present Project input required: ' + row['id'])
            if row['integrity'] == 'verified':
                if row['observed_sha256'] != row['sha256']:
                    raise ValueError('API hash differs from local bytes')
            elif not (row['integrity'] == 'unverified' and row.get('reason') ==
                      'Present; content exceeds the bounded hash inspection budget.'):
                raise ValueError('Only an exhausted preview budget may use explicit verification')
            checked.append(dict(input=row['id'], sha256=row['sha256'], bytes=row['bytes'],
                api_preview_integrity=row['integrity'], local_full_hash_verified=True))
        write(arguments.output / 'input-verification.json', dict(status='verified', inputs=checked,
            scope='Local full-file identity; API preview statuses preserved. Server material/preflight/admission still verify actual input bytes.',
            original_controller_sha256=sha(original), captured_controller_sha256=sha(captured),
            adapter_sha256=sha(__file__)))

    def bindings(definition, project, training):
        # This internal adapter uses the already verified declaration when the
        # bounded preview omitted an observed digest; original API rows stay intact.
        adapted = dict(project, data=[dict(row, observed_sha256=row.get('observed_sha256', row['sha256']))
                                      for row in project['data']])
        return original_bindings(definition, adapted, training)

    study.verify_project_inputs = verify_project_inputs
    study.restore_bindings = bindings
    from manage import execute as managed
    args.root = ROOT
    started = time.perf_counter()
    try:
        managed(args)
    finally:
        if args.output.is_dir():
            write(args.output / 'controller-execution.json', dict(elapsed_s=time.perf_counter()-started,
                adapter_sha256=sha(__file__), original_controller_sha256=sha(original),
                captured_controller_sha256=sha(captured), automatic_retry=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config','assembly','qualification','tuldok','session-file','output']:
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ['endpoint','catalog-id','project-alias']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):
            setattr(args,name,value.resolve())
    def stop(*_):
        raise KeyboardInterrupt('Controller stopped; bounded catalog Run custody remains authoritative')
    signal.signal(signal.SIGTERM, stop)
    execute(args)
