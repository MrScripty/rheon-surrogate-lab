"""Retain isolated Tuldok test outcomes without capturing credentials or runtime data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--patterns', nargs='+', required=True)
    parser.add_argument('--name-pattern')
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    if not os.environ.get('TULDOK_DATABASE_URL'):
        raise ValueError('An explicitly isolated PostgreSQL fixture is required.')
    result = {'schema': 'pressure-throughput-test-receipt-v1',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': str(args.source.resolve()), 'python': str(args.python.absolute()),
        'source_sha256': {name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
                          for name in ('runs.py', 'run_capacity.py')},
        'database': 'explicit isolated fixture; DSN omitted', 'tests': []}
    started = time.perf_counter()
    for pattern in args.patterns:
        command = [str(args.python.absolute()), '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern, '-v']
        if args.name_pattern:
            command += ['-k', args.name_pattern]
        before = time.perf_counter()
        completed = subprocess.run(command, cwd=args.source, capture_output=True, text=True, timeout=600)
        output = completed.stdout + completed.stderr
        # Even a dependency diagnostic must never turn private configuration into a receipt.
        output = output.replace(os.environ['TULDOK_DATABASE_URL'], '<REDACTED>')
        result['tests'].append({'command': command, 'exit_code': completed.returncode,
            'elapsed_seconds': time.perf_counter() - before, 'output': output[-65536:]})
        print(pattern, completed.returncode, flush=True)
    result['elapsed_seconds'] = time.perf_counter() - started
    result['passed'] = all(test['exit_code'] == 0 for test in result['tests'])
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
