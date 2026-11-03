"""One managed optimizer at a time; overlap bounded CPU qualification with CUDA."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import signal
import sys
import threading
import time

from execution import LAB, command, read, utc, write

sys.path.insert(0, str(LAB/'experiments/live-project-v1'))
from observer import Observer, Watch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--torch-python', required=True)
    parser.add_argument('--managed-controller', type=Path, required=True,
                        help='Existing authenticated adapter to conditioning workflow.execute')
    args = parser.parse_args()
    sequence = read(args.sequence)
    root = LAB/sequence['root']
    if (root/'execution.json').exists():
        raise FileExistsError('sequence already attempted; inspect retained outcome')
    for trial in sequence['trials']:
        if not (root/trial['name']/'project/source-witness.json').is_file():
            raise ValueError('assemble committed sources before executing the sequence')
    # The adapter contains no secret values. Record its exact bytes for reproduction.
    adapter = root/'managed-controller.py'
    adapter.write_bytes(args.managed_controller.read_bytes())
    report = {'started_at_utc': utc(), 'status': 'running',
              'managed_controller_sha256': hashlib.sha256(adapter.read_bytes()).hexdigest(),
              'sequence_sha256': hashlib.sha256(args.sequence.read_bytes()).hexdigest()}
    write(root/'execution.json', report)
    started = time.monotonic()
    observer = Observer(LAB/'runs/live-project')
    def poll():
        active = []
        for trial in sequence['trials']:
            for name in ('live','grid-study/live','prior-study/live','long-study/live'):
                path = root/trial['name']/name/'activity.json'
                if path.exists():
                    value = read(path)
                    if value['state'] == 'running':
                        active.append(trial['name']+': '+value['phase']+'; '+value['message'])
        completed = sum((root/t['name']/'cpu-complete.json').exists() for t in sequence['trials'])
        observer.publish('Shape repeatability sequence', ' | '.join(active)[:1600] or
                         'Preparing managed stages and CPU qualification', completed,
                         len(sequence['trials']))
    futures = []
    cancel = threading.Event()
    def interrupted(_signum, _frame):
        cancel.set()
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with Watch(poll), ThreadPoolExecutor(max_workers=1) as pool:
            try:
                for trial in sequence['trials']:
                    directory = root/trial['name']
                    command([sys.executable, adapter, '--root', root,
                             '--trial', trial['name'], '--phase', 'full'], directory, 'managed', cancel)
                    future = pool.submit(command, ['taskset', '-c', '1-23', args.torch_python,
                        Path(__file__).with_name('qualify.py'), '--sequence',
                        args.sequence.absolute(), '--trial', trial['name']], directory, 'qualification', cancel)
                    futures.append(future)
                    for pending in futures:
                        if pending.done():
                            pending.result()
                for future in futures:
                    future.result()
            except BaseException:
                cancel.set()
                for future in futures:
                    future.cancel()
                raise
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__,
                      recovery='Retain attempts; reconcile any Tuldok-owned Run before new execution')
        raise
    finally:
        report.update(finished_at_utc=utc(), elapsed_seconds=time.monotonic()-started)
        write(root/'execution.json', report)
        observer.publish('Shape repeatability '+report['status'],
                         'Retained commands, timings and outcomes; inspect experiment records',
                         sum((root/t['name']/'cpu-complete.json').exists() for t in sequence['trials']),
                         len(sequence['trials']), state='completed' if report['status']=='completed' else 'failed')


if __name__ == '__main__':
    main()
