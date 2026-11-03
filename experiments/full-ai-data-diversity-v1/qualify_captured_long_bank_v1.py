"""Read the complete captured bank without ambient native field access."""
import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import random
import sys
import time

from support import ROOT, read, write, digest
from captured_long_training_bank_v1 import CapturedLongTrainingBank, FIELDS


def run(args):
    if args.output.exists():
        raise ValueError('Fresh captured-bank qualification required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen bank dependency changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    args.output.mkdir(parents=True)
    transport = read(ROOT / reg['transport_results'])
    manifest = read(ROOT / reg['manifest'])
    truth = {}
    for identifier, case in manifest['masters'].items():
        receipt_path = ROOT / case['native_receipt']
        if digest(receipt_path) != case['native_receipt_sha256']:
            raise ValueError('Original audited native receipt changed')
        truth[identifier] = {s['fine_step']: s['arrays'] for s in read(receipt_path)['states']}
    archive = transport['archive']
    bank = CapturedLongTrainingBank(archive['path'], {n: archive[n] for n in ('bytes', 'sha256')})
    if bank.manifest != manifest:
        raise ValueError('Captured serialized bank manifest differs')
    guarded = True
    denied = []

    def restrict(event, values):
        if guarded and event == 'open' and isinstance(values[0], (str, bytes)):
            path = Path(os.fsdecode(values[0])).absolute()
            if path.is_relative_to(ROOT):
                denied.append(str(path.relative_to(ROOT)))
                raise PermissionError('Ambient repository reads denied during captured-bank operation')

    sys.addaudithook(restrict)
    result = dict(status='running', source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol), checked_arrays=0, windows=0,
        cpu_model_forward=False, learned_forwards=0, new_parameter_fits=0,
        reserved52_references_opened=0, full_generator_target_pass=False)
    began = time.monotonic()
    try:
        for identifier, records in bank.states.items():
            for step in records:
                actual = bank.load(identifier, step)
                for name in FIELDS:
                    expected = truth[identifier][step][name]
                    value = actual[name]
                    if (value.dtype.str != expected['dtype'] or list(value.shape) != expected['shape']
                            or hashlib.sha256(value.tobytes()).hexdigest() != expected['sha256']):
                        raise AssertionError('Standalone reader differs from independently bound native array')
                    result['checked_arrays'] += 1
                if time.monotonic() - began > reg['maximum_wall_seconds']:
                    raise ValueError('Captured bank qualification budget exceeded')
        windows = list(bank.windows)
        random.Random(231921).shuffle(windows)
        per_scene_weight = Counter()
        for window in windows:
            for step in window['required_fine_steps']:
                actual = bank.load(window['identifier'], step)
                if set(actual) != set(FIELDS):
                    raise ValueError('Random-access complete target state missing')
            per_scene_weight[window['identifier']] += window['scene_weight'] * window['window_weight']
            result['windows'] += 1
        if any(abs(v - 1 / 24) > 1e-12 for v in per_scene_weight.values()):
            raise ValueError('Training scene weights differ')
        if result['checked_arrays'] != transport['checked_arrays']:
            raise ValueError('Incomplete captured native bank checks')
        try:
            (ROOT / 'docs/plans/full-ai-data-diversity-v1/long-scene-reservation-01.json').read_bytes()
        except PermissionError:
            pass
        else:
            raise ValueError('Ambient-read prohibition witness failed')
        if len(denied) != 1:
            raise ValueError('Captured reader attempted an ambient repository read')
        result.update(status='passed_standalone_complete24training_bank_native_bits_random_access_and_no_ambient_reads',
            training_masters=24, shuffled_window_access=True, equal_scene_weights=True,
            peak_decoded_cache_bytes=bank.peak_cache_bytes, payload_reads=bank.payload_reads,
            payload_bytes_read=bank.payload_bytes_read, intentional_ambient_read_rejections=1,
            unexpected_ambient_reads=0, fitting_entrypoint_qualified=False)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        guarded = False
        bank.close()
        result['elapsed_seconds'] = time.monotonic() - began
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
