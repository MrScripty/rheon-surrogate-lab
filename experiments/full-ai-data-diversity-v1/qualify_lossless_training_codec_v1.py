"""Measure exact candidate training-input compression on real late native states."""
import argparse
import io
import os
from pathlib import Path
import time

import numpy as np

from support import ROOT, read, write, digest
from lossless_training_array_codec_v1 import encode, decode


def run(args):
    if args.output.exists():
        raise ValueError('Fresh codec qualification required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen codec source changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    receipt_path = ROOT / reg['receipt']
    receipt = read(receipt_path)
    states = {s['fine_step']: s for s in receipt['states']}
    args.output.mkdir(parents=True)
    result = dict(status='running', source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol), rows=[], new_parameter_fits=0,
        cpu_model_forward=False, learned_forwards=0, reserved52_references_opened=0,
        full_training_archive_qualified=False, full_generator_target_pass=False)
    began = time.monotonic()
    try:
        for start in reg['starts']:
            previous = {}
            for offset in (0, 2, 4, 6, 8, 12, 16, 24, 32):
                step = start + offset
                entry = states[step]
                source = receipt_path.parent / entry['path']
                if digest(source) != entry['sha256']:
                    raise ValueError('Original typed state changed')
                with np.load(source, allow_pickle=False) as archive:
                    arrays = {n: archive[n].copy() for n in reg['fields']}
                baseline = io.BytesIO()
                np.savez_compressed(baseline, **arrays)
                compressed = 0
                for name, array in arrays.items():
                    native = entry['arrays'][name]
                    if (array.dtype.str != native['dtype'] or list(array.shape) != native['shape']
                            or __import__('hashlib').sha256(array.tobytes()).hexdigest() != native['sha256']):
                        raise ValueError('Native array identity differs')
                    spec, payload = encode(array, previous.get(name))
                    restored = decode(spec, payload, previous.get(name))
                    if restored.dtype != array.dtype or restored.shape != array.shape or restored.tobytes() != array.tobytes():
                        raise AssertionError('Lossless bit restoration failed')
                    compressed += len(payload)
                    previous[name] = restored
                result['rows'].append(dict(start_fine_step=start, fine_step=step,
                    complete_state_and_rhs_fields=len(arrays), raw_bytes=sum(a.nbytes for a in arrays.values()),
                    same_fields_NPZ_bytes=len(baseline.getvalue()), codec_payload_bytes=compressed,
                    every_typed_array_bitwise_exact=True))
                if time.monotonic() - began > reg['maximum_wall_seconds']:
                    raise TimeoutError('Finite codec qualification budget')
        result.update(status='passed_real_native_training_codec_roundtrips_and_size_measurement',
            same_fields_NPZ_bytes=sum(r['same_fields_NPZ_bytes'] for r in result['rows']),
            codec_payload_bytes=sum(r['codec_payload_bytes'] for r in result['rows']),
            checked_arrays=sum(r['complete_state_and_rhs_fields'] for r in result['rows']),
            note='Only payload sizes on declared late windows; complete bank size, framing, managed staging custody and initial trainer remain unqualified. No lossy conversion or original snapshot modifications.')
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - began
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
