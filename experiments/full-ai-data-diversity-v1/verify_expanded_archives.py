"""Verify actual expanded training archives without fitting or reserved access."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import time
from support import ROOT, read, write, digest
import numpy as np
import torch


def run(args):
    if args.output.exists():
        raise ValueError('Fresh expanded archive integrity output required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Archive audit source changed: '+path)
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    began = time.monotonic()
    result = dict(status='running', active=None, scenes=[], files=[],
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        archives_verified=0, archive_bytes_verified=0, uncompressed_array_bytes_verified=0,
        field_arrays_verified=0, reserved_references_opened=0, optimizer_updates=0,
        latency_claim=False, reference_temporal_convergence_claim=False)

    def save():
        write(args.output, result)

    save()
    try:
        seen = set()
        for selected in reg['trajectories']:
            path = ROOT/selected['receipt']
            if digest(path) != selected['receipt_sha256']:
                raise ValueError('Completed train receipt changed')
            receipt = read(path)
            if receipt['split'] != 'train' or receipt['status'] != 'completed':
                raise ValueError('Only completed training archives may be opened')
            identifier = receipt['identifier']
            if identifier in seen:
                raise ValueError('Duplicate reference lineage for one training scene')
            seen.add(identifier)
            result['active'] = dict(identifier=identifier)
            save()
            counts = receipt['scene']['counts']
            cells = math.prod(counts)
            sizes = dict(pressure=cells, tracer=cells, rhs=cells, native_input_residual=cells, pressure_base=cells,
                **{name: (counts[a]+1)*math.prod(counts[b] for b in range(3) if b != a)
                   for a, name in enumerate(('vx', 'vy', 'vz'))})
            for spec in receipt['states']:
                if time.monotonic()-began > reg['maximum_wall_seconds']:
                    raise TimeoutError('Finite archive integrity budget reached')
                source = path.parent/spec['path']
                if digest(source) != spec['sha256'] or source.stat().st_size != spec['bytes']:
                    raise ValueError('Actual training archive bytes changed')
                result['archive_bytes_verified'] += source.stat().st_size
                if result['archive_bytes_verified'] > reg['maximum_archive_bytes']:
                    raise ValueError('Finite archive read budget reached')
                fields = {}
                with np.load(source, allow_pickle=False) as archive:
                    if set(archive.files) != set(sizes):
                        raise ValueError('Complete physical and request-label fields required')
                    for name, size in sizes.items():
                        value = archive[name]
                        dtype = '<f4' if name in ('vx', 'vy', 'vz', 'tracer') else '<f8'
                        if value.shape != (size,) or value.dtype.str != dtype or not np.isfinite(value).all():
                            raise ValueError('Typed finite actual reference field required')
                        if spec['arrays'][name] != dict(dtype=dtype, shape=[size]):
                            raise ValueError('Decoded field differs from recorded schema')
                        fields[name] = dict(dtype=dtype, shape=[size],
                            sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                            minimum=float(value.min()), maximum=float(value.max()))
                        result['uncompressed_array_bytes_verified'] += value.nbytes
                        result['field_arrays_verified'] += 1
                if digest(source) != spec['sha256']:
                    raise ValueError('Archive changed while decoded')
                result['files'].append(dict(path=str(source.relative_to(ROOT)), sha256=spec['sha256'],
                    identifier=identifier, step=spec['step'], time_s=spec['time_s'], fields=fields))
                result['archives_verified'] += 1
            result['scenes'].append(dict(identifier=identifier, group=receipt['group'],
                counts_xyz=counts, cells_per_metre=receipt['scene']['cells_per_metre'],
                reference_substeps=receipt.get('reference_substeps', 1),
                receipt=selected['receipt'], receipt_sha256=selected['receipt_sha256'],
                archives=len(receipt['states'])))
            result['active'] = None
            save()
            print('Verified actual training archives', identifier, len(receipt['states']), flush=True)
        if result['archives_verified'] != reg['expected_archives']:
            raise ValueError('Full registered training archive population required')
        for path, expected in reg['source_sha256'].items():
            if digest(ROOT/path) != expected:
                raise ValueError('Archive audit source changed during verification: '+path)
        result.update(status='completed_expanded_archive_integrity_audit',
            every_registered_archive_hash_rechecked_before_and_after_decoding=True,
            every_field_decoded_typed_finite_and_hashed=True)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['active'] = None
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
