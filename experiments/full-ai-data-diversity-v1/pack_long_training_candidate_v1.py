"""Capture all24 training masters as bounded lossless candidate input transport.

No fitting or reference-policy adoption occurs here. Only training fields are
opened. Native snapshots and prior failed attempts are retained independently.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

from support import ROOT, read, write, digest
from long_temporal_dataset_v1 import build_index, LongTemporalDataset
from lossless_training_array_codec_v1 import encode, decode

FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer', 'rhs')


def run(args):
    reg = read(args.protocol)
    archive_root = Path(reg['archive_root'])
    if args.output.exists() or archive_root.exists():
        raise ValueError('Fresh independently owned candidate transport required')
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Captured source changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    if shutil.disk_usage(archive_root.parent).free < reg['minimum_free_bytes'] + reg['maximum_archive_bytes']:
        raise ValueError('Finite archive admission exceeds available storage')
    args.output.mkdir(parents=True)
    archive_root.mkdir()
    index = build_index([ROOT / p for p in reg['audits']])
    dataset = LongTemporalDataset(index)
    train = {n: m for n, m in index['masters'].items() if m['split'] == 'train'}
    if len(train) != 24:
        raise ValueError('All24 independent training masters required')
    archive_path = archive_root / 'fields.zip'
    manifest = dict(schema='rheon.lossless-long-training-candidate.v1', fields=list(FIELDS),
        reset_interval=reg['reset_interval'], masters={}, training_policy_adopted=False,
        full36_eligibility_pending=True, source_audits_sha256=index['audits'])
    began = time.monotonic()
    result = dict(status='running', source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol), new_parameter_fits=0, learned_forwards=0,
        cpu_model_forward=False, reserved52_references_opened=0,
        training_policy_adopted=False, full_generator_target_pass=False, rows=[])
    try:
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as captured:
            for identifier, master in train.items():
                _, receipt, states = dataset.receipts[identifier]
                selected = sorted({s for w in receipt['windows'] for s in
                    [w['start_fine_step'], *w['target_fine_steps']]})
                case = dict(scene=receipt['scene'], split='train',
                    physical_scenario_group_id=master['physical_scenario_group_id'],
                    native_receipt=master['receipt'], native_receipt_sha256=master['receipt_sha256'],
                    reference_numerical_contract=receipt.get('reference_numerical_contract', 'original normal-only F32'),
                    windows=receipt['windows'], states=[])
                previous, previous_step = {}, None
                for ordinal, step in enumerate(selected):
                    if ordinal % reg['reset_interval'] == 0:
                        previous, previous_step = {}, None
                    arrays = dataset._load(identifier, step)
                    entry = dict(fine_step=step, native_npz_sha256=states[step]['sha256'],
                        previous_fine_step=previous_step, arrays={})
                    for name in FIELDS:
                        array = arrays[name]
                        spec, payload = encode(array, previous.get(name))
                        restored = decode(spec, payload, previous.get(name))
                        if restored.tobytes() != array.tobytes():
                            raise AssertionError('Candidate encoding changed native bits')
                        member = f'states/{identifier}/{step:05}/{name}.z'
                        captured.writestr(member, payload)
                        entry['arrays'][name] = dict(**spec, member=member,
                            compressed_bytes=len(payload), payload_sha256=hashlib.sha256(payload).hexdigest())
                        previous[name] = restored
                    previous_step = step
                    case['states'].append(entry)
                    if (time.monotonic() - began > reg['maximum_wall_seconds']
                            or archive_path.stat().st_size > reg['maximum_archive_bytes']
                            or shutil.disk_usage(archive_root).free < reg['minimum_free_bytes']):
                        raise ValueError('Finite transport resource budget exceeded')
                manifest['masters'][identifier] = case
                result['rows'].append(dict(identifier=identifier, states=len(selected),
                    checked_arrays=len(selected) * len(FIELDS), archive_bytes_so_far=archive_path.stat().st_size))
                write(args.output / 'results.json', result)
                print('Lossless captured training master:', identifier, flush=True)
            captured.writestr('manifest.json', json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode())
        write(args.output / 'manifest.json', manifest)
        with zipfile.ZipFile(archive_path) as captured:
            if json.loads(captured.read('manifest.json')) != manifest or captured.testzip() is not None:
                raise AssertionError('Captured container identity/CRC verification failed')
        result.update(status='completed_candidate_lossless24training_master_transport',
            archive=dict(path=str(archive_path), bytes=archive_path.stat().st_size, sha256=digest(archive_path)),
            manifest_sha256=digest(args.output / 'manifest.json'),
            independent_training_masters=24,
            checked_arrays=sum(r['checked_arrays'] for r in result['rows']),
            note='All required native current/future PVT and matching native RHS for every selected all12window. Candidate transport only; full36 reference adoption, managed staging and actual trainer qualification remain required.')
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
