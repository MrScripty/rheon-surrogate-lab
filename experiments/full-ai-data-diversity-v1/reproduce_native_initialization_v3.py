"""Red-capable replay of the native first-step viscosity failure."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from reserve_long_scenes_v1 import instantiate
from long_reference_scene_v1 import initial_state
from fine_reference_client_v2 import FineReferenceClient


def run(args):
    if args.output.exists():
        raise ValueError('Fresh exact failure reproduction required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Reproduction source changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    row = next(r for r in read(ROOT / reg['reservation'])['rows'] if r['identifier'] == reg['identifier'])
    scene = instantiate(row)
    with np.load(reg['original_initial_state'], allow_pickle=False) as archive:
        fields = {n: archive[n].copy() for n in ('pressure', 'vx', 'vy', 'vz', 'tracer')}
    args.output.mkdir(parents=True)
    client = FineReferenceClient(scene, args.output / 'native', build=ROOT / reg['native_build'])
    result = dict(status='running', identifier=reg['identifier'],
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        initial_fields_bitwise_equal_failed_capture=True, numerical_controls_unchanged=True,
        cpu_model_forward=False, learned_forwards=0, new_parameter_fits=0,
        reserved52_references_opened=0, reference_cpu_policy=reg['reference_cpu_policy'])
    started = time.monotonic()
    try:
        client.restore(reg['start_fine_step'], fields)
        try:
            frame = client.advance()
            result.update(first_advance_succeeded=frame['frame']['advanced'], frame=frame)
        except BaseException as error:
            result.update(first_advance_succeeded=False, advance_error=repr(error))
    finally:
        try:
            client.close()
        except BaseException as error:
            result['close_error'] = repr(error)
        result['native_custody'] = client.last_custody
        stderr = (args.output / 'native/native-stderr.log').read_text()
        reproduced = (not result.get('first_advance_succeeded', False)
            and 'Newtonian box viscosity rejected operation: ArithmeticFailure' in stderr
            and result['native_custody']['reaped'] and result['native_custody']['returncode'] == 1)
        result.update(status='reproduced_exact_restored_next_step_viscosity_failure' if reproduced else 'original_failure_not_reproduced',
            original_failure_reproduced=reproduced, native_stderr=stderr,
            elapsed_seconds=time.monotonic() - started)
        write(args.output / 'results.json', result)
    if reproduced:
        raise AssertionError('Native restored next-step viscosity ArithmeticFailure reproduced')
    if not result.get('first_advance_succeeded', False):
        raise RuntimeError('Different native failure; inspect exact evidence')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
