"""Untimed operator/storage accounting for the frozen complete generator.

Returned storage volumes are logical dispatcher observations, not physical
memory traffic, peak memory, allocation counts or a latency measurement.
Native C++ kernel internals and NumPy allocations are outside this audit.
"""
import argparse
import collections
import contextlib
import os
from pathlib import Path
from support import ROOT, read, write, digest
from divisor_generator import load_model
from diagnose_causal_composition import FIELDS, inventory
from evaluate_recurrent_sampler import identity
import learned_pressure_history as history
import legacy_exact_forcing
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


def storages(value):
    found = {}
    for item in tree_flatten(value)[0]:
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            found[storage.data_ptr()] = storage.nbytes()
    return found


class WorkAudit(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.stage = 'call_overhead'
        self.rows = collections.defaultdict(lambda: dict(calls=0, returned_new_storage_bytes=0))
        self.stage_calls = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        inputs = storages((args, kwargs))
        result = func(*args, **(kwargs or {}))
        outputs = storages(result)
        row = self.rows[(self.stage, str(func))]
        row['calls'] += 1
        row['returned_new_storage_bytes'] += sum(n for p, n in outputs.items() if p not in inputs)
        return result

    def wrap(self, name, function, spatial=False):
        def call(*args, **kwargs):
            previous = self.stage
            self.stage = name+str(args[0]) if spatial else name
            self.stage_calls[self.stage] += 1
            try:
                return function(*args, **kwargs)
            finally:
                self.stage = previous
        return call


@contextlib.contextmanager
def stages(model, audit):
    transport, spatial, forcing = history.transport, history.spatial, legacy_exact_forcing.forcing
    history.transport = audit.wrap('transport', transport)
    history.spatial = audit.wrap('spatial', spatial, spatial=True)
    legacy_exact_forcing.forcing = audit.wrap('forcing', forcing)
    model.history.rhs = audit.wrap('rhs', model.history.rhs)
    model.history.action = audit.wrap('action', model.history.action)
    pressure_forward = model.base.pressure_core.forward
    model.base.pressure_core.forward = audit.wrap('pressure', pressure_forward)
    try:
        yield
    finally:
        history.transport, history.spatial = transport, spatial
        legacy_exact_forcing.forcing = forcing
        del model.history.rhs
        del model.history.action
        del model.base.pressure_core.forward
        if audit.stage != 'call_overhead':
            raise ValueError('Unclosed audit stage')


def run(args):
    if args.output.exists():
        raise ValueError('Fresh untimed accounting output required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Registered accounting source changed: '+path)
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = load_model(reg['pair'], reg['library_path'])
    before = inventory(model)
    rows = []
    try:
        with torch.inference_mode():
            for trajectory in read(ROOT/reg['bank'])['trajectories']:
                path = ROOT/trajectory['receipt']
                if digest(path) != trajectory['receipt_sha256']:
                    raise ValueError('Input receipt changed')
                receipt = read(path)
                for start in reg['starts']:
                    spec = receipt['states'][start]
                    source = path.parent/spec['path']
                    if digest(source) != spec['sha256']:
                        raise ValueError('Input state changed')
                    with np.load(source, allow_pickle=False) as archive:
                        fields = {key: archive[key].copy() for key in FIELDS}
                    for k in reg['horizons']:
                        expected = [identity(v) for v in model.predict(fields, receipt['scene'], start, k)]
                        audit = WorkAudit()
                        with stages(model, audit), audit:
                            outputs = model.predict(fields, receipt['scene'], start, k)
                        if len(outputs) != k or [identity(v) for v in outputs] != expected:
                            raise ValueError('Accounting changed complete output bytes')
                        if audit.stage_calls['forcing'] != 1 or any(audit.stage_calls[key] != k
                                for key in ('transport', 'spatial0', 'spatial2', 'rhs', 'action', 'pressure')):
                            raise ValueError('Every actual stage must be accounted for')
                        operations = [dict(stage=stage, operator=op, **value)
                                      for (stage, op), value in sorted(audit.rows.items())]
                        rows.append(dict(identifier=trajectory['identifier'], start=start, horizons=k,
                            counts_xyz=receipt['scene']['counts'], cells_per_metre=receipt['scene']['cells_per_metre'],
                            complete_output_bitwise_parity=True, operations=operations,
                            stage_calls=dict(audit.stage_calls)))
                        print('Untimed tensor work', trajectory['identifier'], start, k, flush=True)
            if before != inventory(model) or torch.cuda.is_initialized():
                raise ValueError('Audit changed model parameters or device')
    finally:
        model.close()
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Registered source changed during audit: '+path)
    write(args.output/'results.json', dict(status='completed_untimed_tensor_work_audit', rows=rows,
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        measurement='Logical returned storage bytes, counted per dispatcher operation if absent from that operation inputs',
        limitations=['Not physical traffic or allocation counts', 'Not peak memory',
                     'C++ kernel internals and NumPy allocations are not visible'],
        latency_claim=False, clocks_read=False, optimizer_updates=0, full_generator_target_pass=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
