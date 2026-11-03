"""Complete-call CPU measurement with saved output identities and terminal-reference guards."""
import argparse
import collections
import os
from pathlib import Path
import statistics
import time
from support import ROOT, read, write, digest
from divisor_generator import load_model
from diagnose_causal_composition import inventory, FIELDS
from evaluate_recurrent_sampler import identity
import learned_pressure_history as history_module
import legacy_exact_forcing
import numpy as np
import torch


def observe(model, elapsed, counts):
    """Install disjoint stage clocks; return an exact cleanup callback."""
    def wrap(name, fn, spatial=False):
        def timed(*a, **kw):
            key = name+str(a[0]) if spatial else name
            tick = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                elapsed[key] += time.perf_counter()-tick
                counts[key] += 1
        return timed
    originals = {name: getattr(history_module, name) for name in ('transport', 'spatial')}
    original_force = legacy_exact_forcing.forcing
    history_module.transport = wrap('transport', originals['transport'])
    history_module.spatial = wrap('spatial', originals['spatial'], spatial=True)
    legacy_exact_forcing.forcing = wrap('forcing', original_force)
    model.history.rhs = wrap('rhs', model.history.rhs)
    model.history.action = wrap('action', model.history.action)
    pending = []

    def begin(*unused):
        pending.append(time.perf_counter())

    def end(*unused):
        elapsed['pressure'] += time.perf_counter()-pending.pop()
        counts['pressure'] += 1

    handles = [model.base.pressure_core.register_forward_pre_hook(begin),
               model.base.pressure_core.register_forward_hook(end)]

    def cleanup():
        for name, fn in originals.items():
            setattr(history_module, name, fn)
        legacy_exact_forcing.forcing = original_force
        del model.history.rhs
        del model.history.action
        for handle in handles:
            handle.remove()
        if pending:
            raise ValueError('Pressure stage instrumentation did not close')
    return cleanup


def run(args):
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError('Fresh complete generator CPU measurement required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Registered timing source changed: '+path)
    for path, allowed in reg['required_terminal_reference_controllers'].items():
        controller = read(ROOT/path)
        if controller['status'] not in allowed or controller.get('active') is not None:
            raise ValueError('Reference workers must finish before CPU measurement: '+path)
    if not reg['reference_sessions_revalidated_terminal_before_registration']:
        raise ValueError('Terminal process custody must be verified before registration')
    if len(reg['cores']) not in (1, 2, 4) or len(set(reg['cores'])) != len(reg['cores']):
        raise ValueError('One, two or four distinct registered physical cores required')
    if reg['mode'] not in ('profile', 'benchmark'):
        raise ValueError('Separate instrumented profile and acceptance clocks required')
    topology = []
    for core in reg['cores']:
        directory = Path('/sys/devices/system/cpu')/f'cpu{core}'/'topology'
        topology.append(dict(logical_cpu=core,
            package=int((directory/'physical_package_id').read_text()),
            physical_core=int((directory/'core_id').read_text())))
    if len({(row['package'], row['physical_core']) for row in topology}) != len(reg['cores']):
        raise ValueError('Distinct physical cores required; SMT siblings cannot count twice')
    for task in Path('/proc/self/task').iterdir():
        os.sched_setaffinity(int(task.name), set(reg['cores']))
    torch.set_num_threads(len(reg['cores']))
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    setup_tick = time.perf_counter()
    model = load_model(reg['pair'], reg['library_path'])
    setup_s = time.perf_counter()-setup_tick
    before = inventory(model)
    rows = []
    try:
        with torch.inference_mode():
            for trajectory in read(ROOT/reg['bank'])['trajectories']:
                receipt_path = ROOT/trajectory['receipt']
                if digest(receipt_path) != trajectory['receipt_sha256']:
                    raise ValueError('Registered timing input receipt changed')
                receipt = read(receipt_path)
                for start in reg['starts']:
                    spec = receipt['states'][start]
                    source = receipt_path.parent/spec['path']
                    if digest(source) != spec['sha256']:
                        raise ValueError('Registered current-state bytes changed')
                    with np.load(source, allow_pickle=False) as archive:
                        fields = {key: archive[key].copy() for key in FIELDS}
                    for horizons in reg['horizons']:
                        tick = time.perf_counter()
                        warm = model.predict(fields, receipt['scene'], start, horizons)
                        first_call_ms = (time.perf_counter()-tick)*1000
                        expected_all = [identity(value) for value in warm]
                        if len(warm) != horizons:
                            raise ValueError('Every intermediate physical output required')
                        for repeat in range(reg['repeats']):
                            elapsed, counts = collections.defaultdict(float), collections.defaultdict(int)
                            cleanup = observe(model, elapsed, counts) if reg['mode'] == 'profile' else None
                            try:
                                tick = time.perf_counter()
                                outputs = model.predict(fields, receipt['scene'], start, horizons)
                                block_ms = (time.perf_counter()-tick)*1000
                            finally:
                                if cleanup is not None:
                                    cleanup()
                            if len(outputs) != horizons or [identity(v) for v in outputs] != expected_all:
                                raise ValueError('Complete measured output bytes changed')
                            if cleanup is not None and (counts['forcing'] != 1 or any(
                                counts[key] != horizons for key in
                                ('transport', 'spatial0', 'spatial2', 'rhs', 'action', 'pressure'))):
                                raise ValueError('Actual complete learned stage call counts changed')
                            scene = receipt['scene']
                            rows.append(dict(identifier=trajectory['identifier'],
                                counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                                horizons=horizons, start=start, repeat=repeat,
                                complete_block_ms=block_ms, ms_per_generated_step=block_ms/horizons,
                                first_warm_call_ms=first_call_ms,
                                measured_complete_output_identities=expected_all,
                                instrumented_stage_ms={k: v*1000 for k, v in elapsed.items()},
                                actual_stage_calls=dict(counts), every_output_bitwise_verified=True,
                                remaining_call_ms=block_ms-sum(elapsed.values())*1000 if cleanup else None))
                print('Measured', trajectory['identifier'], reg['mode'], flush=True)
        if inventory(model) != before or torch.cuda.is_initialized():
            raise ValueError('Timing changed learned tensors or device')
    finally:
        model.close()
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row['horizons'], row['cells_per_metre'], tuple(row['counts_xyz']))].append(row)
    summaries = []
    for (k, density, grid), selected in sorted(grouped.items()):
        values = sorted(row['ms_per_generated_step'] for row in selected)
        stages = sorted({key for row in selected for key in row['instrumented_stage_ms']})
        summaries.append(dict(horizons=k, cells_per_metre=density, counts_xyz=list(grid),
            observations=len(values), median_ms_per_step=statistics.median(values),
            p95_ms_per_step=values[min(len(values)-1, int(.95*len(values)))],
            maximum_ms_per_step=max(values), maximum_complete_block_ms=k*max(values),
            stage_median_ms={key: statistics.median(row['instrumented_stage_ms'][key]
                for row in selected) for key in stages},
            observed_latency_cap_pass=reg['mode'] == 'benchmark' and max(values) <= reg['maximum_ms_per_step']))
    write(args.output/'results.json', dict(status='completed_divisor_generator_'+reg['mode'],
        rows=rows, summaries=summaries, model_setup_seconds=setup_s,
        cores=reg['cores'], physical_core_topology=topology, source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        model_identity=model.asset_identity, model_inventory=before, optimizer_updates=0,
        profile_overhead_is_not_acceptance_latency=reg['mode'] == 'profile',
        timing_scope='Complete owning predict call: input checks, known forcing, all learned stages, every physical output and binding/context overhead',
        model_target_pass=False, independent_accuracy_claim=False,
        cold_setup_and_first_warm_calls_reported_separately=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
