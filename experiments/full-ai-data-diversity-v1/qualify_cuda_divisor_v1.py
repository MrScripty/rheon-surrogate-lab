"""Source-bound CUDA full-state qualification and explicitly scoped GPU timings."""
import argparse
import collections
import hashlib
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from support import ROOT, read, write, digest
import numpy as np
import torch
from cuda_divisor_generator_v1 import CudaDivisorGenerator, prescribed_inputs, FIELDS
from diagnose_causal_composition import inventory, measure


def host(values):
    return {name: value.detach().cpu().contiguous() for name, value in values.items()}


def hashes(values):
    return {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in values.items()}


def stats(values):
    return dict(median=statistics.median(values), p95=float(np.percentile(values, 95)), maximum=max(values), samples=len(values))


def run(args):
    if args.output.exists():
        raise ValueError('Fresh CUDA qualification output required')
    reg = read(args.protocol)
    if reg['inference_device'] != 'cuda' or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Actual registered CUDA runtime required')
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Frozen CUDA qualification dependency changed: '+path)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    model = CudaDivisorGenerator(reg['pair'], reg['pressure_pair'])
    before = inventory(model)
    if any(v.device.type != 'cuda' for v in model.state_dict().values()):
        raise ValueError('All learned tensors must reside on CUDA')
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=10)
    metadata = dict(display_label='GPU (CUDA)', inference_device='cuda',
        device_index=torch.cuda.current_device(), gpu_model=props.name, gpu_uuid=str(getattr(props, 'uuid', 'unavailable')),
        driver_version=driver.stdout.strip() if driver.returncode == 0 else 'unavailable',
        gpu_total_memory_bytes=props.total_memory, torch_version=torch.__version__, cuda_version=torch.version.cuda,
        python_version=platform.python_version(), pressure_dtype='float64', carrier_dtype='float32',
        autocast=False, tf32=False, host_cpu=reg['host_cpu'], cpu_model_forward=False,
        batch_size=1, warmup_blocks=reg['warmup_blocks'], repetitions=reg['repetitions'])
    result = dict(status='running', active=None, device=metadata, rows=[], timings=[],
        protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'], model_identity=model.asset_identity,
        optimizer_updates=0, new_independent_holdout=False, expanded_reserved_references_opened=0,
        cpu_latency_target_verified=False, full_generator_target_pass=False)
    write(args.output/'results.json', result)
    norm = read(ROOT/reg['normalization'])
    reference_sources = {}
    def native(path, receipt, step):
        spec = next(row for row in receipt['states'] if row['step'] == step)
        source = path.parent/spec['path']
        if digest(source) != spec['sha256']:
            raise ValueError('Actual native reference archive changed')
        reference_sources[str(source.relative_to(ROOT))] = spec['sha256']
        with np.load(source, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in (*FIELDS, 'rhs')}
    def score(outputs, receipt, path, identifier, start, k, repeat, scope):
        identities = []
        for lead, output in enumerate(outputs, 1):
            if set(output) != set(FIELDS) or any(value.device.type != 'cuda' for value in output.values()):
                raise ValueError('Every actual intermediate output must be generated on GPU')
            emitted = host(output)
            if any(not bool(value.isfinite().all()) for value in emitted.values()):
                raise ValueError('Nonfinite actual GPU physical output')
            truth = native(path, receipt, start+lead)
            metrics = measure(emitted, truth, receipt['scene'], norm)
            metrics['all_accuracy_criteria_pass'] = all(value <= reg['maximum_error'] for value in metrics['relative'].values())
            identity = hashes(emitted); identities.append(identity)
            result['rows'].append(dict(identifier=identifier, K=k, start=start, step=start+lead, lead=lead,
                repeat=repeat, timing_scope=scope, inference_device='cuda',
                counts_xyz=receipt['scene']['counts'], cells_per_metre=receipt['scene']['cells_per_metre'],
                metrics=metrics, output_sha256=identity))
        return identities
    try:
        with torch.inference_mode():
            for selected in reg['trajectories']:
                path = ROOT/selected['receipt']
                if digest(path) != selected['receipt_sha256']:
                    raise ValueError('Consumed diagnostic receipt changed')
                receipt = read(path); scene = receipt['scene']
                tick = time.perf_counter()
                model.prepare_geometry(tuple(scene['counts']))
                torch.cuda.synchronize()
                geometry_ms = 1000*(time.perf_counter()-tick)
                for start in reg['starts']:
                    initial = native(path, receipt, start)
                    resident = {name: torch.from_numpy(initial[name]).to('cuda') for name in FIELDS}
                    for k in reg['horizons']:
                        result['active'] = dict(identifier=selected['identifier'], start=start, K=k)
                        write(args.output/'results.json', result)
                        external = prescribed_inputs(scene, start, k)
                        for _ in range(reg['warmup_blocks']):
                            model.predict(resident, scene, start, k, external)
                        torch.cuda.synchronize()
                        event_times, wall_times, end_to_end_times, canonical = [], [], [], None
                        torch.cuda.reset_peak_memory_stats()
                        for repeat in range(reg['repetitions']):
                            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            torch.cuda.synchronize(); tick = time.perf_counter(); begin.record()
                            outputs = model.predict(resident, scene, start, k, external)
                            end.record(); end.synchronize()
                            wall_times.append(1000*(time.perf_counter()-tick))
                            event_times.append(begin.elapsed_time(end))
                            actual = score(outputs, receipt, path, selected['identifier'], start, k, repeat, 'resident_block')
                            if canonical is None: canonical = actual
                            elif canonical != actual: raise ValueError('Repeated CUDA output bytes changed')
                            # Full host-input call: state upload, prescribed input construction/upload,
                            # complete generation and host transfer of every intermediate output.
                            torch.cuda.synchronize(); tick = time.perf_counter()
                            uploaded = {name: torch.from_numpy(initial[name]).to('cuda') for name in FIELDS}
                            supplied = prescribed_inputs(scene, start, k)
                            full = model.predict(uploaded, scene, start, k, supplied)
                            copied = [host(value) for value in full]
                            torch.cuda.synchronize()
                            end_to_end_times.append(1000*(time.perf_counter()-tick))
                            if [hashes(value) for value in copied] != canonical:
                                raise ValueError('Resident and full-transfer calls changed physical output bytes')
                        result['timings'].append(dict(identifier=selected['identifier'], K=k, start=start,
                            inference_device='cuda', display_label='GPU (CUDA)', counts_xyz=scene['counts'],
                            cells_per_metre=scene['cells_per_metre'], geometry_setup_ms=geometry_ms,
                            full_block_ms=dict(cuda_event=stats(event_times), synchronized_resident_wall=stats(wall_times),
                                               synchronized_host_input_and_output_wall=stats(end_to_end_times)),
                            amortized_ms_per_generated_step=dict(cuda_event=stats([x/k for x in event_times]),
                                synchronized_resident_wall=stats([x/k for x in wall_times]),
                                synchronized_host_input_and_output_wall=stats([x/k for x in end_to_end_times])),
                            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                            excludes_one_time_setup=True, all_complete_generator_stages_included=True,
                            host_input_scope='Includes current-state upload, known scene/time controls with upload, and all intermediate output downloads; excludes model loading and geometry preparation.'))
                        print('GPU (CUDA)', selected['identifier'], 'start', start, 'K', k,
                              'resident median ms/step', statistics.median(wall_times)/k, flush=True)
            if inventory(model) != before:
                raise ValueError('GPU evaluation changed restored learned tensors')
        for path, expected in reg['source_sha256'].items():
            if digest(ROOT/path) != expected:
                raise ValueError('CUDA qualification dependency changed during execution: '+path)
        result.update(status='completed_cuda_divisor_block_qualification', active=None,
            all_accuracy_pass=all(row['metrics']['all_accuracy_criteria_pass'] for row in result['rows']),
            every_timed_resident_output_scored=True, full_transfer_output_hashes_match_resident=True,
            worst_error={field:max(row['metrics']['relative'][field] for row in result['rows'])
                         for field in ('equation', 'pressure', 'velocity', 'tracer')},
            reference_sources=reference_sources, learned_inventory_before_and_after=before,
            own_history_240_steps=False, qualification_scope='Consumed diagnostic short blocks only; not full trajectory or new heldout verification.')
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['active'] = None
        model.close()
        write(args.output/'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
