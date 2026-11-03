"""All development windows: native accuracy and complete GPU generation cost."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from long_temporal_dataset_v1 import FIELDS
from long_temporal_dataset_v3 import LongTemporalDataset
from guided_shared_residual_v3 import GuidedSharedResidualGraph
from shared_temporal_native_transfer_v1 import transaction
from captured_parameter_inventory_v1 import inventory
from diagnose_causal_composition import measure


def hashes(fields):
    return {n: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n, v in fields.items()}


FILES = dict(selected='weights.npz', best_mean='best-mean-weights.npz',
    last='last-evaluated-weights.npz', initial='initial-weights.npz')


def load_network(model, pair, checkpoint):
    receipt = read(pair / 'train/receipt.json')
    history = read(pair / 'restore/history.json')
    name = FILES[checkpoint]
    expected = history['checkpoint_inventories'][name]
    path = pair / 'restore' / name
    if digest(path) != receipt['outputs'][name]['sha256']:
        raise ValueError('Exact restored guided checkpoint artifact required')
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'w4.' + n for n in model.network.state_dict()}:
            raise ValueError('Exact guided checkpoint parameter keys required')
        values = {}
        for n, current in model.network.state_dict().items():
            value = archive['w4.' + n]
            if (value.dtype.str != '<f4' or tuple(value.shape) != tuple(current.shape)
                    or hashlib.sha256(value.tobytes()).hexdigest() != expected[n]['sha256']):
                raise ValueError('Exact guided restored parameter identity differs')
            values[n] = torch.from_numpy(value.copy()).to('cuda')
    model.network.load_state_dict(values, strict=True)
    if inventory(model.network) != expected:
        raise ValueError('Guided CUDA network inventory differs from managed CPU restore')


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only complete development evaluation required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen trained evaluation source changed: ' + name)
    pair = ROOT / reg['pair']
    parity = read(pair / 'receipt.json')
    if parity['status'] != 'completed_managed_guided_fit_and_four_checkpoint_CPUrestore' or not all(parity['artifact_parity'].values()):
        raise ValueError('Completed drained managed fits and exact CPUrestore parity required')
    trained = read(pair / 'train/receipt.json')
    if (trained['model_family'] != 'guided_shared_residual_v3'
            or trained['closure']['config_sha256'] != reg['expected_fit_config_sha256']
            or trained['closure']['inputs']['model_source']['sha256'] != reg['expected_model_source_sha256']):
        raise ValueError('Exact qualified guided architecture and fit source required')
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    dataset = LongTemporalDataset(read(ROOT / reg['qualified_index']))
    examples = [e for e in dataset.index['examples'] if e['split'] == 'development']
    examples.sort(key=lambda e: (e['identifier'], e['output_steps_per_s'], e['K'], e['start_fine_step']))
    norm = read(ROOT / reg['normalization'])
    history = read(pair / 'restore/history.json')
    representatives, aliases, identities = [], {}, {}
    for name in reg['checkpoints']:
        identity = json.dumps(history['checkpoint_inventories'][FILES[name]], sort_keys=True)
        if identity in identities:
            aliases[name] = identities[identity]
        else:
            identities[identity] = name
            representatives.append(name)
    result = dict(status='running', rows=[], preparations=[], inference_device='cuda', display_label='GPU (CUDA)',
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        new_parameter_fits=0, reserved52_references_opened=0, cpu_model_forward=False,
        model_promotion=False, full_generator_target_pass=False,
        checkpoint_aliases=aliases, representative_checkpoints=representatives,
        alias_rule='OnlyexactcompleteF32parameterinventoryidentitywithinthissamearchitecture/base/sourceRun. Rows/timingsmeasuredforrepresentativeonly;aliasesarenotnewmeasurements.',
        pressure_dtype='float64', carrier_dtype='float32', neural_dtype='float32',
        tf32=False, autocast=False, independent_confirmation=False,
        scope='Complete currentnativeCPUfields to allKownedCPUstates: validation/packing/H2D, full known240controlpath andeventmetadata, frozenwholecoarseforecast/Kfineadvances/momenttracerguides/newsharedRMSresidualtrunk/allheadlearnedprojections, reconstruction/clones/D2H/sync. Divide onlybyKphysicaloutputs; setup/warmups and native metrics separate. Development selection data, not untouched reserved confirmation.')
    started = time.monotonic()
    model = None
    try:
        for checkpoint in representatives:
            width = 4
            model = GuidedSharedResidualGraph(reg['transport_pair'], reg['shared_pair'], reg['build'],
                reg['control_build'], hidden=width, force_sum_build=reg['force_sum_build'])
            load_network(model, pair, checkpoint)
            before = inventory(model)
            for example in examples:
                identifier, start, rate, k = (example[n] for n in ('identifier', 'start_fine_step', 'output_steps_per_s', 'K'))
                scene = dataset.receipts[identifier][1]['scene']
                fields = {n: torch.from_numpy(a) for n, a in dataset.inputs(example)['current'].items()}
                original = hashes(fields)
                targets = [dataset._load(identifier, step) for step in example['target_fine_steps']]
                torch.cuda.synchronize()
                tick = time.perf_counter()
                outputs = transaction(model, fields, scene, start, rate, k)
                first_ms = 1000 * (time.perf_counter() - tick)
                expected = [hashes(output) for output in outputs]
                native = [measure(output, truth, scene, norm) for output, truth in zip(outputs, targets)]

                def check(actual):
                    if len(actual) != k or hashes(fields) != original or [hashes(o) for o in actual] != expected:
                        raise ValueError('Repeated complete transaction changed outputs or caller fields')

                for unused in range(reg['warmups']):
                    check(transaction(model, fields, scene, start, rate, k))
                result['preparations'].append(dict(identifier=identifier, start_fine_step=start,
                    output_steps_per_s=rate, K=k, width=width, checkpoint=checkpoint, first_complete_prewarmup_ms=first_ms,
                    warmups=reg['warmups']))
                samples = []
                for unused in range(reg['repetitions']):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    actual = transaction(model, fields, scene, start, rate, k)
                    torch.cuda.synchronize()
                    samples.append(1000 * (time.perf_counter() - tick))
                    check(actual)
                budget = reg['strict_ms_per_physical_output'] * max(1, math.prod(scene['counts']) / 32768)
                result['rows'].append(dict(identifier=identifier, split='development', width=width, checkpoint=checkpoint,
                    managed_CPUrestore_certified=True,
                    counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                    force_profile=scene['forcing_program']['profile'], force_motion=scene['forcing_program']['motion'],
                    start_fine_step=start, output_steps_per_s=rate, K=k,
                    block_duration_s=k / rate, native_metrics_per_output=native,
                    all_accuracy_pass=all(all(v <= reg['maximum_error'] for v in m['relative'].values()) for m in native),
                    block_wall_ms=samples, median_block_ms=float(np.median(samples)),
                    p95_block_ms=float(np.percentile(samples, 95)), maximum_block_ms=max(samples),
                    median_ms_per_physical_output=float(np.median(samples)) / k,
                    maximum_ms_per_physical_output=max(samples) / k,
                    strict_ms_per_physical_output_limit=budget,
                    all_latency_samples_pass=max(samples) / k < budget,
                    output_sha256=expected, all_outputs_owned_and_caller_unchanged=True))
                if (time.monotonic() - started > reg['maximum_wall_seconds']
                        or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']):
                    raise ValueError('Declared finite evaluation budget exceeded')
                if len(result['rows']) % 12 == 0:
                    write(args.output / 'results.json', result)
            if inventory(model) != before:
                raise ValueError('Development evaluation changed trained parameters')
            model.close()
            model = None
        if len(result['rows']) != len(examples) * len(representatives):
            raise ValueError('Incomplete frozen all12 development population')
        result.update(status='completed_guided_all_development_native_accuracy_and_GPU_cost',
            independent_development_masters=12, expected_blocks=len(examples)*len(representatives),
            all_requested_checkpoints_covered=sorted(representatives+list(aliases))==sorted(reg['checkpoints']),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), gpu_name=torch.cuda.get_device_name())
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        if model is not None:
            model.close()
        result['elapsed_seconds'] = time.monotonic() - started
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
