"""Exercise real physical objectives across all cadences, heads and pilot scenes.

No optimizer is created. Future states and fine RHS are read only for losses.
The stationary-prefix change must match both the current and native future
state whenever it changes an older output; active outputs must stay identical.
"""
import argparse
import math
import os
from pathlib import Path
import time

import torch

from support import ROOT, read, write, digest
from long_temporal_dataset_v1 import build_index, LongTemporalDataset, FIELDS
from shared_temporal_generator_v3 import SharedTemporalGenerator
from shared_temporal_generator_v2 import SharedTemporalGenerator as PreviousGenerator
from internal_residual_loss_v1 import metrics
from recurrent_sampler_native import dimensions
from diagnose_causal_composition import inventory


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only physical gradient probe required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen gradient source changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result = dict(status='running', source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol), inference_device='cuda',
        cpu_model_forward=False, new_parameter_fits=0, optimizer_updates=0,
        backward_calls=0, reserved52_references_opened=0,
        full_generator_target_pass=False, rows=[], changed_stationary_heads=0,
        unchanged_active_heads=0, failures=[])
    model = None
    try:
        dataset = LongTemporalDataset(build_index([ROOT / p for p in reg['audits']]))
        examples = [e for e in dataset.index['examples']
                    if e['start_fine_step'] in reg['starts']]
        normalization = read(ROOT / reg['normalization'])
        for hidden in reg['hidden_widths']:
            model = SharedTemporalGenerator(reg['transport_pair'], reg['shared_pair'],
                reg['build'], reg['control_build'], hidden=hidden,
                force_sum_build=reg['force_sum_build'])
            model.network.requires_grad_(True)
            before = inventory(model)
            for example in examples:
                model.network.zero_grad(set_to_none=True)
                scene = dataset.receipts[example['identifier']][1]['scene']
                inputs = dataset.inputs(example)
                fields = {n: torch.from_numpy(inputs['current'][n]).to('cuda') for n in FIELDS}
                shape = tuple(reversed(scene['counts']))
                shapes = dict(pressure=shape, **dict(zip(FIELDS[1:], dimensions(tuple(scene['counts'])))))
                rate, k, start = (example[n] for n in ('output_steps_per_s', 'K', 'start_fine_step'))
                packet, events, mask = model.packets(scene, start, rate, k)
                outputs = model.predict_packet(fields, scene, rate, k, packet, events, mask)
                with torch.no_grad():
                    previous = PreviousGenerator.predict_packet(model, fields, scene, rate, k, packet, events, mask)
                objective = fields['pressure'].new_zeros(())
                row = dict(identifier=example['identifier'], start_fine_step=start,
                    output_steps_per_s=rate, K=k, hidden_width=hidden, head_losses=[])
                for lead, (output, old, step) in enumerate(zip(outputs, previous, example['target_fine_steps']), 1):
                    truth = dataset._load(example['identifier'], step)
                    sample = dict(h=1 / scene['cells_per_metre'], rho=1000.,
                        target_rhs=torch.from_numpy(truth['rhs']).to('cuda').reshape(shape))
                    for n in FIELDS:
                        sample['target_' + n] = torch.from_numpy(truth[n]).to('cuda').reshape(shapes[n])
                    changed = any(not torch.equal(output[n], old[n]) for n in FIELDS)
                    if changed:
                        if not all(torch.equal(output[n].reshape(shapes[n]), sample['target_' + n])
                                   and torch.equal(output[n], fields[n]) for n in FIELDS):
                            raise AssertionError('Changed stationary prefix differs from native complete state')
                        result['changed_stationary_heads'] += 1
                    else:
                        result['unchanged_active_heads'] += 1
                    values = metrics({n: v.reshape(shapes[n]) for n, v in output.items()}, sample, normalization)
                    objective = objective + sum(values.values()) / k
                    row['head_losses'].append(dict(lead=lead, changed=changed,
                        components={n: float(v.detach()) if math.isfinite(float(v.detach())) else None
                                    for n, v in values.items()}))
                finite = bool(torch.isfinite(objective))
                row['objective'] = float(objective.detach()) if finite else None
                objective.backward()
                result['backward_calls'] += 1
                expected_unused = {n for n, p in model.network.named_parameters()
                                   if k == 1 and n.startswith('interior_')}
                missing = [n for n, p in model.network.named_parameters()
                           if p.grad is None and n not in expected_unused]
                nonfinite = [n for n, p in model.network.named_parameters()
                             if p.grad is not None and not torch.isfinite(p.grad).all()]
                bad = missing + nonfinite
                row['expected_unused_K1_interior_parameters'] = sorted(expected_unused)
                row['unexpected_missing_gradients'] = missing
                row['nonfinite_gradients'] = nonfinite
                row['bad_gradients'] = bad
                row['finite'] = finite and not bad
                result['rows'].append(row)
                if not row['finite']:
                    result['failures'].append({n: row[n] for n in
                        ('identifier', 'start_fine_step', 'output_steps_per_s', 'K', 'hidden_width', 'bad_gradients')})
                del outputs, previous, objective, values
                if (time.monotonic() - started > reg['maximum_wall_seconds']
                        or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']):
                    raise ValueError('Registered real-gradient resource budget exceeded')
            if inventory(model) != before:
                raise AssertionError('Backward-only qualification changed parameters')
            model.close()
            model = None
            write(args.output / 'results.json', result)
        if len(result['rows']) != reg['expected_blocks'] or not result['changed_stationary_heads']:
            raise AssertionError('Required complete population and stationary regression witness missing')
        result['status'] = 'failed_real_physical_gradients' if result['failures'] else 'passed_all12_real_physical_gradients_and_stationary_native_regression'
        if result['failures']:
            raise AssertionError('Nonfinite real physical gradients remain')
    except BaseException as error:
        if result['status'] == 'running':
            result['status'] = 'failed'
        result['error'] = repr(error)
        raise
    finally:
        if model is not None:
            model.close()
        result.update(elapsed_seconds=time.monotonic() - started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
