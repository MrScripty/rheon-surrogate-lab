"""Frozen named checkpoints and explicit half-width channel-sliced variants."""
import argparse
import copy
import hashlib
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'experiments/pressure-four-pass-distillation-v1'),
    str(ROOT/'experiments/pressure-pass-sweep-v1'), str(ROOT/'experiments/pressure-methods-v1'),
    str(ROOT/'experiments/pressure-methods-v2')]
from common import read, write, sha

def half_width(network, torch):
    """Keep each stage's first half; decoder concat groups retain their offsets."""
    result = copy.deepcopy(network)
    witness = []
    def convolution(layer, outputs, inputs, name):
        if layer.bias is not None or layer.groups != 1:
            raise ValueError('Only captured bias-free dense convolutions qualified')
        old = layer.weight.detach()
        values = old[outputs][:, inputs].clone()
        layer.weight = torch.nn.Parameter(values, requires_grad=False)
        layer.out_channels = len(outputs)
        layer.in_channels = len(inputs)
        if not torch.equal(layer.weight, old[outputs][:, inputs]):
            raise ValueError('Channel slice differs')
        witness.append(dict(name=name, original_shape=list(old.shape), shape=list(values.shape),
            input_indices=inputs, output_indices=outputs,
            sha256=hashlib.sha256(values.numpy().tobytes()).hexdigest()))
    def block(module, inputs, name):
        selected = inputs
        for index, layer in enumerate(module.layers):
            if isinstance(layer, torch.nn.Conv3d):
                outputs = list(range(layer.out_channels//2))
                convolution(layer, outputs, selected, name+f'.layers.{index}')
                selected = outputs
            elif any(True for _ in layer.parameters()):
                raise ValueError('Unexpected parameterized non-convolution layer')
        return selected
    selected = [0]
    skips = []
    for index, encoder in enumerate(result.encoders):
        selected = block(encoder, selected, f'encoders.{index}')
        skips.append(selected)
    selected = block(result.bottleneck, selected, 'bottleneck')
    for name, layer in result.global_pool.named_modules():
        if isinstance(layer, torch.nn.Conv3d):
            convolution(layer, selected, selected, 'global_pool.'+name)
    original_coarse = network.bottleneck.layers[0].out_channels
    for index, (decoder, skip) in enumerate(zip(result.decoders, reversed(skips))):
        selected = block(decoder, selected+[original_coarse+i for i in skip], f'decoders.{index}')
        original_coarse = network.decoders[index].layers[0].out_channels
    convolution(result.head, [0], selected, 'head')
    return result.eval(), witness

def learned_branch(np, torch):
    prior = read(ROOT/'runs/pressure-methods-v2/evaluation-01/v2-headshape/results.json')
    entry = prior['entry']
    for path, expected in entry['identities'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Frozen coarse-branch source changed')
    with np.load(ROOT/entry['restore']/'weights.npz', allow_pickle=False) as weights:
        keys = ['coarse_logits', 'shape_head.0.weight', 'shape_head.0.bias', 'shape_head.2.weight', 'shape_head.2.bias']
        values = {key:torch.from_numpy(weights[key].copy()) for key in keys}
    return values, entry

def run_policy(wrapper, sample, normalization, branch, kind, max_passes=8):
    import torch
    import coarse, numerics, numerical, scalable
    pressure = sample['base']
    h = sample['cell_size_m']; rho = sample['rho']
    denominator = max(float(torch.linalg.vector_norm(sample['rhs'])), 1e-300)
    limit = max(1e-12, 1e-6*denominator)
    initial = sample['native_input_residual']; calls = 0; coarse_calls = 0
    def residual():
        return initial-numerics.fine_action(pressure-sample['base'], h, rho)
    def accepted(value):
        return float(torch.linalg.vector_norm(value)) <= limit and float(sample['dt'][0]*value.abs().max()/h[0]**3) <= 1e-7
    if kind == 'learned':
        shape = tuple(pressure.shape[2:])
        axes = numerics.geometry(shape, 3)[0]
        features = torch.tensor([math.log(n/16) for n in shape], dtype=torch.float32)
        hidden = torch.tanh(torch.nn.functional.linear(features, branch['shape_head.0.weight'], branch['shape_head.0.bias']))
        logits = branch['coarse_logits']+torch.nn.functional.linear(hidden, branch['shape_head.2.weight'], branch['shape_head.2.bias'])
        gains = torch.cat((logits.new_zeros(1, dtype=torch.float64), max(shape)**2/2*torch.sigmoid(logits.to(torch.float64)))).reshape(4,4,4)
    for _ in range(max_passes):
        r = residual()
        if accepted(r): break
        if kind == 'numerical':
            pressure = pressure+numerical.correction(r, h, rho, 6)
        elif kind == 'learned':
            proposal = coarse.prolong(coarse.restrict(r*(rho/h).reshape(1,1,1,1,1), axes)*gains, axes)
            proposal = proposal-proposal[:,:,:1,:1,:1].clone()
            remaining = r-numerics.fine_action(proposal,h,rho)
            if float(torch.linalg.vector_norm(remaining)) <= float(torch.linalg.vector_norm(r)):
                pressure = pressure+proposal
        else: raise ValueError('Declared coarse method required')
        coarse_calls += 1
        if accepted(residual()): break
        pressure = scalable.fine_step(wrapper, sample, normalization, pressure)
        calls += 1
    r = residual()
    return pressure, dict(network_calls=calls, coarse_calls=coarse_calls, qualified=accepted(r),
        relative_residual=float(torch.linalg.vector_norm(r))/denominator,
        predicted_divergence=float(sample['dt'][0]*r.abs().max()/h[0]**3))

def evaluate(args):
    if args.output.exists(): raise ValueError('Fresh frozen evaluation required')
    os.sched_setaffinity(0, {args.cpu})
    import inference, cases, scalable
    state = inference.load(read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json'), args.checkpoint)
    torch = state.torch
    inventory = state.data.helpers().state_inventory(state.network)
    full, parity = scalable.qualify(state)
    # The complete retained single-step recurrence is checked before slicing.
    recurrence = scalable.qualify_recurrence(state, full, 'fine', passes=2)
    network, slices = half_width(state.network, torch) if args.width == 'half' else (state.network, [])
    wrapper = scalable.ScalableFine(network).eval()
    count = sum(p.numel() for p in network.parameters())
    original_count = sum(p.numel() for p in state.network.parameters())
    if args.width == 'half' and count >= original_count//2: raise ValueError('Variant is not substantially smaller')
    branch, branch_entry = learned_branch(state.np, torch)
    population = cases.load(state)
    if args.screen: population = cases.screen(population)
    rows = []; started = time.perf_counter()
    with torch.no_grad():
        for identity, sample in population:
            for kind in ['numerical', 'learned']:
                tick = time.perf_counter()
                pressure, stats = run_policy(wrapper, sample, state.normalization, branch, kind)
                elapsed = time.perf_counter()-tick
                difference = pressure-sample['reference']; difference -= difference.mean()
                reference = sample['reference']-sample['reference'].mean()
                rows.append(dict(**identity, policy=kind, **stats, seconds=elapsed,
                    pressure_relative_l2=float(torch.linalg.vector_norm(difference))/max(float(torch.linalg.vector_norm(reference)),1e-300)))
    if inventory != state.data.helpers().state_inventory(state.network) or torch.cuda.is_initialized():
        raise ValueError('Captured weights changed or CUDA initialized')
    summary = []
    for kind in ['numerical','learned']:
        group = [r for r in rows if r['policy']==kind]
        summary.append(dict(policy=kind,cases=len(group),qualified=sum(r['qualified'] for r in group),
            worst_relative_residual=max(r['relative_residual'] for r in group),
            worst_pressure_relative_l2=max(r['pressure_relative_l2'] for r in group),
            median_ms=1000*statistics.median(r['seconds'] for r in group),
            median_network_calls=statistics.median(r['network_calls'] for r in group)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, dict(status='completed', checkpoint=args.checkpoint,width=args.width,parameters=count,
        original_parameters=original_count,entry=state.entry,coarse_branch_entry=branch_entry,
        original_forward_parity=parity,original_recurrence_parity=recurrence,slices=slices,
        rows=rows,summary=summary,optimizer_updates=0,unchanged_original_inventory=True,
        source_sha256=sha(__file__),elapsed_s=time.perf_counter()-started,
        scope='Frozen transfer and deterministic structured pruning; failure after slicing does not measure fine-tuned reduced capacity.',
        timing_scope='One CPU thread, pinned core; exploratory concurrent measurements, not a deployment speed claim.'))
    print(args.checkpoint,args.width,summary,flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', choices=['C37','C44','C53'], required=True)
    parser.add_argument('--width', choices=['full','half'], required=True)
    parser.add_argument('--cpu', type=int, default=21)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--screen', action='store_true')
    evaluate(parser.parse_args())
