"""Bounded architecture admission, with retained checkpoint and consumed shapes.

No optimizer or simulation is constructed. CUDA mode checks deterministic
forward/backward only; CPU mode measures raw network forwards, separately from
the paired comparison worker's native diagnostic and physical reconstruction.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'water-residual-learning-v1'))
import audit


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) * 1024


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'artifacts', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    a = p.parse_args()
    if a.output.exists() or available() < 10 * 1024**3:
        raise ValueError('fresh report and 2 GiB accounting plus 8 GiB reserve required')
    started = time.perf_counter()
    audit.initialize(a.project.resolve(strict=True), a.artifacts.resolve(strict=True), sorted(os.sched_getaffinity(0)))
    torch = audit.TORCH
    sys.modules['base_model'] = audit.MODEL
    sys.path.insert(0, str(HERE / 'overlay'))
    spec = importlib.util.spec_from_file_location('spatial_coarse_prototype_model', HERE / 'overlay/model.py')
    variant = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(variant)
    variant.select(json.loads((a.project / 'config.json').read_text()))
    torch.manual_seed(20261015)
    candidate = variant.PressureCorrectionNet().eval()
    # The branch is the sole new state. Existing checkpoint tensors load exactly.
    missing, unexpected = candidate.load_state_dict(audit.NETWORK.state_dict(), strict=False)
    if missing != ['spatial_coarse.weight'] or unexpected:
        raise ValueError('unexpected prototype checkpoint mapping')
    for name, tensor in audit.NETWORK.state_dict().items():
        if not torch.equal(tensor, candidate.state_dict()[name]):
            raise ValueError('surviving checkpoint tensor mismatch')
    baseline = audit.NETWORK
    audit.MODEL.configure(a.device)
    baseline = baseline.to(a.device)
    candidate = candidate.to(a.device)
    rows = []
    for counts in ((16,16,16), (24,24,24), (32,32,32), (40,40,40), (33,11,9), (25,16,23), (1,3,7)):
        if available() < 8 * 1024**3:
            raise MemoryError('eight GiB reserve lost')
        shape = counts[::-1]
        axes = torch.meshgrid(*(torch.arange(n, device=a.device, dtype=torch.float32) + .5
                                for n in shape), indexing='ij')
        values = (torch.cos(torch.pi * axes[2] / shape[2]) +
                  .25 * torch.cos(3 * torch.pi * axes[1] / shape[1]))[None, None].contiguous()
        with torch.no_grad():
            expected = baseline(values)
            actual = candidate(values)
            exact = torch.equal(expected, actual)
        if not exact:
            raise ValueError('zero coarse addition changed retained checkpoint output')
        row = dict(counts_xyz=counts, exact_zero_branch_output=True)
        if a.device == 'cpu':
            clocks = {'baseline': [], 'candidate': []}
            with torch.no_grad():
                for network in (baseline, candidate):
                    for _ in range(3):
                        network(values)
                for repeat in range(20):
                    order = [('baseline', baseline), ('candidate', candidate)]
                    if repeat % 2:
                        order.reverse()
                    for name, network in order:
                        before = time.perf_counter()
                        network(values)
                        clocks[name].append(time.perf_counter() - before)
            row.update(forward_seconds=clocks,
                       median_overhead_fraction=statistics.median(clocks['candidate']) / statistics.median(clocks['baseline']) - 1)
        else:
            values.requires_grad_(True)
            outputs = []
            gradients = []
            for _ in range(2):
                candidate.zero_grad(set_to_none=True)
                values.grad = None
                output = candidate(values)
                output.square().mean().backward()
                torch.cuda.synchronize()
                outputs.append(output.detach().cpu())
                gradients.append({name: value.grad.detach().cpu().clone() for name, value in candidate.named_parameters()})
            if not torch.equal(outputs[0], outputs[1]) or any(
                    not torch.equal(gradients[0][name], gradients[1][name]) or
                    not torch.isfinite(gradients[0][name]).all() for name in gradients[0]):
                raise ValueError('nonfinite or nondeterministic CUDA gradients')
            if not gradients[0]['spatial_coarse.weight'].abs().max() > 0:
                raise ValueError('new coarse parameters receive no training gradient')
            row.update(deterministic_parameter_gradients=True, nonzero_coarse_gradient=True)
        rows.append(row)
    admitted = a.device == 'cuda' or all(r['median_overhead_fraction'] <= .25 for r in rows[:4])
    result = dict(format='pressure-spatial-coarse-prototype-v1', status='passed' if admitted else 'rejected_cost',
        device=a.device, rows=rows, baseline_parameters=sum(p.numel() for p in baseline.parameters()),
        candidate_parameters=sum(p.numel() for p in candidate.parameters()),
        constructed_active_modules={name: sum(type(m).__name__ == name for m in baseline.modules())
                                    for name in ('GroupNorm', 'SiLU', 'Identity')},
        weights_sha256=digest(a.artifacts/'weights.npz'), source_witness_sha256=digest(a.project/'source-witness.json'),
        sources={str(p):digest(p) for p in (Path(__file__), HERE/'overlay/model.py', HERE/'overlay/coarse.py')},
        cpu_affinity=sorted(os.sched_getaffinity(0)), cpu_threads=torch.get_num_threads(),
        cuda_initialized=torch.cuda.is_initialized(), optimizer_updates=0, simulations_advanced=0,
        elapsed_s=time.perf_counter()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        scope='Prototype architecture/gradient and standalone raw-network cost only. Same retained checkpoint plus zero coarse weights. No learned accuracy or trajectory/promotion claim.')
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','device','elapsed_s','baseline_parameters','candidate_parameters')}), flush=True)


if __name__ == '__main__':
    main()
