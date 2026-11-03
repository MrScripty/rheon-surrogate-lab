"""Independent generic matrix, roundtrip, and changed-input CUDA graph controls."""
import argparse
import math
import os
from pathlib import Path
import resource
import time
import torch
from support import ROOT, read, write, digest
from fft_dct_v2 import FFTDCT


def matrix(value, bases, inverse):
    for axis in ((2, 1, 0) if inverse else (0, 1, 2)):
        work = value.movedim(axis, -1).contiguous()
        basis = bases[value.shape[axis]]
        value = torch.matmul(work, basis if inverse else basis.T).movedim(-1, axis).contiguous()
    return value


def error(actual, reference):
    difference = actual-reference
    return dict(relative_l2=float(difference.norm()/reference.norm().clamp_min(1.)),
                scaled_maximum=float(difference.abs().max()/reference.abs().max().clamp_min(1.)))


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh GPU-only generic transform controls required')
    reg = read(args.protocol)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT/p) != h:
            raise ValueError('Frozen FFT control dependency changed: '+p)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True); started = time.monotonic(); fft = FFTDCT()
    result = dict(status='running', active=None, rows=[], graph_rows=[], runtime=fft.identity,
                  inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
                  learned_model_forward=False, cpu_model_forward=False, optimizer_updates=0,
                  expanded_reserved52_references_opened=0, latency_claim=False,
                  source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    def save(): write(args.output/'results.json', result)
    def check(metrics):
        if not all(math.isfinite(v) and v <= reg['maximum_component_error'] for v in metrics.values()):
            raise ValueError('Generic FFT transform accuracy gate failed: '+str(metrics))
    try:
        with torch.inference_mode():
            for shape in reg['shapes_zyx']:
                shape = tuple(shape); result['active'] = dict(shape_zyx=list(shape)); save()
                fft.prepare(shape); bases = {}
                for n in set(shape):
                    j = torch.arange(n, device='cuda', dtype=torch.float64)
                    basis = torch.cos((math.pi/n)*j[:, None]*(j[None, :]+.5))*math.sqrt(2/n)
                    basis[0] = 1/math.sqrt(n); bases[n] = basis
                generator = torch.Generator(device='cuda').manual_seed(reg['seed']+sum((a+1)*n for a, n in enumerate(shape)))
                for kind in reg['populations']:
                    value = torch.zeros(shape, device='cuda', dtype=torch.float64)
                    if kind == 'constant': value.fill_(1.)
                    elif kind == 'impulse': value[tuple(n//2 for n in shape)] = 1.
                    elif kind == 'random': value = torch.randn(shape, generator=generator, device='cuda', dtype=torch.float64)
                    elif kind in ('modal', 'alternating'):
                        value.fill_(1.)
                        for axis, n in enumerate(shape):
                            at = torch.arange(n, device='cuda', dtype=torch.float64)
                            v = torch.cos((math.pi/n)*(at+.5)*min(axis+1, n-1)) if kind == 'modal' else 1.-2.*(at.to(torch.int64)%2)
                            view = [1]*3; view[axis] = n; value.mul_(v.reshape(view))
                    elif kind != 'zero': raise ValueError('Unknown frozen population')
                    transformed = fft.forward(value)
                    metrics = dict(forward_matrix=error(transformed, matrix(value, bases, False)),
                                   inverse_matrix=error(fft.inverse(value), matrix(value, bases, True)),
                                   roundtrip=error(fft.inverse(transformed), value))
                    result['rows'].append(dict(shape_zyx=list(shape), population=kind, metrics=metrics))
                    for m in metrics.values(): check(m)
                static = value.clone()
                stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3): fft.inverse(fft.forward(static))
                torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured_coefficients = fft.forward(static)
                    captured_reconstruction = fft.inverse(captured_coefficients)
                torch.cuda.synchronize()
                # A new random field after capture prevents a stale-output pass.
                changed = torch.randn(shape, generator=generator, device='cuda', dtype=torch.float64)
                static.copy_(changed); graph.replay(); torch.cuda.synchronize()
                metrics = dict(forward_matrix=error(captured_coefficients, matrix(changed, bases, False)),
                               changed_input_roundtrip=error(captured_reconstruction, changed))
                result['graph_rows'].append(dict(shape_zyx=list(shape), input_changed_after_capture=True, metrics=metrics))
                for m in metrics.values(): check(m)
                del graph, captured_coefficients, captured_reconstruction
                if time.monotonic()-started > reg['maximum_wall_seconds']:
                    raise TimeoutError('Finite FFT primitive deadline')
                if (torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']
                        or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > reg['maximum_peak_rss_bytes']):
                    raise ValueError('FFT primitive resource ceiling exceeded')
                save(); print('FFT double transform controls and changed-input graph', shape, 'passed', flush=True)
        for p, h in reg['source_sha256'].items():
            if digest(ROOT/p) != h: raise ValueError('Frozen FFT control dependency changed during execution')
        result.update(status='passed_generic_F64_FFT_DCT_matrix_roundtrip_and_changed_input_graph', active=None,
                      all_component_controls_pass=True, setup_observations=fft.setup_observations,
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      elapsed_diagnostic_seconds=time.monotonic()-started, full_generator_targets_verified=False)
    except BaseException as e:
        result.update(status='failed', error=repr(e)); raise
    finally: save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
