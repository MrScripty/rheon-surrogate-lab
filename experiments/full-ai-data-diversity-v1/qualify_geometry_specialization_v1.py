"""Finite GPU transport equivalence and geometry-specialization argument/graph custody."""
import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
from support import ROOT, read, write, digest
from debug.gather_fused_graph_v1 import GatherFusedGraph
from debug.geometry_specialized_graph_v1 import GeometrySpecializedGraph
from diagnose_causal_composition import inventory
from recurrent_sampler_native import dimensions


def run(args):
    if args.output.exists():
        raise ValueError('Fresh geometry-specialization component evidence required')
    reg = read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('GPU-only component qualification')
    for p, h in reg['source_sha256'].items():
        if digest(ROOT / p) != h:
            raise ValueError('Frozen component source changed: ' + p)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(reg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    baseline = GatherFusedGraph(reg['transport_pair'], reg['shared_pair'], reg['baseline_build'], reg['control_build'])
    candidate = GeometrySpecializedGraph(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = [inventory(m) for m in (baseline, candidate)]
    if before[0] != before[1]:
        raise ValueError('Identical restored learned tensors required')
    rows = []
    started = time.monotonic()
    result = dict(status='running', rows=rows, inference_device='cuda', display_label='GPU (CUDA)',
                  cpu_model_forward=False, optimizer_updates=0, latency_claim=False,
                  expanded_reserved52_references_opened=0, source_sha256=reg['source_sha256'],
                  protocol_sha256=digest(args.protocol))
    def compare(a, b):
        if len(a) != 4 or len(b) != 4:
            raise ValueError('All four carrier fields required')
        differences = []
        for x, y in zip(a, b):
            if x.dtype != y.dtype or x.shape != y.shape:
                raise ValueError('Exact native shape/dtype required')
            differences.append(int((x.contiguous().view(torch.int32) != y.contiguous().view(torch.int32)).sum()))
        if any(differences):
            raise ValueError('Geometry-specialization changed transport output bytes: ' + str(differences))
        return differences
    try:
        with torch.inference_mode():
            for case in reg['cases']:
                if time.monotonic() - started > reg['maximum_wall_seconds']:
                    raise TimeoutError('Finite geometry-specialization component deadline')
                scene = read(ROOT / case['receipt'])['scene']
                counts = tuple(scene['counts'])
                shapes = dimensions(counts)
                h, dt = 1 / scene['cells_per_metre'], 1 / (60 * case['reference_substeps'])
                for m in (baseline, candidate):
                    m.model.configure_transport(h, dt)
                spec = case['primitive_state']
                if digest(ROOT / spec['path']) != spec['sha256']:
                    raise ValueError('Native component input changed')
                with np.load(ROOT / spec['path'], allow_pickle=False) as archive:
                    actual = [torch.from_numpy(archive[n].copy()).to('cuda').reshape(s)
                              for n, s in zip(['vx', 'vy', 'vz', 'tracer'], shapes)]
                small = [torch.randn(s, dtype=torch.float32, device='cuda') * .01 for s in shapes]
                wide = [torch.randn(s, dtype=torch.float32, device='cuda') * 1e4 for s in shapes]
                signed_zero = [torch.zeros(s, dtype=torch.float32, device='cuda') for s in shapes]
                for v in signed_zero:
                    v.flatten()[::2] = -0.
                patterns = dict(actual=actual, arbitrary_small=small, arbitrary_wide=wide, signed_zero=signed_zero)
                for pattern, inputs in patterns.items():
                    expected = baseline.model.transport(inputs, counts, dt / h)
                    outputs = candidate.model.transport(inputs, counts, dt / h)
                    rows.append(dict(identifier=case['identifier'], mode='direct', pattern=pattern,
                                     different_bits=compare(outputs, expected)))
                # Geometry-specialization objects have left the Python stack by replay time.
                # Replay with new values in stable graph storage; allocations
                # for direct expected outputs deliberately alternate meanwhile.
                stable = [v.clone() for v in actual]
                for _ in range(2):
                    candidate.model.transport(stable, counts, dt / h)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_outputs = candidate.model.transport(stable, counts, dt / h)
                snapshot = None
                for pattern in ['actual', 'arbitrary_small', 'arbitrary_wide', 'actual']:
                    for destination, source in zip(stable, patterns[pattern]):
                        destination.copy_(source)
                    expected = baseline.model.transport(patterns[pattern], counts, dt / h)
                    graph.replay()
                    torch.cuda.synchronize()
                    copied = [v.clone() for v in graph_outputs]
                    if snapshot is None:
                        snapshot = copied
                    rows.append(dict(identifier=case['identifier'], mode='graph_replay', pattern=pattern,
                                     different_bits=compare(copied, expected)))
                compare(snapshot, graph_outputs)
                graph.reset()
                del graph, graph_outputs, copied, snapshot, stable, patterns, actual, small, wide, signed_zero, outputs, expected
                if torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']:
                    raise ValueError('Geometry-specialization component GPU memory ceiling')
                write(args.output / 'results.json', result)
                print('Grid geometry-specialization direct/replay bytes exact', case['identifier'], flush=True)
            if len(rows) != 208 or [inventory(m) for m in (baseline, candidate)] != before:
                raise ValueError('Complete population and unchanged learned tensors required')
        for p, h in reg['source_sha256'].items():
            if digest(ROOT / p) != h:
                raise ValueError('Component source changed during qualification: ' + p)
        result.update(status='passed_all26_geometry_specialized_transport_and_graph_custody',
                      compared_output_arrays=832, every_component_output_bitwise_equal=True,
                      geometry_specialization_stack_lifetime_and_dynamic_graph_replay_verified=True,
                      learned_weights_unchanged=True, gpu_model=torch.cuda.get_device_name(),
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      elapsed_diagnostic_seconds=time.monotonic() - started)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        baseline.close()
        candidate.close()
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
