"""Native-scored grouped learned transitions, with exact original N8 controls."""
import argparse
import collections
import hashlib
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from support import ROOT, read, write, digest
from grouped_macrostep_backbone_v1 import GroupedMacroBackbone
from cuda_native_transfer_v1 import FIELDS
from diagnose_causal_composition import inventory, measure


def hashes(fields):
    return {n: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n, v in fields.items()}


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only architecture diagnostic required')
    reg = read(args.protocol)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT / p) != h: raise ValueError('Frozen grouped diagnostic dependency changed: ' + p)
    cases = read(ROOT / reg['case_selection'])['cases']
    if len(cases) != 30 or sum(c['split'] == 'train' for c in cases) != 29 or any(c['reference_substeps'] != 8 for c in cases):
        raise ValueError('All29training and one already consumed N8development scene required')
    gold = read(ROOT / reg['original_controls'])
    identities = {(r['identifier'], r['start'], r['K'], r['lead']): r['output_sha256']
                  for r in gold['rows'] if r['mode'] == 'teacher_schedule_N8_control'}
    if len(identities) != 98: raise ValueError('All98original N8output identities required')
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = GroupedMacroBackbone(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = inventory(model); norm = read(ROOT / reg['normalization'])
    args.output.mkdir(parents=True); started = time.monotonic()
    result = dict(status='running', active=None, rows=[], summaries=[], source_evidence={},
                  inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
                  cpu_model_forward=False, optimizer_updates=0, new_model_trained=False, latency_claim=False,
                  full240history=False, new_independent_confirmation=False, expanded_reserved52_references_opened=0,
                  protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'])
    def save(): write(args.output / 'results.json', result)
    try:
        with torch.no_grad():
            for case in cases:
                if digest(ROOT / case['receipt']) != case['receipt_sha256']: raise ValueError('Consumed scene receipt changed')
                scene = read(ROOT / case['receipt'])['scene']; specs = {s['step']: s for s in case['states']}
                def load(step):
                    spec = specs[step]; p = ROOT / spec['path']
                    if digest(p) != spec['sha256']: raise ValueError('Native diagnostic field bytes changed')
                    result['source_evidence'][str(p.relative_to(ROOT))] = spec['sha256']
                    with np.load(p, allow_pickle=False) as a: return {n: a[n].copy() for n in (*FIELDS, 'rhs')}
                for start in case['diagnostic_starts']:
                    initial = load(start)
                    fields = {n: torch.from_numpy(initial[n]).to('cuda') for n in FIELDS}
                    original = hashes(fields)
                    prefixes = {name: [] for name in reg['schedules']}
                    for k in case['diagnostic_horizons']:
                        emitted = {}
                        for name, widths in reg['schedules'].items():
                            if time.monotonic() - started > reg['maximum_wall_seconds']: raise TimeoutError('Finite grouped architecture diagnostic deadline')
                            result['active'] = dict(identifier=case['identifier'], start=start, K=k, mode=name); save()
                            blocks = model.feature_states_grouped(fields, scene, start, widths, k)
                            outputs = [{n: v.detach().cpu().contiguous() for n, v in features[-1].items()} for features in blocks]
                            ids = [hashes(o) for o in outputs]
                            if len(outputs) != k or hashes(fields) != original or ids[:len(prefixes[name])] != prefixes[name]:
                                raise ValueError('Complete outputs, caller ownership and crossKcausal prefixes required')
                            prefixes[name] = ids
                            if name == 'original_N8_control':
                                for lead, identity in enumerate(ids, 1):
                                    if identity != identities[(case['identifier'], start, k, lead)]:
                                        raise ValueError('Grouped width1control changed original N8field bytes')
                            emitted[name] = (outputs, ids)
                        # Future native fields are loss/scoring labels only.
                        for lead in range(1, k + 1):
                            truth = load(start + lead)
                            for name, widths in reg['schedules'].items():
                                outputs, ids = emitted[name]; out = outputs[lead - 1]
                                if set(out) != set(FIELDS) or any(v.dtype != (torch.float64 if n == 'pressure' else torch.float32)
                                        or tuple(v.shape) != truth[n].shape for n, v in out.items()):
                                    raise ValueError('Complete correctly typed physical fields required')
                                nonfinite = {n: int((~v.isfinite()).sum()) for n, v in out.items()}
                                metrics = None; error = None
                                if not any(nonfinite.values()):
                                    try: metrics = measure(out, truth, scene, norm)
                                    except ValueError as e: error = repr(e)
                                result['rows'].append(dict(identifier=case['identifier'], split=case['split'], group=case['group'],
                                    counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'], start=start, K=k, lead=lead,
                                    mode=name, grouped_fine_intervals=widths, actual_internal_advances_per_block=k * len(widths),
                                    known_force_evaluations_per_block=k * 8, metrics=metrics, metric_error=error, nonfinite_counts=nonfinite,
                                    all_accuracy_pass=bool(metrics and metrics['all_below_1e5']), output_sha256=ids[lead - 1]))
                        del blocks, emitted, outputs
                    if (torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']
                            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > reg['maximum_peak_rss_bytes']):
                        raise ValueError('Grouped diagnostic memory ceiling exceeded')
                save(); print('Grouped learned transition', case['identifier'], 'all states scored and N8bytes exact', flush=True)
        if len(result['rows']) != reg['expected_complete_outputs'] or inventory(model) != before:
            raise ValueError('Full grouped population and unchanged restored parameters required')
        groups = collections.defaultdict(list)
        for row in result['rows']:
            groups[(row['mode'], row['split'], row['K'], row['cells_per_metre'], tuple(row['counts_xyz']))].append(row)
        result['summaries'] = [dict(mode=mode, split=split, K=k, cells_per_metre=cpm, counts_xyz=list(counts),
            complete_outputs=len(rows), failures=sum(not r['all_accuracy_pass'] for r in rows),
            worst_relative={m: max((r['metrics']['relative'][m] for r in rows if r['metrics']), default=None)
                            for m in ('equation', 'pressure', 'velocity', 'tracer')})
            for (mode, split, k, cpm, counts), rows in sorted(groups.items())]
        for p, h in reg['source_sha256'].items():
            if digest(ROOT / p) != h: raise ValueError('Grouped diagnostic dependency changed during execution')
        result.update(status='completed_grouped_macrostep_native_control_diagnostic', active=None,
                      original_N8_outputs_bitwise_exact=98, causal_prefix_and_input_ownership_pass=True,
                      learned_parameters_unchanged=True, every_registered_K_intermediate_scored=True,
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      elapsed_diagnostic_seconds=time.monotonic() - started, full_generator_target_pass=False)
    except BaseException as e:
        result.update(status='failed', error=repr(e)); raise
    finally:
        model.close(); save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
