"""Every-output GPU replay against already native-scored full240field hashes.

N1uses the original learned transition and its original native-scored fields;
N8uses the direct transition and its successful uninterrupted native240fields.
This is byte inheritance of existing numerical evidence, not a fresh native
teacher run and not independent confirmation. References never enter inference.
"""
import argparse
import hashlib
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from support import ROOT, read, write, digest
from nested_F64_pressure_graph_v1 import NestedF64PressureGraph as AbsolutePressureGraph
from cuda_native_transfer_v1 import FIELDS, download_outputs
from diagnose_causal_composition import inventory


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda': raise ValueError('Fresh GPU-only history proof required')
    reg = read(args.protocol)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT/p) != h: raise ValueError('Frozen full-history dependency changed: '+p)
    gate = read(ROOT/reg['block_qualification'])
    if not gate['all_accuracy_pass'] or gate['complete_outputs_exact_parent']!=1018 or not gate['N1_all_output_bits_unchanged']: raise ValueError('Actual fixed-regime block proof required')
    if len(reg['cases']) != 26 or sum(c['split']=='development' for c in reg['cases'])!=24 or sum(c['split']=='train' for c in reg['cases'])!=2: raise ValueError('Full24development plus2alreadyconsumedhighvolumetrainingpopulation required')
    os.sched_setaffinity(0, {reg['host_cpu']}); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True); started = time.monotonic()
    model = AbsolutePressureGraph(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = inventory(model)
    result = dict(status='running', active=None, cases=[], rows=[], exact_complete_outputs=0,
                  inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
                  cpu_model_forward=False, optimizer_updates=0, expanded_reserved52_references_opened=0,
                  fresh_native_capture=False, independent_confirmation=False, latency_claim=False,
                  protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'])
    def save(): write(args.output/'results.json', result)
    try:
        with torch.inference_mode():
            for case in reg['cases']:
                name = case['identifier']; native_path = ROOT/reg['native_scored_history_by_identifier'][name]
                gold = read(native_path)
                if gold['status'] != 'completed_streamed_expanded_gpu_240step_scene' or not gold['all_accuracy_pass'] or len(gold['rows']) != 960:
                    raise ValueError('Complete successful native-scored source history required')
                process = gold['native_process']
                if process['returncode'] != 0 or not process['reaped'] or not process['graceful_protocol_complete']:
                    raise ValueError('Source native process must be terminal and drained')
                expected = {(r['K'], r['step']): r['output_sha256'] for r in gold['rows']}
                if digest(ROOT/case['receipt']) != case['receipt_sha256']: raise ValueError('Scene receipt changed')
                scene = read(ROOT/case['receipt'])['scene']; n = case['reference_substeps']
                spec = next(s for s in case['states'] if s['step'] == 0)
                if digest(ROOT/spec['path']) != spec['sha256']: raise ValueError('Actual native initial field bytes changed')
                with np.load(ROOT/spec['path'], allow_pickle=False) as a:
                    initial = {f: torch.from_numpy(a[f].copy()).to('cuda') for f in FIELDS}
                model.prepare_geometry(tuple(scene['counts']))
                for k in (1, 2, 3, 4):
                    model.prepare_block(initial, scene, 0, k, n); current = initial
                    for start in range(0, 240, k):
                        result['active'] = dict(identifier=name, K=k, start=start)
                        outputs = model.predict(current, scene, start, k, n)
                        if len(outputs) != k: raise ValueError('Every physical intermediate required')
                        current = outputs[-1]
                        downloaded = download_outputs(outputs)
                        for lead, output in enumerate(downloaded, 1):
                            identities = {f: hashlib.sha256(v.numpy().tobytes()).hexdigest() for f, v in output.items()}
                            if identities != expected[(k, start+lead)]:
                                raise ValueError('Fixed-regime output differs from native-scored source: '+str((name, k, start+lead)))
                            result['rows'].append(dict(identifier=name, K=k, step=start+lead, all_field_bits_exact=True, output_sha256=identities))
                            result['exact_complete_outputs'] += 1
                        if time.monotonic()-started > reg['maximum_wall_seconds']: raise TimeoutError('Finite every-output history proof deadline')
                    save()
                result['cases'].append(dict(identifier=name, counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                                            reference_substeps=n, complete_outputs_exact=960, native_scored_source=str(native_path.relative_to(ROOT)),
                                            native_scored_source_sha256=digest(native_path), inherited_worst_error=gold['worst_error']))
                model.release_graphs()
                if (torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']
                        or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > reg['maximum_peak_rss_bytes']): raise ValueError('Full-history proof resource ceiling exceeded')
                save(); print('NestedF64full240GPUhashproof', name, '960completeoutputs exact', flush=True)
        if result['exact_complete_outputs'] != 24960 or len(result['cases']) != 26 or inventory(model) != before:
            raise ValueError('All24/K1-4/240physicaloutputs and unchanged model assets required')
        for p, h in reg['source_sha256'].items():
            if digest(ROOT/p) != h: raise ValueError('History proof dependency changed during execution')
        result.update(status='passed_all26_nested_F64_full240_native_scored_hash_inheritance', active=None,
                      inherited_worst_error={m: max(c['inherited_worst_error'][m] for c in result['cases']) for m in ('equation', 'pressure', 'velocity', 'tracer')},
                      frozen_assets_unchanged=True, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      elapsed_diagnostic_seconds=time.monotonic()-started, full_generator_target_pass=False)
    except BaseException as e:
        result.update(status='failed', error=repr(e)); raise
    finally:
        model.close(); save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--protocol', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
