"""Export exact deployed sampler parameters for a future compiler specialization.

Artifact inventory only: no model forward, coefficient fitting or native data.
"""
import argparse
import hashlib
import os
from pathlib import Path
import numpy as np
import torch
from support import ROOT, read, write, digest
from nested_F64_pressure_graph_v1 import NestedF64PressureGraph
from diagnose_causal_composition import inventory


def tensor_record(value):
    array = value.detach().cpu().numpy().copy()
    if array.dtype.str != '<f8' or not np.isfinite(array).all(): raise ValueError('Exact finite deployed F64parameter tensor required')
    return dict(dtype=array.dtype.str, shape=list(array.shape), sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                values_hex=[float(v).hex() for v in array.flatten()], bits_u64=[int(v) for v in array.view('<u8').flatten()],
                elements=array.size, nonzero=int(np.count_nonzero(array)), zeros=int(np.count_nonzero(array == 0.)),
                exact_positive_one=int(np.count_nonzero(array == 1.)), exact_negative_one=int(np.count_nonzero(array == -1.)))


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda': raise ValueError('Fresh qualified artifact inventory required')
    reg = read(args.protocol)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT/p) != h: raise ValueError('Frozen manifest dependency changed: '+p)
    torch.set_num_threads(1); torch.empty(0, device='cuda')
    model = NestedF64PressureGraph(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = inventory(model)
    try:
        core = model.model
        rows, channels, offsets, weights = core.sparse['point']
        point = dict(rows=rows, channels=channels, offsets=offsets, coefficients=tensor_record(weights),
                     head_nonzero_counts=[rows[i+1]-rows[i] for i in range(len(rows)-1)])
        tensors = dict(point=point, mid_gate=tensor_record(core.mid_gate), mix=tensor_record(core.mix))
        if inventory(model) != before: raise ValueError('Manifest export changed learned tensors')
        for p,h in reg['source_sha256'].items():
            if digest(ROOT/p) != h: raise ValueError('Manifest dependency changed during export')
        args.output.mkdir(parents=True)
        write(args.output/'manifest.json', dict(status='exported_exact_deployed_sampler_parameter_inventory', tensors=tensors,
              parent_parameter_provenance=core.asset_identity, source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
              parameter_binding='All literals/metadata derive only from exact managed-restored and uniformly deployed learned tensors. Future codegen must verify the same deployed hashes, not assign stencil coefficients or infer a pressure inverse.',
              inference_device='cuda', display_label='GPU (CUDA)', cpu_model_forward=False, learned_model_forward_executed=False,
              optimizer_updates=0, native_fields_or_RHS_used=False, expanded_reserved52_references_opened=0,
              new_candidate_compiled=False, latency_claim=False, frozen_assets_unchanged=True))
        print('Exactlearnedsamplerinventory', 'pointCSR', point['coefficients']['elements'], 'gates', tensors['mid_gate']['nonzero'], 'nonzeroMix', tensors['mix']['nonzero'], flush=True)
    finally:
        model.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
