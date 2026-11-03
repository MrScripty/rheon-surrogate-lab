"""Qualify the complete parallel graph against serial bytes and tensor outputs."""
import argparse
from pathlib import Path
from profile_frozen_components import ROOT, read, write, digest
from minmax_sampler_core_support import prepare
from warm_pressure_sampler import load_model
from recurrent_sampler_noinline_v5 import library as serial
from recurrent_sampler_minmax_v12 import library as parallel
import recurrent_sampler_native as native
from benchmark_recurrent_sampler import output_identity
from diagnose_causal_composition import inventory, FIELDS
import numpy as np
import torch


def run(args):
    if args.output.exists():
        raise ValueError('Fresh complete parallel graph qualification required')
    reg = read(args.protocol)
    team = prepare(reg)
    model = load_model(reg['pair'], reg['library_path'])
    before = inventory(model)
    rows = []
    try:
        with torch.inference_mode():
            for trajectory in read(ROOT / reg['bank'])['trajectories']:
                path = ROOT / trajectory['receipt']
                if digest(path) != trajectory['receipt_sha256']:
                    raise ValueError('Current-state source receipt changed')
                receipt = read(path)
                for start in reg['starts']:
                    spec = receipt['states'][start]
                    state = path.parent / spec['path']
                    if digest(state) != spec['sha256']:
                        raise ValueError('Qualified input archive changed')
                    with np.load(state, allow_pickle=False) as archive:
                        fields = {key: archive[key].copy() for key in FIELDS}
                    native.library = serial
                    original = model.predict(fields, receipt['scene'], start, 4)
                    native.library = parallel
                    actual = model.predict(fields, receipt['scene'], start, 4)
                    if output_identity(original) != output_identity(actual):
                        raise ValueError('Parallel execution changed physical output bytes')
                    rows.append(dict(identifier=trajectory['identifier'], start=start,
                                     counts_xyz=receipt['scene']['counts'], complete_states=4,
                                     all_physical_output_bytes_exact=True))
                print(trajectory['identifier'], 'complete parallel graph qualified', flush=True)
        if inventory(model) != before or torch.cuda.is_initialized():
            raise ValueError('Qualification changed model or device')
        write(args.output, dict(status='passed_minmax_complete_graph_byte_execution',
              cores=reg['cores'], execution_team=team, rows=rows, complete_states=4 * len(rows),
              optimizer_updates=0, learned_tensors_unchanged=True,
              all_physical_output_bytes_exact=True, model_identity=model.asset_identity,
              source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
              cuda_initialized=False, all_plan_owners_closed=True,
              accuracy_qualified=False, latency_claim=False, fresh_holdout_consumed=False))
    finally:
        native.library = serial
        model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
