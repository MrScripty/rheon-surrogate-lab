"""Audit complete refined-reference labels and an optional frozen endpoint control."""
import argparse
import os
from pathlib import Path
from support import ROOT, read, write, digest
from recurrent_sampler_native import dimensions
import numerics
import numpy as np
import torch


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--endpoint-control', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Fresh reference-label audit required')
    os.sched_setaffinity(0, {6})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    receipt = read(args.receipt)
    if receipt['status'] != 'completed' or receipt['actual_steps'] != 240:
        raise ValueError('Complete registered reference trajectory required')
    factor = receipt['reference_substeps']
    if receipt['actual_native_steps'] != 240*factor or len(receipt['frames']) != 240:
        raise ValueError('All actual fine reference advances must be counted')
    if any(len(frame['fine_frames']) != factor or not frame['frame']['advanced']
           or not all(f['frame']['advanced'] and f['frame']['quality']['qualified']
                      for f in frame['fine_frames']) for frame in receipt['frames']):
        raise ValueError('Every actual fine step must pass unchanged native quality gates')
    scene = receipt['scene']
    shape = (1, 1, *scene['counts'][::-1])
    sizes = [int(np.prod(s)) for s in dimensions(scene['counts'])]
    expected_sizes = dict(zip(('vx', 'vy', 'vz', 'tracer'), sizes))
    expected_sizes.update(pressure=sizes[-1], rhs=sizes[-1],
                          native_input_residual=sizes[-1], pressure_base=sizes[-1])
    h = torch.tensor([1/scene['cells_per_metre']], dtype=torch.float64)
    rho = torch.tensor([1000.], dtype=torch.float64)
    rows = []
    hashes = {str(args.receipt.resolve().relative_to(ROOT)): digest(args.receipt)}
    endpoint = None
    with torch.inference_mode():
        for spec in receipt['states']:
            archive_path = args.receipt.parent/spec['path']
            if digest(archive_path) != spec['sha256']:
                raise ValueError('Refined reference archive changed')
            with np.load(archive_path, allow_pickle=False) as archive:
                state = dict(archive)
            if set(state) != set(expected_sizes):
                raise ValueError('Complete physical fields and explicit auxiliary labels required')
            for key, value in state.items():
                dtype = '<f4' if key in ('vx', 'vy', 'vz', 'tracer') else '<f8'
                if value.dtype.str != dtype or value.size != expected_sizes[key] or not np.isfinite(value).all():
                    raise ValueError('Refined field type, shape or finiteness failed')
            pressure = torch.from_numpy(state['pressure'].reshape(shape))
            rhs = torch.from_numpy(state['rhs'].reshape(shape))
            delta = numerics.fine_action(pressure, h, rho)-rhs
            relative = float(torch.linalg.vector_norm(delta)/max(float(torch.linalg.vector_norm(rhs)), 1e-30))
            if relative > 1e-8:
                raise ValueError('Stored refined pressure and final fine RHS disagree')
            if spec['native_step'] != spec['step']*factor or spec['reference_dt_s'] != 1/(60*factor):
                raise ValueError('Stored label clock must match its actual native step')
            hashes[str(archive_path.resolve().relative_to(ROOT))] = digest(archive_path)
            rows.append(dict(output_step=spec['step'], native_step=spec['native_step'],
                             final_fine_pressure_equation_relative_l2=relative))
            if spec['step'] == 240:
                endpoint = state
    control_equal = None
    if args.endpoint_control:
        with np.load(args.endpoint_control, allow_pickle=False) as control:
            control_equal = all(endpoint[k].tobytes() == control[k].tobytes()
                                for k in ('pressure', 'vx', 'vy', 'vz', 'tracer'))
        if not control_equal:
            raise ValueError('Independent frozen sampled-control endpoint changed bytes')
        hashes[str(args.endpoint_control.resolve().relative_to(ROOT))] = digest(args.endpoint_control)
    hashes[str(Path(__file__).resolve().relative_to(ROOT))] = digest(__file__)
    write(args.output, dict(status='passed_refined_reference_label_audit',
        identifier=receipt['identifier'], stored_states=len(rows), output_steps=240,
        native_steps=240*factor, reference_substeps=factor, rows=rows,
        frozen_complete_endpoint_bitwise_equal=control_equal, source_sha256=hashes,
        model_targets_verified=False, temporal_convergence_qualified=False,
        optimizer_updates=0, reference_only=True))
    print('Verified', len(rows), 'complete refined states and', 240*factor,
          'actual internal steps; endpoint bitwise match:', control_equal)
