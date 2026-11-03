"""Verify and reconstruct independently reserved native pressure requests."""
from pathlib import Path
import numpy as np
from study_support import ROOT, read, sha


def cases(bank_path, torch, timing_only=False):
    bank = read(bank_path)
    if bank['status'] != 'completed' or bank['temporal_reserved_scenes_consumed']:
        raise ValueError('Completed independent Track A capture required')
    result = []
    for row in bank['trajectories']:
        receipt_path = ROOT / row['receipt']
        if sha(receipt_path) != row['receipt_sha256']:
            raise ValueError('Fresh native capture receipt changed')
        receipt = read(receipt_path)
        scene = receipt['scene']
        shape = (1, 1, *scene['counts'][::-1])
        for record in receipt['cases']:
            if timing_only and record['step'] not in (1, 16, 32, 64):
                continue
            source = receipt_path.parent / record['path']
            if sha(source) != record['sha256']:
                raise ValueError('Fresh pressure request changed')
            with np.load(source, allow_pickle=False) as arrays:
                sample = {name: torch.from_numpy(arrays[name].copy().reshape(shape))
                          for name in ('base', 'native_input_residual', 'rhs', 'reference')}
            if any(not bool(torch.isfinite(value).all()) for value in sample.values()):
                raise ValueError('Nonfinite captured pressure request')
            sample.update(cell_size_m=torch.tensor([1 / scene['cells_per_metre']], dtype=torch.float64),
                          rho=torch.tensor([1000.], dtype=torch.float64),
                          dt=torch.tensor([1 / 60], dtype=torch.float64))
            identity = dict(scene=row['identifier'], step=record['step'], time_s=record['time_s'],
                            counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                            population='small' if max(scene['counts']) <= 16 else '32cubed_extension')
            result.append((identity, sample))
    if len(result) != (64 if timing_only else 1024):
        raise ValueError('Every reserved scene and declared request must be retained')
    return result
