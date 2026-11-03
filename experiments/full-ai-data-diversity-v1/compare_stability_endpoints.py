"""Report timestep sensitivity on the exact native stability-repair scene."""
import argparse
import os
from pathlib import Path
from support import ROOT, write, read, digest
from reference_difference import difference
import numpy as np
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh reference difference report required')
    os.sched_setaffinity(0, {6})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    directory = ROOT/'runs/full-ai-data-diversity-v1/reference-substeps-stability-01'
    source = read(directory/'results.json')
    states = {}
    hashes = {}
    for trial in source['results']:
        factor = trial['substeps']
        if not trial['complete_240_macro_steps']:
            continue
        archive = directory/f'endpoint-substeps-{factor}.npz'
        if digest(archive) != trial['endpoint_sha256']:
            raise ValueError('Frozen reference endpoint changed')
        with np.load(archive, allow_pickle=False) as contents:
            states[factor] = dict(contents)
        hashes[str(archive.relative_to(ROOT))] = digest(archive)
    rows = [dict(coarser_substeps=a, finer_substeps=b,
                 fields=difference(states[a], states[b])) for a, b in ((2, 4), (4, 8), (2, 8))]
    write(args.output, dict(status='completed_descriptive_reference_endpoint_comparison',
        identifier=source['identifier'], physical_time_s=4., rows=rows,
        source_sha256={str((directory/'results.json').relative_to(ROOT)): digest(directory/'results.json'),
                       **hashes, str(Path(__file__).relative_to(ROOT)): digest(__file__),
                       'experiments/full-ai-data-diversity-v1/reference_difference.py':
                           digest(Path(__file__).with_name('reference_difference.py'))},
        temporal_convergence_qualified=False, model_targets_verified=False, optimizer_updates=0,
        interpretation='Stable completion alone does not establish timestep-independent reference truth. Pressure is the final internal projection pressure.'))
    for row in rows:
        print(row['coarser_substeps'], 'vs', row['finer_substeps'],
              {k: v['relative_linf_to_reference_peak'] for k, v in row['fields'].items()})
