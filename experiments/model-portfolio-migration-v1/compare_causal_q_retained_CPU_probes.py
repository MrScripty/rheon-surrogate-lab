"""Compare retained original CPUprobe data to new GPUfields; no model runtime."""
import math
from pathlib import Path
import sys
import time

import numpy as np

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from component_material import observed_file


def run():
    output = DOC/'reports/causal-q-retained-CPU-probe-comparison-01.json'
    require(not output.exists(), 'Preserve original CPUprobe comparison')
    benchmark_path = DOC/'reports/causal-request-component-GPU-benchmark-01.json';benchmark = read(benchmark_path)
    material = read(ROOT/benchmark['material']['path']);entries = {row['fit_item_id']:row for row in material['models']}
    models = []
    for measured in benchmark['models']:
        fid = measured['fit_item_id'];entry = entries[fid]
        observed_file(ROOT,entry['original_receipt']['path'],entry['original_receipt'])
        receipt = read(ROOT/entry['original_receipt']['path']);probe_path = Path(entry['original_receipt']['path']).parent/'probe.npy'
        probe = observed_file(ROOT,probe_path,receipt['outputs']['probe.npy'])
        require(measured['weights'] == entry['outputs']['weights.npz'], 'Same original selected q weights required')
        manifest = read(ROOT/entry['inputs']['manifest']['path'])
        saved = np.load(ROOT/probe_path,allow_pickle=False)
        require(saved.dtype.str == '<f4' and saved.ndim == 1 and np.isfinite(saved).all(), 'Original saved CPUprobe layout required')
        rows = [];start = 0
        require(binding(ROOT/measured['prediction_archive']['path']) == measured['prediction_archive'], 'GPUfield archive changed')
        with np.load(ROOT/measured['prediction_archive']['path'],allow_pickle=False) as archive:
            for counts in ([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
                parent = next(index for index,row in enumerate(manifest['parents']) if row['scene']['counts'] == counts)
                size = math.prod(counts)
                for step in (0,8,120):
                    cpu = saved[start:start+size].astype('<f8');gpu = archive[f'case{parent*128+step:04d}'].reshape(-1).astype('<f8')
                    require(len(cpu) == len(gpu) == size, 'Exact original12probe field mapping required')
                    delta = gpu-cpu
                    rows.append(dict(parent_index=parent,current_state=step,counts_xyz=counts,
                        saved_CPU_l2=float(np.linalg.norm(cpu)),GPU_minus_saved_CPU_relative_l2=float(np.linalg.norm(delta))/max(float(np.linalg.norm(cpu)),1e-30),
                        GPU_minus_saved_CPU_absolute_max=float(np.abs(delta).max()),GPU_minus_saved_CPU_absolute_rms=float(np.sqrt(np.mean(delta**2)))))
                    start += size
        require(start == len(saved) and len(rows) == 12, 'All original CPUprobe data must be consumed exactly')
        models.append(dict(fit_item_id=fid,original_saved_CPU_probe=probe,rows=rows,
            maximum_GPU_minus_saved_CPU_relative_l2=max(row['GPU_minus_saved_CPU_relative_l2'] for row in rows),
            maximum_GPU_minus_saved_CPU_absolute=max(row['GPU_minus_saved_CPU_absolute_max'] for row in rows)))
    require('torch' not in sys.modules, 'Retained-data comparison imported a model runtime')
    save(output,dict(status='both_original_saved_CPU_q_probes_compared_without_model_execution',created_unix=time.time(),
        benchmark=binding(benchmark_path),models=models,fields_per_fit=12,CPU_neural_forwards=0,new_GPU_model_forwards=0,
        scope='Observed numerical differences onthe12originalCPUprobe cases perfit. No newCPU inference,no CPUtiming,nopredeclaredgeneral CPU/GPUequivalence tolerance orwholepopulation equivalence claim.',
        original_CPU_probes_regenerated=False,new_parameter_fits=0,producer=binding(Path(__file__))))
    for row in models:
        print(row['fit_item_id'],'maxGPU/savedCPUrelative delta',row['maximum_GPU_minus_saved_CPU_relative_l2'],flush=True)


if __name__ == '__main__':
    run()
