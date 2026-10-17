"""Bounded spatial frequency/amplitude audit of retained, consumed sparse states.

One state per task; aggregate cells into finite bins rather than retaining a
mode-by-state tensor. No simulation, optimizer, model change or holdout access.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'water-residual-learning-v1'))
import audit


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def available():return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024


def initialize(cpus):
    os.sched_setaffinity(0,{cpus[(multiprocessing.current_process()._identity[0]-1)%len(cpus)]})
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);audit.TORCH=torch


def bins(prediction, exact, base, bases, spacing):
    torch=audit.TORCH;shape=prediction.shape[-3:];cells=np.prod(shape)
    target=exact-base;centered=target-target.mean();scale=max(float(centered.square().mean().sqrt()),1e-30)
    signal=audit.cosine_transform(target[0,0],bases).numpy()/np.sqrt(cells)
    error=audit.cosine_transform((prediction-exact)[0,0],bases).numpy()/np.sqrt(cells)
    indices=np.indices(shape);fundamental=1/(2*max(shape)*spacing)
    frequency=np.sqrt(sum((indices[a]/(2*n*spacing))**2 for a,n in enumerate(shape)))
    amplitude=np.abs(signal)/scale
    frequency_bin=np.searchsorted([2.,4.,8.,16.],frequency/fundamental,side='right')
    amplitude_bin=np.searchsorted([1e-8,1e-6,1e-4,1e-2],amplitude,side='right')
    low=np.all(indices<=3,axis=0);mask=np.ones(shape,dtype=bool);mask[0,0,0]=False
    records=[]
    for band in range(5):
        for strength in range(5):
            for is_low in (False,True):
                selected=mask&(frequency_bin==band)&(amplitude_bin==strength)&(low==is_low)
                if not selected.any():continue
                energy=float(np.square(signal[selected]).sum());error_energy=float(np.square(error[selected]).sum())
                records.append(dict(frequency_bin=band,relative_amplitude_bin=strength,training_low_block=is_low,
                    modes=int(selected.sum()),signal_energy_pa2=energy,error_energy_pa2=error_energy,
                    relative_rms_error=(error_energy/energy)**.5 if energy else None))
    return records


def evaluate(task):
    if available()<8*1024**3:raise MemoryError('spectral audit host reserve reached')
    path,identity,scene,step=task;path=Path(path);assert digest(path)==identity
    torch=audit.TORCH
    with np.load(path,allow_pickle=False) as archive:
        sample={k:torch.from_numpy(archive[k].copy()) for k in archive.files}
    predicted=sample.pop('prediction');assert predicted.numel()<=65536
    with torch.no_grad():
        delta=audit.spectral_correction(sample['native_input_residual'],sample)
        exact=sample['base']+delta;bases,_=audit.BASES[tuple(predicted.shape[-3:])]
        rows=bins(predicted,exact,sample['base'],bases,float(sample['cell_size_m'][0]))
    assert not torch.cuda.is_initialized()
    return dict(scene=scene,step=step,capture_sha256=identity,counts_xyz=list(predicted.shape[-3:][::-1]),
        cell_size_m=float(sample['cell_size_m'][0]),rows=rows,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--exclude-cpu',type=int,action='append',default=[])
    a=p.parse_args();assert not a.output.exists();cpus=[c for c in sorted(os.sched_getaffinity(0))[2:] if c not in a.exclude_cpu]
    assert 1<=a.workers<=len(cpus) and available()>8*1024**3+a.workers*1024**3
    campaign=json.loads((a.campaign/'receipt.json').read_text());assert campaign['status']=='passed' and len(campaign['records'])==74
    tasks=[]
    for path in sorted(a.campaign.glob('case-*/episode.json')):
        episode=json.loads(path.read_text());assert episode['status']=='passed'
        for capture in episode['captures']:tasks.append((capture['path'],capture['sha256'],episode['metrics']['scene'],capture['step']))
    assert len(tasks)==233;start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(cpus[:a.workers],)) as pool:
        records=list(pool.map(evaluate,tasks))
    result=dict(format='rheon_spatial_frequency_amplitude_audit_v1',scope='Consumed autonomous states; amplitude correlations are descriptive, not a causal noise-floor measurement or temporal Hz spectrum.',
        campaign_sha256=digest(a.campaign/'receipt.json'),sources={str(path):digest(path) for path in (Path(__file__),Path(audit.__file__))},
        frequency_bin_edges_in_fundamentals=[0,2,4,8,16,None],relative_amplitude_bin_edges=[0,1e-8,1e-6,1e-4,1e-2,None],
        amplitude_units='orthonormal cosine coefficient divided by sqrt(cells), relative to mean-free correction RMS',
        training_low_block='all per-axis indices <=3; DC excluded',workers=a.workers,threads_per_worker=1,cpu_affinity=cpus[:a.workers],
        optimizer_updates=0,simulations_advanced=0,cuda_initialized=False,elapsed_s=time.perf_counter()-start,records=records)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(states=len(records),elapsed_s=result['elapsed_s'])))

if __name__=='__main__':main()
