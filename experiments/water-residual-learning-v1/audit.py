"""CPU-only checkpoint diagnostics; analytic spectral controls remain labelled.

Owns independent per-state work, one thread/core per process. It never constructs
an optimizer, opens a CUDA context or advances a simulation trajectory.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import sys
import time

LAB=Path(__file__).resolve().parents[2]
BASES={}


def initialize(project,artifacts,cpus):
    global MODEL,DATA,TORCH,NETWORK,NORM
    identity=multiprocessing.current_process()._identity
    index=identity[0]-1 if identity else 0
    os.sched_setaffinity(0,{cpus[index%len(cpus)]})
    sys.path.insert(0,str(project))
    import torch
    import model
    import data
    import policy
    TORCH=torch;MODEL=model;DATA=data
    model.configure('cpu');torch.set_num_threads(1)
    witness_raw=policy.read(Path(project)/'source-witness.json');witness=policy.decode(witness_raw)
    for name,digest in witness['sources'].items():assert policy.sha(policy.read(Path(project)/name))==digest
    config_raw=policy.read(Path(project)/'config.json');config=policy.decode(config_raw);model.select(config)
    artifacts=Path(artifacts)
    receipt=policy.decode(policy.read(artifacts/'receipt.json'));history=policy.decode(policy.read(artifacts/'history.json'))
    assert receipt['mode']=='restore' and receipt['optimizer_updates']==0
    assert receipt['execution']['cuda_initialized'] is False and receipt['architecture']==model.ARCHITECTURE
    assert receipt['closure']['source_witness_sha256']==policy.sha(witness_raw)
    assert receipt['closure']['experiment_sha256']==policy.sha(config_raw)
    for name in ('weights.npz','normalization.json','history.json'):
        assert policy.sha(policy.read(artifacts/name))==receipt['outputs'][name]['sha256']
    NORM=policy.decode(policy.read(artifacts/'normalization.json'))
    final=f'checkpoint-{config["training"]["updates"]:03d}.npz'
    NETWORK=model.PressureCorrectionNet().eval()
    data.helpers().load_weights(NETWORK,artifacts/'weights.npz',history['checkpoints'][final]['inventory'])
    assert not torch.cuda.is_initialized()


def vector_norm(value):return float(TORCH.linalg.vector_norm(value))


def ratio(pressure,sample):
    error=MODEL.full_operator(pressure,sample['cell_size_m'],sample['rho'])-sample['rhs']
    return vector_norm(error)/max(vector_norm(sample['rhs']),1e-30)


def cosine_transform(values,bases,inverse=False):
    """Contract one axis at a time, keeping intermediates the size of the grid.

    A four-operand einsum may form an n^6 outer product without opt_einsum.
    Explicit two-operand contractions keep memory independent of that optional
    optimizer and preserve the same orthonormal separable transform.
    """
    torch=TORCH
    a,b,c=bases
    if inverse:
        values=torch.einsum('az,abc->zbc',a,values)
        values=torch.einsum('by,zbc->zyc',b,values)
        return torch.einsum('cx,zyc->zyx',c,values)
    values=torch.einsum('az,zyx->ayx',a,values)
    values=torch.einsum('by,ayx->abx',b,values)
    return torch.einsum('cx,abx->abc',c,values)


def spectral_correction(residual,sample,modes=None):
    """Analytic pseudoinverse of sealed L, optionally truncated to low modes.

    Independent cosine eigenvectors: lambda_k=2-2cos(pi*k/n), k=0..n-1.
    Constant mode is excluded; returned correction has cell-zero gauge.
    """
    torch=TORCH;shape=residual.shape[2:]
    key=shape
    if key not in BASES:
        bases=[];eigenvalues=[]
        for n in shape:
            k=torch.arange(n,dtype=torch.float64);j=k+.5
            matrix=torch.cos(torch.pi*k[:,None]*j[None,:]/n)*(2/n)**.5
            matrix[0]/=2**.5
            bases.append(matrix);eigenvalues.append(2-2*torch.cos(torch.pi*k/n))
        eigen=eigenvalues[0][:,None,None]+eigenvalues[1][None,:,None]+eigenvalues[2][None,None,:]
        eigen[0,0,0]=1
        BASES[key]=(bases,eigen)
    (a,b,c),eigen=BASES[key]
    q=residual[0,0]*(sample['rho']/sample['cell_size_m']).item()
    coefficients=cosine_transform(q,(a,b,c))/eigen
    coefficients[0,0,0]=0
    if modes is not None:
        coefficients[modes:]=0;coefficients[:,modes:]=0;coefficients[:,:,modes:]=0
    delta=cosine_transform(coefficients,(a,b,c),inverse=True)[None,None]
    return delta-delta[:,:,:1,:1,:1]


def evaluate(task):
    parent,slot=task;torch=TORCH;sample=DATA.sample(parent,slot);base=sample['base']
    with torch.no_grad():
        start=time.perf_counter();prediction=MODEL.predict_pressure(NETWORK,sample,NORM);inference=time.perf_counter()-start
        correction=prediction-base
        amplitudes=[]
        for gain in (-2.,-1.,.25,.5,2.,4.):
            scaled={**sample,'native_input_residual':gain*sample['native_input_residual']}
            delta=MODEL.predict_pressure(NETWORK,scaled,NORM)-base
            amplitudes.append({'gain':gain,'linearity_error':vector_norm(delta-gain*correction)/max(vector_norm(gain*correction),1e-30)})
        other=torch.flip(sample['native_input_residual'],[2,4])
        def delta(residual):return MODEL.predict_pressure(NETWORK,{**sample,'native_input_residual':residual},NORM)-base
        superposition=vector_norm(delta(sample['native_input_residual']+other)-correction-delta(other))/max(vector_norm(correction)+vector_norm(delta(other)),1e-30)
        residual=sample['rhs']-MODEL.full_operator(prediction,sample['cell_size_m'],sample['rho'])
        spectral=[]
        for modes in (2,4,8,None):
            start=time.perf_counter();extra=spectral_correction(residual,sample,modes)
            proposed=prediction+extra
            spectral.append({'modes_per_axis':modes,'relative_residual':ratio(proposed,sample),
                'elapsed_s':time.perf_counter()-start,'kind':'analytic-f64-spectral-correction-not-AI'})
        scale=NORM['scales']['pressure']['value']
        ideal=((sample['reference']-base)/scale).float()
        rounded=base+MODEL.frozen.gauge_correction(ideal,scale)
        iterations=[];current=base.clone()
        for _ in range(8):
            state={**sample,'base':current,'native_input_residual':sample['rhs']-MODEL.full_operator(current,sample['cell_size_m'],sample['rho'])}
            current=MODEL.predict_pressure(NETWORK,state,NORM);iterations.append(ratio(current,sample))
        error=prediction-sample['reference'];gradient_squared=reference_gradient_squared=0.
        for axis in (2,3,4):
            gradient_squared+=float(torch.diff(error,dim=axis).square().sum())
            reference_gradient_squared+=float(torch.diff(sample['reference'],dim=axis).square().sum())
    return {'trajectory_id':parent['entry']['trajectory_id'],'native_frame':parent['frames'][slot]['index'],
        'split':parent['entry']['split'],'counts_xyz':parent['counts_xyz'],'cell_size_m':parent['cell_size_m'],
        'relative_residual':ratio(prediction,sample),'relative_pressure_error':vector_norm(error)/max(vector_norm(sample['reference']),1e-30),
        'relative_pressure_gradient_error':(gradient_squared/max(reference_gradient_squared,1e-30))**.5,
        'amplitude':amplitudes,'superposition_error':superposition,'repeated_relative_residual':iterations,
        'spectral_controls':spectral,'rounded_reference_relative_residual':ratio(rounded,sample),
        'inference_s':inference,'pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0)),
        'threads':torch.get_num_threads(),'cuda_initialized':torch.cuda.is_initialized()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','artifacts','packet','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--training-only',action='store_true');parser.add_argument('--workers',type=int)
    args=parser.parse_args();sys.path.insert(0,str(args.project.absolute()))
    import data
    _reservation,parents=data.read_view(args.packet.absolute())
    tasks=[(p,i) for p in parents if not args.training_only or p['entry']['split']=='train' for i in range(len(p['frames']))]
    cpus=sorted(os.sched_getaffinity(0));cpus=cpus[1:] or cpus
    workers=min(args.workers or len(cpus),len(cpus),len(tasks))
    args.output.mkdir(parents=True,exist_ok=False);records=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,
        initargs=(args.project.absolute(),args.artifacts.absolute(),cpus)) as pool:
        for record in pool.map(evaluate,tasks):
            records.append(record)
            with (args.output/'cases.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
    summary={'schema':'rheon-residual-cpu-audit-v1','scope':'recorded teacher states; no autonomous rollout; analytic spectral controls separately labelled',
        'cases':len(records),'workers':workers,'used_cpus':sorted({c for r in records for c in r['affinity']}),
        'cuda_initialized':any(r['cuda_initialized'] for r in records),'elapsed_s':time.perf_counter()-start,
        'median_relative_residual':statistics.median(r['relative_residual'] for r in records),
        'max_relative_residual':max(r['relative_residual'] for r in records),
        'median_superposition_error':statistics.median(r['superposition_error'] for r in records),
        'median_repeated_8':statistics.median(r['repeated_relative_residual'][-1] for r in records),
        'spectral_control_medians':{str(m):statistics.median(next(s['relative_residual'] for s in r['spectral_controls'] if s['modes_per_axis']==m) for r in records) for m in (2,4,8,None)},
        'rounded_reference_equation_passed':sum(r['rounded_reference_relative_residual']<=1e-6 for r in records),
        'physical_qualification':'not measured by this diagnostic; use native_parallel.py'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
