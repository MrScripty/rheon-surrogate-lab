"""Separate temporal signal/error power and controlled spatial mode amplitude.

CPU diagnostics on consumed teacher states and labelled manufactured fields;
never training data selection or an independent rollout qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np

LAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/water-residual-learning-v1'))
import audit
sys.path.insert(0,str(LAB/'experiments/water-stability-learning-v1'))
from learned_iteration import predict_proposal


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def temporal_power(values,dt,length=32,hop=8):
    """Periodic Hann, cell-wise demean, one-sided PSD, mean over cells."""
    values=np.asarray(values,dtype=np.float64).reshape(len(values),-1)
    window=.5-.5*np.cos(2*np.pi*np.arange(length)/length)
    starts=list(range(0,len(values)-length+1,hop))
    result=[]
    for start in starts:
        block=values[start:start+length]
        spectrum=np.fft.rfft((block-block.mean(axis=0))*window[:,None],axis=0)
        power=(np.abs(spectrum)**2).mean(axis=1)*dt/(window**2).sum()
        power[1:-1]*=2
        result.append(power)
    return starts,np.fft.rfftfreq(length,dt),np.asarray(result)


def temporal(config):
    torch=audit.TORCH
    rows=[];sources=[]
    for folder in sorted((LAB/config['teacher_root']).glob('parent-*')):
        receipt=json.loads((folder/'receipt.json').read_text())
        assert receipt['status']=='qualified_cpu_teacher_parent'
        path=folder/'fields.npz'
        assert digest(path)==receipt['fields_sha256']
        frames=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
        steps=[r['step'] for r in receipt['samples']]
        assert steps==list(range(16,65))
        times=[next(f['time_s'] for f in frames if f['step']==step) for step in steps]
        dt=float(np.median(np.diff(times)))
        assert np.allclose(np.diff(times),dt,rtol=1e-10,atol=1e-12)
        errors={n:[] for n in config['temporal_calls']};references=[]
        with np.load(path,allow_pickle=False) as fields,torch.no_grad():
            for step in steps:
                base=torch.from_numpy(fields[f'{step:03d}-base'].copy())[None,None]
                residual=torch.from_numpy(fields[f'{step:03d}-residual'].copy())[None,None]
                reference=fields[f'{step:03d}-pressure'].copy()
                references.append(reference-reference.mean())
                sample={'base':base,'native_input_residual':residual,
                    'cell_size_m':torch.tensor([1/receipt['scene']['cells_per_metre']],dtype=torch.float64),
                    'rho':torch.tensor([1000.],dtype=torch.float64),
                    'dt':torch.tensor([dt],dtype=torch.float64)}
                pressure=base.clone()
                for calls in range(1,max(errors)+1):
                    state={**sample,'base':pressure,'native_input_residual':residual-audit.MODEL.full_operator(
                        pressure-base,sample['cell_size_m'],sample['rho'])}
                    pressure=predict_proposal(audit.MODEL,audit.NETWORK,state,audit.NORM,True)
                    if calls in errors:
                        error=pressure[0,0].numpy()-reference
                        errors[calls].append(error-error.mean())
        starts,frequencies,signal=temporal_power(references,dt)
        for calls,values in errors.items():
            _,_,error=temporal_power(values,dt)
            for w,start in enumerate(starts):
                for k,hz in enumerate(frequencies):
                    if k==0:continue  # Detrended DC is not a physical temporal mode.
                    s,e=float(signal[w,k]),float(error[w,k])
                    rows.append({'scene':folder.name,'calls':calls,'window_start':start,
                        'frequency_hz':float(hz),'signal_power_pa2_per_hz':s,'error_power_pa2_per_hz':e,
                        'error_over_signal_power':e/s if s>0 else None,
                        'relative_rms_error':float(np.sqrt(e/s)) if s>0 else None})
        sources.append({'fields':str(path.relative_to(LAB)),'fields_sha256':digest(path),
                        'receipt_sha256':digest(folder/'receipt.json'),'dt_s':dt,'steps':steps})
        print('temporal',folder.name,flush=True)
    return {'method':'Temporal PSD, Pa²/Hz, periodic Hann32, hop8, per-cell temporal mean removed; spatial pressure gauge removed per frame. Signal and error undergo identical transforms. Ratio of mean cell power is not phase/coherence or a causal noise-floor estimate.',
            'sources':sources,'rows':rows}


def spatial(config):
    torch=audit.TORCH
    rows=[]
    for counts in config['spatial_counts']:
        shape=counts[::-1]
        long_axis=int(np.argmax(shape))
        frequencies=[0,0,0];frequencies[long_axis]=1
        high=[0,0,0];high[long_axis]=min(7,shape[long_axis]-1)
        def mode(freq):
            factors=[torch.cos(torch.pi*k*(torch.arange(n,dtype=torch.float64)+.5)/n) for n,k in zip(shape,freq)]
            return torch.einsum('z,y,x->zyx',*factors)[None,None]
        low_wave,high_wave=mode(frequencies),mode(high)
        norm=float(low_wave.square().sum())
        scalar={k:torch.tensor([v],dtype=torch.float64) for k,v in [('cell_size_m',.125),('rho',1000.),('dt',1/60)]}
        def coefficient(value):return float((value*low_wave).sum())/norm
        with torch.no_grad():
            # High-only identifies cross-band leakage into the low mode.
            for kind in ('pure-low','low-plus-high','high-only'):
                for alpha in (config['amplitudes'] if kind!='high-only' else [0.]):
                    exact=alpha*low_wave+(high_wave if kind!='pure-low' else 0.)
                    exact=exact-exact.flatten()[0]
                    rhs=audit.MODEL.full_operator(exact,scalar['cell_size_m'],scalar['rho'])
                    # RHS uses shared operator; verify its analytical cosine eigenvalue independently.
                    eigen_low=sum(2-2*np.cos(np.pi*k/n) for k,n in zip(frequencies,shape))
                    eigen_high=sum(2-2*np.cos(np.pi*k/n) for k,n in zip(high,shape))
                    expected=(alpha*eigen_low*low_wave+(eigen_high*high_wave if kind!='pure-low' else 0.))*.125/1000
                    defect=audit.vector_norm(rhs-expected)/max(audit.vector_norm(expected),1e-300)
                    assert defect<1e-10
                    pressure=torch.zeros_like(exact)
                    sample={**scalar,'base':pressure,'native_input_residual':rhs}
                    rounded=exact.float().double()
                    rounding=coefficient(rounded-exact)
                    for calls in range(1,max(config['spatial_calls'])+1):
                        state={**sample,'base':pressure,'native_input_residual':rhs-audit.MODEL.full_operator(
                            pressure,scalar['cell_size_m'],scalar['rho'])}
                        pressure=predict_proposal(audit.MODEL,audit.NETWORK,state,audit.NORM,True)
                        if calls in config['spatial_calls']:
                            delta=coefficient(pressure-exact)
                            rows.append({'counts_xyz':counts,'frequency_modes_zyx':frequencies,'kind':kind,
                                'low_pressure_amplitude_pa':alpha,'high_pressure_amplitude_pa':0 if kind=='pure-low' else 1.,
                                'calls':calls,'low_coefficient_error_pa':delta,
                                'low_relative_amplitude_error':abs(delta/alpha) if alpha else None,
                                'float32_rounded_exact_low_error_pa':rounding,
                                'independent_rhs_relative_defect':defect})
    return {'method':'Manufactured stationary spatial cosine modes. Vary low pressure coefficient; pure-low globally scales field, mixed holds high-mode coefficient1Pa. High-only measures low-mode leakage. Same grid and scalar physics. Float32-rounded exact pressure is a representability comparator, not the learned inference arithmetic floor (which accumulates physical pressure in float64).',
            'rows':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--model',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    config=json.loads(args.config.read_text());source=config['models'][args.model]
    started=time.perf_counter()
    audit.initialize(LAB/source['project'],LAB/source['artifacts'],sorted(os.sched_getaffinity(0)))
    current=int(next(s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmSize:')).split()[1])*1024
    resource.setrlimit(resource.RLIMIT_AS,(current+512*1024**2,current+512*1024**2))
    result={'schema':'rheon-amplitude-frequency-diagnostic-v1','model':args.model,'source':source,
        'config_sha256':digest(args.config),'source_sha256':digest(__file__),
        'weights_sha256':digest(LAB/source['artifacts']/'weights.npz'),
        'source_witness_sha256':digest(LAB/source['project']/'source-witness.json'),
        'temporal':temporal(config),'spatial':spatial(config),'optimizer_updates':0,'simulations_advanced':0,
        'cuda_initialized':audit.TORCH.cuda.is_initialized(),'cpu_threads':audit.TORCH.get_num_threads(),
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'elapsed_seconds':time.perf_counter()-started,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'scope':'Consumed teacher-state replay and manufactured diagnostics; not independent validation, temporal dynamics intervention or training curriculum.'}
    assert not result['cuda_initialized']
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('model','elapsed_seconds','peak_rss_bytes')}),flush=True)


if __name__=='__main__':main()
