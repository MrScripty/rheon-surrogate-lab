"""Exact current-adapter parity on the immutable first saved control frame."""
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import sys
import time
import qualify


def request_for(sample, manifest):
    counts=list(sample['base'].shape[2:][::-1]);h=float(sample['cell_size_m'][0]);dt=float(sample['dt'][0]);start=15*dt
    fields={'previous_pressure':sample['base'].reshape(-1).tolist(),
            'initial_residual':sample['native_input_residual'].reshape(-1).tolist(),
            'density':[float(sample['rho'][0])],'dt':[dt]}
    units={'previous_pressure':'Pa','initial_residual':'m3/s2','density':'kg/m3','dt':'s'}
    inputs={}
    for name,semantic in manifest['inputs'].items():
        scalar=semantic in ('density','dt')
        inputs[name]={'semantic':semantic,'units':units[semantic],'association':'global' if scalar else 'cell',
                      'dimensions':[1,1,1] if scalar else counts,'scalar':'f64','layout':'x_fast','values':fields[semantic]}
    return {'kind':'predict','schema':'rheon-pressure-context-request-v1','id':16,'stage':'sealed-mac-pressure-context-v1',
            'context':{'run_id':'migration-fixed-scene0-native-prefix','generation':16,'mode':'diagnostic_candidate',
                       'interval_start_s':start,'interval_end_s':start+dt,'dt_s':dt,
                       'grid':{'counts_xyz':counts,'spacing_xyz_m':[h]*3,'origin_xyz_m':[0.,0.,0.]},
                       'prior':{'generation':15,'interval_end_s':start,'status':{'kind':'accepted'}}},'inputs':inputs}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','bindings','reference','manifest','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    receipt={'command':sys.argv,'source_sha256':qualify.digest(__file__),'manifest_sha256':qualify.digest(a.manifest),
             'reference_receipt_sha256':qualify.digest(a.reference/'receipt.json'),'bindings_sha256':qualify.digest(a.bindings),
             'python':sys.executable,'cpu_affinity':sorted(os.sched_getaffinity(0)),'optimizer_updates':0}
    try:
        b=qualify.read(a.bindings);r=qualify.read(a.reference/'receipt.json')
        qualify.require(r['status']=='passed' and r['result']['same_input_exact_to_retained'],'old saved-frame reproduction required')
        raw=qualify.bound_bytes(a.root,b['regression']['first_frame']['capture'])
        import model_adapter
        stamp=time.perf_counter();state=model_adapter.load(a.manifest);receipt['startup_hash_restore_seconds']=time.perf_counter()-stamp
        np,torch=state.np,state.torch
        qualify.require(state.model.ARCHITECTURE['input_channels']==1,'incorrect control channel metadata')
        with np.load(a.root/b['regression']['first_frame']['capture']['path'],allow_pickle=False) as arrays:
            original={k:arrays[k].copy() for k in arrays.files if k!='prediction'}
        value=request_for(original,state.manifest);sample=model_adapter.sample_from_request(value,state)
        for key,array in original.items():
            actual=sample[key].numpy();qualify.require(actual.dtype==array.dtype and actual.shape==array.shape and actual.tobytes()==array.tobytes(),'input mapping changed '+key)
        with torch.no_grad():
            inputs=state.model.build_inputs(sample['native_input_residual'],sample['base'],sample['cell_size_m'],sample['rho'],sample['dt'],state.normalization)
            raw=state.network(inputs)
        trace=[];stamp=time.perf_counter();result=model_adapter.predict(value,state,trace);receipt['fixed_frame_model_seconds']=time.perf_counter()-stamp
        pressure=np.asarray(result['pressure'],dtype='<f8').reshape(original['base'].shape)
        zero=copy.deepcopy(value)
        residual_name=next(k for k,v in state.manifest['inputs'].items() if v=='initial_residual')
        zero['inputs'][residual_name]['values']=[0.]*original['base'].size
        calls=[];hook=state.network.register_forward_hook(lambda *args:calls.append(1))
        try:
            bypass=np.asarray(model_adapter.predict(zero,state)['pressure'],dtype='<f8').reshape(original['base'].shape)
            qualify.require(not calls,'zero residual invoked network')
            rejected=[]
            bad_values=[]
            for name,mutate in [('cold',lambda q:q['context'].update(generation=1)),
                                ('stale_prior',lambda q:q['context']['prior'].update(generation=13)),
                                ('anisotropic',lambda q:q['context']['grid'].update(spacing_xyz_m=[.125,.25,.125])),
                                ('origin',lambda q:q['context']['grid'].update(origin_xyz_m=[1.,0.,0.])),
                                ('density',lambda q:q['inputs'][next(k for k,v in state.manifest['inputs'].items() if v=='density')].update(values=[999.]))]:
                q=copy.deepcopy(value);mutate(q);bad_values.append((name,q))
            for name,q in bad_values:
                try:model_adapter.predict(q,state)
                except model_adapter.ProtocolError:rejected.append(name)
                else:raise ValueError('unsupported context admitted '+name)
            qualify.require(not calls,'unsupported context invoked network')
        finally:hook.remove()
        measured={'input_float32':inputs.numpy(),'network_output_float32':raw.numpy(),'full_pressure_float64':pressure,'zero_bypass_float64':bypass}
        with np.load(a.reference/'reference-arrays.npz',allow_pickle=False) as reference:
            for key,array in measured.items():
                expected=reference[key]
                qualify.require(array.shape==expected.shape and array.dtype==expected.dtype and array.tobytes()==expected.tobytes(),'non-bit-identical migration '+key)
        qualify.require(trace==r['result']['trace'],'correction/stopping trace differs')
        qualify.require(not torch.cuda.is_initialized() and torch.get_num_threads()==1,'CPU execution changed')
        with (a.output/'request.json').open('x') as stream:json.dump(value,stream,allow_nan=False)
        with (a.output/'expected.npy').open('xb') as stream:np.save(stream,pressure.reshape(-1),allow_pickle=False)
        with (a.output/'expected-pressure.f64le').open('xb') as stream:stream.write(pressure.tobytes())
        receipt.update(status='passed',architecture_input_channels=1,exact_shapes_dtypes_preprocessing_network_output_reconstruction=True,
                       zero_residual_exact_no_network=True,unsupported_context_rejected_before_inference=rejected,
                       same_call_trace=True,actual_network_calls=trace[-1]['proposals'],cuda_initialized=False,torch_threads=1,
                       arrays={k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':qualify.sha(v.tobytes())} for k,v in measured.items()})
    except BaseException as exc:
        receipt.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        receipt.update(elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        with (a.output/'receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2,allow_nan=False)
    print(json.dumps({k:receipt[k] for k in ('status','elapsed_seconds','actual_network_calls','peak_rss_bytes')}))


if __name__=='__main__':main()
