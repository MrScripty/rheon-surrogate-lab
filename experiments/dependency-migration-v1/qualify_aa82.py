"""Offline original seven-channel reference; bounded reads of one existing frame.

Historical formats are decoded only here. The current adapter receives current
requests, and this reference decoder is never an application dependency.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import qualify
from qualify_adapter import request_for


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('old','new'))
    for n in ('root','bindings','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--manifest',type=Path);p.add_argument('--reference',type=Path)
    a=p.parse_args();started=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    receipt={'command':sys.argv,'source_sha256':qualify.digest(__file__),'bindings_sha256':qualify.digest(a.bindings),
             'python':sys.executable,'cpu_affinity':sorted(os.sched_getaffinity(0)),'optimizer_updates':0}
    try:
        b=qualify.read(a.bindings);legacy=b['legacy_desktop'];qualify.bound_bytes(a.root,legacy['descriptor'])
        if a.mode=='old':
            path=a.root/legacy['source']/'cpu_bridge.py';spec=importlib.util.spec_from_file_location('offline_original_aa82_bridge',path)
            bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
            np,torch,model,network,norm,policy=bridge.load(a.root/legacy['descriptor']['path'])
            root=a.root/'runs/sealed-water-v1-3';packet=root/'packets/train_validation';restored=root/'managed/stage-01/artifacts'
            splits=qualify.read(packet/'splits.json');rest=qualify.read(restored/'receipt.json');case=rest['cases'][0]
            qualify.require(case=={'cell_size_m':.125,'counts_xyz':[16,16,16],'frame':0,'native_frame':15,'parent':0,'split':'train','trajectory_id':'water-train-0000'},'changed fixed aa82 case')
            qualify.require(qualify.digest(packet/'dataset.npz')==splits['dataset_sha256'],'historical packet hash mismatch')
            qualify.require(qualify.digest(restored/'predictions.npz')==rest['outputs']['predictions.npz']['sha256'],'historical predictions archive mismatch')
            arrays={}
            with np.load(packet/'dataset.npz',allow_pickle=False) as z:
                for name in ('base','native_input_residual'):
                    key='p000_'+name;array=z[key];identity=splits['arrays'][key]
                    qualify.require(array.dtype.str==identity['dtype'] and list(array.shape)==identity['shape'] and qualify.sha(array.tobytes())==identity['sha256'],'changed historical selected input array')
                    arrays[name]=array[0].copy().reshape(1,1,16,16,16)
            arrays.update(cell_size_m=np.asarray([.125],dtype='<f8'),rho=np.asarray([1000.],dtype='<f8'),dt=np.asarray([1/60],dtype='<f8'))
            with np.load(restored/'predictions.npz',allow_pickle=False) as z:retained=z['case_000'].copy().reshape(1,1,16,16,16)
            qualify.require(qualify.sha(retained.tobytes())=='9af4715d737da4404093f8b80cf709ce9f676d364dd71148c160cd5873148361','fixed retained prediction changed')
            sample={k:torch.from_numpy(v.copy()) for k,v in arrays.items()}
            with torch.no_grad():
                inputs=model.build_inputs(sample['native_input_residual'],sample['base'],sample['cell_size_m'],sample['rho'],sample['dt'],norm)
                raw=network(inputs);pressure=model.predict_pressure(network,sample,norm)
                bypass=model.predict_pressure(network,{**sample,'native_input_residual':torch.zeros_like(sample['native_input_residual'])},norm)
            qualify.require(pressure.numpy().tobytes()==retained.tobytes() and bypass.numpy().tobytes()==arrays['base'].tobytes(),'old aa82 prediction/bypass no longer exact')
            with (a.output/'sample.npz').open('xb') as f:np.savez(f,**arrays)
            with (a.output/'reference-arrays.npz').open('xb') as f:np.savez(f,input_float32=inputs.numpy(),network_output_float32=raw.numpy(),full_pressure_float64=pressure.numpy(),zero_bypass_float64=bypass.numpy())
            receipt.update(status='passed',historical_prediction_exact=True,selected_array_bytes=2*65536,
                           dataset_sha256=splits['dataset_sha256'],predictions_sha256=qualify.digest(restored/'predictions.npz'),
                           pressure_sha256=qualify.sha(retained.tobytes()),reference_bridge_sha256=qualify.digest(path))
        else:
            import model_adapter
            qualify.require(qualify.read(a.reference/'receipt.json')['status']=='passed','old aa82 reproduction required')
            state=model_adapter.load(a.manifest);np,torch=state.np,state.torch
            qualify.require(state.model.ARCHITECTURE['input_channels']==7,'incorrect aa82 channel metadata')
            with np.load(a.reference/'sample.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            value=request_for(arrays,state.manifest);sample=model_adapter.sample_from_request(value,state)
            with torch.no_grad():
                inputs=state.model.build_inputs(sample['native_input_residual'],sample['base'],sample['cell_size_m'],sample['rho'],sample['dt'],state.normalization)
                raw=state.network(inputs)
            pressure=np.asarray(model_adapter.predict(value,state)['pressure'],dtype='<f8').reshape(arrays['base'].shape)
            zero=json.loads(json.dumps(value));key=next(k for k,v in state.manifest['inputs'].items() if v=='initial_residual');zero['inputs'][key]['values']=[0.]*arrays['base'].size
            calls=[];hook=state.network.register_forward_hook(lambda *args:calls.append(1))
            try:bypass=np.asarray(model_adapter.predict(zero,state)['pressure'],dtype='<f8').reshape(arrays['base'].shape)
            finally:hook.remove()
            qualify.require(not calls,'aa82 zero bypass invoked network')
            measured={'input_float32':inputs.numpy(),'network_output_float32':raw.numpy(),'full_pressure_float64':pressure,'zero_bypass_float64':bypass}
            with np.load(a.reference/'reference-arrays.npz',allow_pickle=False) as z:
                for k,v in measured.items():
                    qualify.require(v.shape==z[k].shape and v.dtype==z[k].dtype and v.tobytes()==z[k].tobytes(),'aa82 bitwise parity failure '+k)
            with (a.output/'request.json').open('x') as f:json.dump(value,f,allow_nan=False)
            with (a.output/'expected.npy').open('xb') as f:np.save(f,pressure.reshape(-1),allow_pickle=False)
            with (a.output/'expected-pressure.f64le').open('xb') as f:f.write(pressure.tobytes())
            receipt.update(status='passed',architecture_input_channels=7,manifest_sha256=qualify.digest(a.manifest),reference_receipt_sha256=qualify.digest(a.reference/'receipt.json'),
                           exact_preprocessing_network_output_full_pressure=True,zero_residual_exact_no_network=True,
                           pressure_sha256=qualify.sha(pressure.tobytes()))
        qualify.require(torch.get_num_threads()==1 and not torch.cuda.is_initialized(),'CPU one-thread inference required')
        receipt.update(torch_threads=1,cuda_initialized=False)
    except BaseException as exc:
        receipt.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        receipt.update(elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        with (a.output/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps({k:receipt[k] for k in ('status','elapsed_seconds','peak_rss_bytes')}))


if __name__=='__main__':main()
