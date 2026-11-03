"""Restore an original guided composition from captured code;compare GPU probe."""
import argparse
import importlib
import os
from pathlib import Path
import resource
import time

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require
from component_material import current_publication
from cuda_forward_observer import CUDAForwardObserver
from guided_captured_runtime import GuidedCapturedRuntime, restore_network, guard_direct_methods
from macro_readout_fields import FIELDS, digest
from modal_component_benchmark import verify_sources
from pressure_portfolio_gpu_timing import resources


def inventory(model):
    return {name:dict(dtype=value.detach().cpu().numpy().dtype.str,shape=list(value.shape),
                     sha256=digest(value.detach().cpu().numpy())) for name,value in model.state_dict().items()}


def run(protocol_path,fit_id):
    protocol=read(protocol_path);current_publication(ROOT);verify_sources(protocol)
    require(os.environ.get('TULDOK_TORCH_DEVICE')=='cuda' and fit_id in protocol['fit_ids'], 'Explicit captured CUDA probe required')
    contract=read(ROOT/protocol['contract']['path']);item=next(m for m in contract['models'] if m['fit_item_id']==fit_id)
    meta=read(ROOT/item['manifest']['path'])
    folder=ROOT/protocol['output_root']/fit_id
    require(folder.is_relative_to(RUN) and not folder.exists(), 'Fresh original guided probe owner required;no retry')
    folder.mkdir(parents=True);os.sched_setaffinity(0,{protocol['host_cpu']})
    result=dict(status='original_guided_GPU_probe_running',created_unix=time.time(),pid=os.getpid(),
        protocol=binding(protocol_path),fit_item_id=fit_id,resource_observations=[resources()],
        inference_device='cuda',CPU_neural_forwards=0,latency_qualified=False,primary_timing_samples=0,
        full2880block_accuracy_qualified=False,new_parameter_fits=0,new_reference_generation=0,
        model_dataset_or_project_copies=0,native_mutations=0,full_portfolio_complete=False)
    save(folder/'result.json',result)
    observer=None;model=bank=runtime=None
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        require(torch.cuda.is_available() and torch.cuda.get_device_name()==protocol['expected_GPU']
                and not torch.is_autocast_enabled('cuda'), 'Registered CUDA hardware/precision required')
        observer=CUDAForwardObserver(torch);observer.enable('captured_CUDA_constructor')
        with GuidedCapturedRuntime(item,meta,contract,folder,extra_read_paths=protocol['source_sha256']) as runtime:
            view_root=runtime.root
            try:
                core=importlib.import_module(item['actual_fit_core_module'])
                require(core.GuidedSharedResidual.__module__==item['actual_model_module'] if item['task']=='guided_shared_component'
                        else core.GuidedPressureResidual.__module__==item['actual_model_module'], 'Actual captured trainer model implementation required')
                with torch.device('cuda'),torch.no_grad():model=core.make_model(item['configuration'],4)
                actual_initial=inventory(model)
                expected_initial=read(ROOT/item['original_configuration']['path'])['initial_model_inventory']
                require(actual_initial==expected_initial, 'Every original initial frozen/model tensor must match')
                restore_network(torch,np,model,item)
                selected=inventory(model)
                direct=guard_direct_methods(torch,model.network)
                guide_direct=guard_direct_methods(torch,model.guide.network)
                from captured_long_training_bank_v1 import CapturedLongTrainingBank
                bank=CapturedLongTrainingBank(ROOT/contract['data_archive']['path'],{k:contract['data_archive'][k] for k in ('bytes','sha256')})
                window=next(w for w in bank.windows if w['identifier']==item['probe_identifier'] and w['start_fine_step']==item['probe_start_fine_step'])
                scene=bank.manifest['masters'][window['identifier']]['scene']
                native=bank.load(window['identifier'],window['start_fine_step'])
                current={name:torch.from_numpy(native[name]).to('cuda') for name in FIELDS}
                require(time.time()-result['created_unix']<protocol['worker_deadline_seconds'], 'Finite original probe setup deadline exceeded')
                observer.set_phase('original_causal_guided_probe')
                with torch.inference_mode():outputs=model.predict(current,scene,window['start_fine_step'],60,4)
                require(len(outputs)==4 and all(set(o)==set(FIELDS) and all(v.device.type=='cuda' for v in o.values()) for o in outputs), 'All four original causal outputs must remain CUDA')
                vector=torch.cat([output[name].double().flatten() for output in outputs for name in FIELDS]).cpu().numpy().copy()
                torch.cuda.synchronize()
                with (ROOT/item['probe']['path']).open('rb') as stream:original=np.load(stream,allow_pickle=False)
                require(original.dtype.str==vector.dtype.str=='<f8' and original.shape==(3,667648)
                        and vector.shape==(667648,) and np.isfinite(vector).all(), 'Original selected complete4output probe ABI required')
                require(inventory(model)==selected, 'Original guided probe changed learned tensors')
                difference=vector-original[0]
                result.update(status='original_guided_selected_GPU_probe_replayed',
                    original_selected_GPU_probe_bitwise_equal=bool(np.array_equal(vector,original[0])),
                    maximum_absolute_probe_difference=float(np.max(np.abs(difference))),
                    relative_probe_l2=float(np.linalg.norm(difference)/max(np.linalg.norm(original[0]),1e-30)),
                    actual_model_module=item['actual_model_module'],captured_manifest_model_family=item['captured_model_family'],
                    initial_all_tensor_inventory_exact=True,selected_composition_inventory=selected,
                    model_asset_identity=model.asset_identity,physical_outputs=4,output_dt_s=1/60,block_duration_s=4/60,
                    input_hashes={name:digest(native[name]) for name in FIELDS},
                    output_hashes=[{name:digest(v.detach().cpu().numpy()) for name,v in output.items()} for output in outputs],
                    output_dtypes=[{name:v.detach().cpu().numpy().dtype.str for name,v in output.items()} for output in outputs],
                    residual_direct_CUDA_calls=direct,frozen_guide_direct_CUDA_calls=guide_direct,
                    original_weights=item['weights'],original_probe=item['probe'],future_targets_forwarded=False,
                    captured_imports=runtime.imported_sources(),temporary_Python_bytes=runtime.code_bytes,
                    scientific_payload_symlink_bindings=runtime.links,ambient_access_attempts=runtime.denied,
                    peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
                    maximum_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    GPU=torch.cuda.get_device_name(),Torch_version=torch.__version__,CUDA_version=torch.version.cuda)
                require(result['peak_CUDA_allocated_bytes']<=protocol['maximum_CUDA_allocated_bytes']
                        and result['maximum_RSS_bytes']<=protocol['maximum_RSS_bytes'], 'Registered finite probe resource limits exceeded')
            finally:
                if model is not None:model.close();model=None
                if bank is not None:bank.close();bank=None
        result['temporary_source_view_removed']=not view_root.exists()
        require(not runtime.denied, 'Actual captured constructor/predict must not read ambient scientific material')
        require(time.time()-result['created_unix']<protocol['worker_deadline_seconds'], 'Finite original probe deadline exceeded')
        verify_sources(protocol)
    except BaseException as error:
        result.update(status='original_guided_probe_failed_requires_reconciliation',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if observer is not None:observer.close();result['module_forward_guard']=observer.evidence()
        if runtime is not None:
            result['ambient_access_attempts']=runtime.denied
            if hasattr(runtime,'root'):result['temporary_source_view_removed']=not runtime.root.exists()
        result['cleanup_complete']=True;result['resource_observations'].append(resources())
        result['finished_unix']=time.time();save(folder/'result.json',result)
    print(fit_id,result['status'],'bitwise exact:',result['original_selected_GPU_probe_bitwise_equal'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--fit-id',required=True)
    args=parser.parse_args();run(args.protocol.resolve(),args.fit_id)
