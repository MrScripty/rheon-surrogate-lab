"""Restore original correction heads on CUDA and compare retained GPU probes."""
import hashlib
import io
import os
from pathlib import Path
import sys
import time
import zipfile

from import_common_generator_measurements import ROOT, DOC, RUN, read, save, binding, require
from component_material import current_publication, observed_file
from cuda_forward_observer import CUDAForwardObserver
from macro_readout_fields import FIELDS, digest
from modal_component_benchmark import imported, verify_sources
from pressure_portfolio_gpu_timing import resources


def run():
    output = DOC/'reports/internal-residual-original-GPU-probe-01.json'
    require(not output.exists(), 'Preserve original correction probe replay')
    current_publication(ROOT)
    contract_path = RUN/'internal-residual-component-contract-01.json'
    contract = read(contract_path)
    require(os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda', 'Explicit CUDA required')
    verify_sources(contract)
    observed_file(ROOT,contract['archive']['path'],contract['archive'])
    os.sched_setaffinity(0,{contract['host_cpu']})
    result = dict(status='original_internal_residual_GPU_probe_running',created_unix=time.time(),pid=os.getpid(),
        contract=binding(contract_path),models=[],resource_observations=[resources()],
        inference_device='cuda',CPU_neural_forwards=0,accuracy_qualified=False,latency_qualified=False,
        full783request_benchmark_pending=True,primary_timing_samples=0,new_parameter_fits=0,
        new_reference_generation=0,model_dataset_or_project_copies=0,native_mutations=0,
        full_portfolio_complete=False)
    save(output,result)
    observer = None
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        require(torch.cuda.is_available() and torch.cuda.get_device_name() == contract['expected_GPU']
                and not torch.is_autocast_enabled('cuda'), 'Original CUDA hardware/precision required')
        observer=CUDAForwardObserver(torch);observer.enable('CUDA_restore_and_probe')
        model_sources=contract['models'][0]['sources']
        base=imported('internal_residual_network_v1',ROOT/model_sources['internal_residual_network_v1.py']['path'])
        network=imported('internal_residual_network_v2',ROOT/model_sources['internal_residual_network_v2.py']['path'])
        loss=imported('internal_residual_loss_v1',ROOT/model_sources['internal_residual_loss_v1.py']['path'])
        manifest=read(ROOT/contract['manifest']['path'])
        with zipfile.ZipFile(ROOT/contract['archive']['path']) as archive:
            def arrays(spec):
                with archive.open(spec['member']) as handle:raw=handle.read(spec['bytes']+1)
                require(len(raw)==spec['bytes'] and hashlib.sha256(raw).hexdigest()==spec['sha256'],
                        'Original archive member bytes differ')
                with np.load(io.BytesIO(raw),allow_pickle=False) as stored:
                    require(set(stored.files)==set(spec['arrays']), 'Original member array membership differs')
                    values={name:stored[name].copy() for name in stored.files}
                for name,value in values.items():
                    expected=spec['arrays'][name]
                    require(value.dtype.str==expected['dtype'] and list(value.shape)==expected['shape']
                            and np.isfinite(value).all(), 'Original member typed finite array differs')
                return values
            row=manifest['parts'][0]
            data=arrays(row)
            sparse=arrays(manifest['sparse_part'])
            geometry=arrays(next(p for p in manifest['geometry_parts'] if p['member']==row['geometry_member']))
        normalization=read(ROOT/contract['normalization']['path'])
        with torch.inference_mode():
            response=loss.FrozenCoupledTraining(sparse,geometry,device='cuda')
            features=torch.from_numpy(data['features']).to('cuda')
            causal={name:float(data[name]) for name in ('h','dt','rho')}
            causal.update({'base_'+name:torch.from_numpy(data['base_'+name]).to('cuda') for name in FIELDS})
            require(set(causal)=={'h','dt','rho',*('base_'+name for name in FIELDS)}
                    and all(value.device.type=='cuda' for value in [features,*response.axes,response.divisor,
                        *(value[3] for value in response.sparse.values()),
                        *(causal['base_'+name] for name in FIELDS)]), 'Only original causal CUDA response operands required')
            for item in contract['models']:
                model=network.InternalGroupResidual(device='cuda',normalization=normalization).eval().requires_grad_(False)
                with np.load(ROOT/item['weights']['path'],allow_pickle=False) as archive:
                    require(set(archive.files)==set(item['inventory'])==set(model.state_dict()), 'Exact original checkpoint membership required')
                    values={}
                    for name in archive.files:
                        value=archive[name];spec=item['inventory'][name]
                        require(value.dtype.str==spec['dtype']=='<f4' and list(value.shape)==spec['shape']
                                and digest(value)==spec['sha256'], 'Original correction checkpoint tensor differs')
                        values[name]=torch.from_numpy(value.copy()).to('cuda')
                    model.load_state_dict(values,strict=True)
                require(sum(p.numel() for p in model.parameters())==10876
                        and all(p.device.type=='cuda' for p in [*model.parameters(),*model.buffers()]),
                        'CUDA-only original10876parameter correction required')
                delta=model(features)
                require(len(delta)==4 and all(v.device.type=='cuda' and v.dtype==torch.float32 for v in delta),
                        'Original CUDA delta ABI required')
                emitted=response.emit(causal,delta)
                require(set(emitted)==set(FIELDS) and all(v.device.type=='cuda' and v.dtype==torch.float64
                        for v in emitted.values()), 'Original F64 frozen response fields required')
                vector=torch.cat([emitted[name].flatten() for name in FIELDS]).cpu().numpy().copy()
                torch.cuda.synchronize()
                with (ROOT/item['probe']['path']).open('rb') as stream:original=np.load(stream,allow_pickle=False)
                require(original.dtype.str==vector.dtype.str=='<f8'
                        and original.shape==vector.shape==(item['probe_abi']['elements'],)
                        and np.isfinite(vector).all(), 'Complete retained GPU probe ABI required')
                still=features.clone();still[:,:7]=0;still[:,8:39]=0
                quiescent=model(still)
                require(all(torch.count_nonzero(v).item()==0 for v in quiescent), 'Original exact quiescence gate must remain effective')
                require(all(digest(v.detach().cpu().numpy())==item['inventory'][name]['sha256']
                        for name,v in model.state_dict().items()), 'Probe changed original learned weights/buffers')
                result['models'].append(dict(fit_item_id=item['fit_item_id'],weights=item['weights'],
                    retained_probe=item['probe'],probe_elements=vector.size,
                    retained_GPU_probe_bitwise_equal=bool(np.array_equal(original,vector)),
                    maximum_absolute_probe_difference=float(np.max(np.abs(original-vector))),
                    probe_relative_l2=float(np.linalg.norm(original-vector)/max(np.linalg.norm(original),1e-30)),
                    emitted_field_hashes={name:digest(value.detach().cpu().numpy()) for name,value in emitted.items()},
                    CUDA_model_calls=2,quiescent_delta_exact_zero=True,targets_forwarded=False))
                save(output,result)
        verify_sources(contract)
        from python_source_coverage import collect
        coverage=collect(sys.modules,ROOT)
        require(all(name in contract['source_sha256'] for name in coverage['lab_python']), 'Unbound actual imported correction helper')
        result.update(status='four_original_internal_residual_GPU_probes_replayed',GPU=torch.cuda.get_device_name(),
            Torch_version=torch.__version__,CUDA_version=torch.version.cuda,source_coverage=coverage,
            all_retained_GPU_probes_bitwise_equal=all(m['retained_GPU_probe_bitwise_equal'] for m in result['models']),
            original_source_and_parameter_values_unchanged=True,CUDA_model_calls=8,
            source_internal_request_identity={key:row[key] for key in
                ('identifier','start','group_index','width','counts_xyz','cells_per_metre')})
    except BaseException as error:
        result.update(status='internal_residual_probe_failed_requires_reconciliation',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if observer is not None:observer.close();result['module_forward_guard']=observer.evidence()
        result['cleanup_complete']=True;result['resource_observations'].append(resources())
        result['finished_unix']=time.time();save(output,result)
    print(result['status'],'all bitwise exact:',result['all_retained_GPU_probes_bitwise_equal'],flush=True)


if __name__ == '__main__':
    run()
