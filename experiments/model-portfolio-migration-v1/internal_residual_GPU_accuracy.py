"""Original CUDA corrections, original coupled response, independent field scores.

The supplied internal fine-history context is explicit. Targets are opened only
after all four causal component outputs have been generated for the request.
No complete-generator or latency claim follows from this component benchmark.
"""
import argparse
from contextlib import ExitStack
import hashlib
import io
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import zipfile

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require
from component_material import current_publication, observed_file
from cuda_forward_observer import CUDAForwardObserver
from macro_readout_fields import FIELDS, digest, physical_scores
from modal_component_benchmark import imported, verify_sources
from pressure_portfolio_gpu_timing import resources


METRICS=('pressure','equation','velocity','tracer')


def run(protocol_path,phase):
    protocol=read(protocol_path)
    current_publication(ROOT)
    require(phase in (0,1) and os.environ.get('TULDOK_TORCH_DEVICE')=='cuda', 'Explicit original CUDA phase required')
    verify_sources(protocol)
    contract=read(ROOT/protocol['contract']['path'])
    evaluation=read(ROOT/protocol['evaluation_normalization']['path'])
    require(isinstance(evaluation,dict) and 'relative_denominator_rms_floor' in evaluation,
            'Evaluation normalization document required before CUDA construction')
    observed_file(ROOT,contract['archive']['path'],contract['archive'])
    folder=ROOT/protocol['output_root']/('phase'+str(phase))
    require(folder.is_relative_to(RUN) and not folder.exists(), 'Fresh original correction phase required;no retry')
    prior=None
    if phase:
        prior=read(folder.parent/'phase0/result.json')
        require(prior['status']=='all_original_correction_requests_GPU_scored'
                and prior['cleanup_complete'] and not Path('/proc',str(prior['pid'])).exists(),
                'Original first phase must be terminal before reverse-order replay')
    require(shutil.disk_usage(ROOT).free >= protocol['minimum_free_disk_bytes'], 'Registered free disk floor required')
    folder.mkdir(parents=True)
    os.sched_setaffinity(0,{protocol['host_cpu']})
    began=time.monotonic()
    result=dict(status='original_internal_residual_GPU_accuracy_running',created_unix=time.time(),pid=os.getpid(),
        protocol=binding(protocol_path),phase=phase,models=[],resource_observations=[resources()],
        inference_device='cuda',CPU_neural_forwards=0,primary_timing_samples=0,latency_qualified=False,
        complete_generator_accuracy_claim=False,independent_holdout=False,own_history_drift=False,
        future_targets_forwarded=False,new_parameter_fits=0,new_reference_generation=0,
        model_dataset_or_project_copies=0,native_mutations=0,full_portfolio_complete=False)
    save(folder/'result.json',result)
    observer=None
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        require(torch.cuda.is_available() and torch.cuda.get_device_name()==protocol['expected_GPU']
                and not torch.is_autocast_enabled('cuda'), 'Original CUDA precision/hardware required')
        observer=CUDAForwardObserver(torch);observer.enable('CUDA_restore')
        source=contract['models'][0]['sources']
        imported('internal_residual_network_v1',ROOT/source['internal_residual_network_v1.py']['path'])
        network=imported('internal_residual_network_v2',ROOT/source['internal_residual_network_v2.py']['path'])
        loss=imported('internal_residual_loss_v1',ROOT/source['internal_residual_loss_v1.py']['path'])
        manifest=read(ROOT/contract['manifest']['path'])
        normalization=read(ROOT/contract['normalization']['path'])
        models,states,worst,prior_rows={},{},{},{}
        model_order=contract['models'][::-1] if phase else contract['models']
        for item in model_order:
            fid=item['fit_item_id']
            model=network.InternalGroupResidual(device='cuda',normalization=normalization).eval().requires_grad_(False)
            with np.load(ROOT/item['weights']['path'],allow_pickle=False) as archive:
                require(set(archive.files)==set(item['inventory'])==set(model.state_dict()), 'Exact original checkpoint membership required')
                values={}
                for name in archive.files:
                    value=archive[name];spec=item['inventory'][name]
                    require(value.dtype.str==spec['dtype']=='<f4' and list(value.shape)==spec['shape']
                            and digest(value)==spec['sha256'], 'Original checkpoint tensor differs')
                    values[name]=torch.from_numpy(value.copy()).to('cuda')
                model.load_state_dict(values,strict=True)
            require(all(p.device.type=='cuda' for p in [*model.parameters(),*model.buffers()]), 'CUDA-only checkpoint restore required')
            models[fid]=model
            states[fid]=dict(fit_item_id=fid,weights=item['weights'],rows=[],
                supplied_internal_stage_context=True,original_raw_weights_preserved=True)
            result['models'].append(states[fid]);worst[fid]={}
            if prior:
                prev=next(m for m in prior['models'] if m['fit_item_id']==fid)
                prior_rows[fid]={r['case_index']:r for r in prev['rows']}
        cases=list(enumerate(manifest['parts']))
        if phase:cases.reverse()
        geometry_specs={p['member']:p for p in manifest['geometry_parts']}
        responses={}
        maximum_metric_difference=0.
        torch.cuda.reset_peak_memory_stats()
        with ExitStack() as stack,torch.inference_mode():
            archive=stack.enter_context(zipfile.ZipFile(ROOT/contract['archive']['path']))
            def member(spec):
                with archive.open(spec['member']) as stream:raw=stream.read(spec['bytes']+1)
                require(len(raw)==spec['bytes'] and hashlib.sha256(raw).hexdigest()==spec['sha256'], 'Original corpus member changed')
                stored=np.load(io.BytesIO(raw),allow_pickle=False)
                require(set(stored.files)==set(spec['arrays']), 'Original corpus array membership differs')
                return stored
            def take(stored,spec,name):
                value=stored[name];expected=spec['arrays'][name]
                require(value.dtype.str==expected['dtype'] and list(value.shape)==expected['shape']
                        and np.isfinite(value).all(), 'Original typed finite corpus array differs: '+name)
                return value
            with member(manifest['sparse_part']) as stored:
                sparse={name:take(stored,manifest['sparse_part'],name) for name in stored.files}
            for position,(index,case) in enumerate(cases):
                require(time.monotonic()-began < protocol['worker_deadline_seconds'], 'Registered finite accuracy deadline exceeded')
                geometry_member=case['geometry_member']
                if geometry_member not in responses:
                    spec=geometry_specs[geometry_member]
                    with member(spec) as stored:
                        geometry={name:take(stored,spec,name) for name in stored.files}
                    responses[geometry_member]=loss.FrozenCoupledTraining(sparse,geometry,device='cuda')
                response=responses[geometry_member]
                with member(case) as stored:
                    scalar={name:float(take(stored,case,name)) for name in ('h','dt','rho')}
                    require(scalar['h']==1./case['cells_per_metre'] and scalar['rho']==1000.
                            and math.isclose(scalar['dt'],case['width']/480,rel_tol=1e-14)
                            and case['width']==(2,2,3)[case['group_index']], 'Original physical group duration/spacing required')
                    input_arrays={'features':take(stored,case,'features'),
                                  **{'base_'+n:take(stored,case,'base_'+n) for n in FIELDS}}
                    require(input_arrays['features'].shape==(1,42,*reversed(case['counts_xyz'])), 'Original42channel input ABI required')
                    features=torch.from_numpy(input_arrays['features']).to('cuda')
                    causal=dict(scalar,**{n:torch.from_numpy(v).to('cuda') for n,v in input_arrays.items() if n!='features'})
                    require(set(causal)=={'h','dt','rho',*('base_'+n for n in FIELDS)}
                            and all(v.device.type=='cuda' for v in [features,*response.axes,response.divisor,
                                *(p[3] for p in response.sparse.values()),*(causal['base_'+n] for n in FIELDS)]),
                            'Only causal current/base fields and frozen CUDA response operands required')
                    identity={name:case[name] for name in ('identifier','start','group_index','width','counts_xyz','cells_per_metre')}
                    inputs={name:digest(value) for name,value in input_arrays.items()}
                    outputs={}
                    observer.set_phase('original_CUDA_correction_and_response')
                    for item in model_order:
                        fid=item['fit_item_id']
                        delta=models[fid](features)
                        require(len(delta)==4 and all(v.device.type=='cuda' and v.dtype==torch.float32 for v in delta), 'Original CUDA delta ABI required')
                        emitted=response.emit(causal,delta)
                        require(set(emitted)==set(FIELDS) and all(v.device.type=='cuda' and v.dtype==torch.float64 for v in emitted.values()), 'Original F64 coupled response ABI required')
                        arrays={name:value.detach().cpu().numpy().reshape(-1).copy() for name,value in emitted.items()}
                        torch.cuda.synchronize()
                        require(all(np.isfinite(v).all() for v in arrays.values()) and arrays['pressure'][0]==0., 'Finite original output fields/gauge required')
                        outputs[fid]=(arrays,emitted)
                    # Future labels are decoded only after every component output.
                    target={name:take(stored,case,'target_'+name).reshape(-1) for name in FIELDS}
                    target['rhs']=take(stored,case,'target_rhs').reshape(-1)
                    sample=dict(scalar,**{'target_'+n:torch.from_numpy(v.reshape(response.shape if n in ('pressure','tracer','rhs') else response.sizes[('vx','vy','vz').index(n)])).to('cuda') for n,v in target.items()})
                    scene=dict(counts=case['counts_xyz'],cells_per_metre=case['cells_per_metre'])
                    baseline=physical_scores({n:input_arrays['base_'+n].reshape(-1) for n in FIELDS},target,scene,evaluation)
                    observer.set_phase('fixed_original_metrics_and_independent_NumPy_scoring')
                    for item in model_order:
                        fid=item['fit_item_id'];arrays,emitted=outputs[fid]
                        measured=physical_scores(arrays,target,scene,evaluation)
                        original={name:math.sqrt(max(float(value),0.)) for name,value in loss.metrics(emitted,sample,evaluation).items()}
                        for name in METRICS:
                            a,b=measured['relative'][name],original[name]
                            maximum_metric_difference=max(maximum_metric_difference,abs(a-b))
                            require(abs(a-b)<=protocol['independent_metric_tolerance']['absolute']
                                    +protocol['independent_metric_tolerance']['relative']*max(abs(a),abs(b)), 'Independent physical score disagrees with original: '+name)
                        row=dict(case_index=index,identity=identity,physical_internal_group_dt_s=scalar['dt'],
                            source_member_sha256=case['sha256'],input_hashes=inputs,
                            output_hashes={n:digest(v) for n,v in arrays.items()},metrics=measured,
                            original_Torch_relative=original,archived_zero_correction_baseline=baseline,
                            targets_decoded_after_all_component_outputs=True)
                        if prior:
                            previous=prior_rows[fid][index]
                            require(row['identity']==previous['identity'] and row['input_hashes']==previous['input_hashes']
                                    and row['output_hashes']==previous['output_hashes'] and row['metrics']==previous['metrics'],
                                    'Fresh-process reverse-order correction fields/scores differ')
                        else:
                            for name in METRICS:
                                current=worst[fid].get(name)
                                if current is None or (measured['relative'][name],-index)>(current[0],-current[1]):
                                    worst[fid][name]=(measured['relative'][name],index,arrays)
                        states[fid]['rows'].append(row)
                if position%27==0:
                    result['completed_cases_per_fit']=position+1
                    save(folder/'result.json',result)
                    print('phase',phase,'complete',position+1,'/783requests perfit',flush=True)
                require(torch.cuda.max_memory_allocated()<=protocol['maximum_CUDA_allocated_bytes']
                        and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=protocol['maximum_RSS_bytes'], 'Registered accuracy resource ceiling exceeded')
            for item in model_order:
                fid=item['fit_item_id'];state=states[fid]
                require(len(state['rows'])==783 and all(digest(v.detach().cpu().numpy())==item['inventory'][name]['sha256']
                        for name,v in models[fid].state_dict().items()), 'All783requests and unchanged checkpoint required')
                if prior:
                    previous=next(m for m in prior['models'] if m['fit_item_id']==fid)
                    state['worst_field_capture']=previous['worst_field_capture']
                    state['retained_case_indices']=previous['retained_case_indices']
                else:
                    retained={value[1]:value[2] for value in worst[fid].values()}
                    path=folder/(fid+'-worst-fields.npz')
                    np.savez_compressed(path,**{f'case{index:03d}_{name}':value for index,fields in retained.items() for name,value in fields.items()})
                    state['worst_field_capture']=dict(**binding(path),bytes=path.stat().st_size)
                    state['retained_case_indices']=sorted(retained)
        from python_source_coverage import collect
        coverage=collect(sys.modules,ROOT)
        require(all(name in protocol['source_sha256'] for name in coverage['lab_python']), 'Unbound imported original correction helper')
        verify_sources(protocol)
        result.update(status='all_original_correction_requests_GPU_scored',completed_cases_per_fit=783,
            source_coverage=coverage,original_weights_and_sources_unchanged=True,
            CUDA_correction_requests=3132,frozen_CUDA_response_requests=3132,
            all_cross_process_fields_exact=bool(prior),
            maximum_absolute_Torch_NumPy_metric_difference=maximum_metric_difference,
            independent_scoring='Original captured Torch loss versus independent NumPy field/operator scorer on each actual output;separate process final report remains required.',
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
            maximum_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            GPU=torch.cuda.get_device_name(),Torch_version=torch.__version__,CUDA_version=torch.version.cuda)
    except BaseException as error:
        result.update(status='original_correction_accuracy_failed_requires_reconciliation',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if observer is not None:observer.close();result['module_forward_guard']=observer.evidence()
        result['cleanup_complete']=True;result['resource_observations'].append(resources())
        result['finished_unix']=time.time();save(folder/'result.json',result)
    print('Original correction phase',phase,':3132requests scored;no timing.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--phase',type=int,required=True)
    args=parser.parse_args()
    run(args.protocol.resolve(),args.phase)
