"""Complete original guided causal outputs with independent physical scoring."""
import argparse
from collections import Counter
import importlib
import math
import os
from pathlib import Path
import resource
import sys
import time

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require
from component_material import current_publication
from cuda_forward_observer import CUDAForwardObserver
from guided_captured_runtime import GuidedCapturedRuntime, restore_network, guard_direct_methods
from guided_original_GPU_probe import inventory
from macro_readout_fields import FIELDS, digest, physical_scores
from modal_component_benchmark import verify_sources
from pressure_portfolio_gpu_timing import resources
import review_direction_pressure_fits


def predict_work(model,current,scene,start,rate,k):
    """Observe Python entrypoints only;never treat this as a timing sample."""
    widths=Counter();calls=Counter()
    previous=sys.getprofile();require(previous is None, 'One owned scientific work observer required')
    def profile(frame,event,arg):
        if event!='call':return
        code=frame.f_code;file=Path(code.co_filename).name
        if file=='temporal_group_sum_projector_v1.py' and code.co_name=='phase_packet':
            widths[str(frame.f_locals['width'])]+=1
        if (file,code.co_qualname) in {
            ('coupled_macrostep_v1.py','_RestoredPressure.forward'),
            ('temporal_group_projector_v1.py','TemporalGroupProjector.emit')}:
            calls[file+':'+code.co_qualname]+=1
    sys.setprofile(profile)
    try:output=model.predict(current,scene,start,rate,k)
    finally:sys.setprofile(previous)
    return output,dict(phase_packet_width_counts=dict(widths),learned_projection_entry_calls=dict(calls),
        observer='Scoped Python call/width observation;no tensor computation or primary timing')


def run(protocol_path,fit_id,phase):
    require(callable(review_direction_pressure_fits.operator), 'Preload bound independent scoring before source isolation')
    protocol=read(protocol_path);current_publication(ROOT);verify_sources(protocol)
    require(os.environ.get('TULDOK_TORCH_DEVICE')=='cuda' and phase in (0,1) and fit_id in protocol['fit_ids'], 'Explicit original CUDA phase required')
    contract=read(ROOT/protocol['contract']['path']);item=next(m for m in contract['models'] if m['fit_item_id']==fit_id)
    meta=read(ROOT/item['manifest']['path'])
    folder=ROOT/protocol['output_root']/f'{fit_id}-phase{phase}'
    require(folder.is_relative_to(RUN) and not folder.exists(), 'Fresh full guided phase required;no retry')
    prior=None
    if phase:
        prior=read(folder.parent/f'{fit_id}-phase0/result.json')
        require(prior['status']=='all2880original_guided_blocks_GPU_scored' and prior['cleanup_complete']
                and not Path('/proc',str(prior['pid'])).exists(), 'Original phase0 must be terminal before reverse-order replay')
    folder.mkdir(parents=True);os.sched_setaffinity(0,{protocol['host_cpu']})
    result=dict(status='original_guided_full_GPU_accuracy_running',created_unix=time.time(),pid=os.getpid(),
        protocol=binding(protocol_path),fit_item_id=fit_id,phase=phase,rows=[],resource_observations=[resources()],
        inference_device='cuda',CPU_neural_forwards=0,primary_timing_samples=0,latency_qualified=False,
        independent_holdout=False,own_history_drift=False,future_targets_forwarded=False,
        new_parameter_fits=0,new_reference_generation=0,model_dataset_or_project_copies=0,
        native_mutations=0,full_portfolio_complete=False)
    save(folder/'result.json',result)
    observer=model=bank=runtime=None
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        require(torch.cuda.is_available() and torch.cuda.get_device_name()==protocol['expected_GPU']
                and not torch.is_autocast_enabled('cuda'), 'Original CUDA hardware/precision required')
        observer=CUDAForwardObserver(torch);observer.enable('captured_CUDA_constructor')
        normalization=read(ROOT/item['normalization']['path'])
        prior_rows={r['case_index']:r for r in prior['rows']} if prior else {}
        worst={};objective=0.;maximum_metric_difference=0.;outputs_checked=0;probe_control_exact=False
        probe=read(ROOT/next(s['path'] for s in protocol['original_probe_controls'] if fit_id in s['path']))
        with GuidedCapturedRuntime(item,meta,contract,folder,extra_read_paths=protocol['source_sha256']) as runtime:
            view_root=runtime.root
            try:
                core=importlib.import_module(item['actual_fit_core_module'])
                with torch.device('cuda'),torch.no_grad():model=core.make_model(item['configuration'],4)
                require(inventory(model)==read(ROOT/item['original_configuration']['path'])['initial_model_inventory'], 'Original entire initial composition inventory required')
                restore_network(torch,np,model,item);selected=inventory(model)
                direct=guard_direct_methods(torch,model.network);guide_direct=guard_direct_methods(torch,model.guide.network)
                from captured_long_training_bank_v1 import CapturedLongTrainingBank
                from internal_residual_loss_v1 import metrics as original_metrics
                bank=CapturedLongTrainingBank(ROOT/contract['data_archive']['path'],{n:contract['data_archive'][n] for n in ('bytes','sha256')})
                torch.cuda.reset_peak_memory_stats()
                cases=list(enumerate(contract['cases']))
                if phase:cases.reverse()
                with torch.inference_mode():
                    for position,(index,case) in enumerate(cases):
                        require(time.time()-result['created_unix']<protocol['worker_deadline_seconds'], 'Finite full-population accuracy deadline exceeded')
                        identifier,start,rate,k=case['identifier'],case['start_fine_step'],case['output_steps_per_s'],case['K']
                        master=bank.manifest['masters'][identifier];scene=master['scene'];shape=tuple(reversed(scene['counts']))
                        native=bank.load(identifier,start)
                        current={n:torch.from_numpy(native[n]).to('cuda') for n in FIELDS}
                        observer.set_phase('captured_original_causal_prediction')
                        outputs,work=predict_work(model,current,scene,start,rate,k)
                        guides=model.last_guide_outputs
                        require(len(outputs)==len(guides)==k and all(set(o)==set(FIELDS)
                                and all(v.device.type=='cuda' for v in o.values()) for o in outputs), 'All Kcomplete causal CUDA outputs required')
                        arrays=[{n:v.detach().cpu().numpy().reshape(-1).copy() for n,v in o.items()} for o in outputs]
                        torch.cuda.synchronize()
                        require(all(np.isfinite(v).all() for o in arrays for v in o.values()), 'Finite complete physical output fields required')
                        hashes=[{n:digest(v) for n,v in o.items()} for o in arrays]
                        if identifier==item['probe_identifier'] and start==item['probe_start_fine_step'] and rate==60 and k==4:
                            require(hashes==probe['output_hashes'], 'Full-work observer changed original saved-probe output bytes')
                            probe_control_exact=True
                        row=dict(case_index=index,identity=dict(identifier=identifier,start_fine_step=start,
                            counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],output_steps_per_s=rate,
                            output_dt_s=1/rate,K=k,block_duration_s=k/rate),
                            input_hashes={n:digest(native[n]) for n in FIELDS},outputs=[],actual_work=work,
                            future_targets_decoded_after_causal_outputs=True)
                        block_loss=current['pressure'].new_zeros(())
                        observer.set_phase('fixed_original_Torch_and_independent_NumPy_scoring')
                        for lead,(output,fields,target_step,guide) in enumerate(zip(outputs,arrays,case['target_fine_steps'],guides),1):
                            truth=bank.load(identifier,target_step)
                            measured=physical_scores(fields,truth,scene,normalization)
                            baseline=physical_scores({n:v.detach().cpu().numpy().reshape(-1) for n,v in guide.items()},truth,scene,normalization)
                            shapes={n:shape if n in ('pressure','tracer') else (
                                shape[0]+(n=='vz'),shape[1]+(n=='vy'),shape[2]+(n=='vx')) for n in FIELDS}
                            sample=dict(h=1/scene['cells_per_metre'],rho=1000.,target_rhs=torch.from_numpy(truth['rhs']).to('cuda').reshape(shape))
                            sample.update({'target_'+n:torch.from_numpy(truth[n]).to('cuda').reshape(shapes[n]) for n in FIELDS})
                            reshaped={n:v.reshape(shapes[n]) for n,v in output.items()}
                            squared=original_metrics(reshaped,sample,normalization)
                            block_loss=block_loss+sum(squared.values())/k
                            original={n:math.sqrt(max(float(v),0.)) for n,v in squared.items()}
                            for name,value in original.items():
                                actual=measured['relative'][name];delta=abs(actual-value)
                                maximum_metric_difference=max(maximum_metric_difference,delta)
                                require(delta<=protocol['independent_metric_tolerance']['absolute']+
                                        protocol['independent_metric_tolerance']['relative']*max(abs(actual),abs(value)), 'Independent guided field scores disagree: '+name)
                            entry=dict(lead=lead,target_fine_step=target_step,physical_lead_dt_s=lead/rate,
                                output_hashes=hashes[lead-1],output_dtypes={n:v.dtype.str for n,v in fields.items()},
                                metrics=measured,original_Torch_relative=original,unchanged_guide_metrics=baseline)
                            row['outputs'].append(entry);outputs_checked+=1
                            if not phase:
                                for name in ('pressure','equation','velocity','tracer'):
                                    previous=worst.get(name);key=(measured['relative'][name],-index,-lead)
                                    if previous is None or key>previous[0]:worst[name]=(key,index,lead,fields)
                        weight=(1/24)*(1/len({w['start_fine_step'] for w in master['windows']}))/12
                        row['original_Torch_block_loss']=float(block_loss);objective+=float(block_loss)*weight
                        if prior:
                            require(row==prior_rows[index], 'Every full guided input/output/score/work row must repeat exactly')
                        result['rows'].append(row)
                        if position%120==0:
                            result['completed_blocks']=position+1;save(folder/'result.json',result)
                            print(fit_id,'phase',phase,'complete',position+1,'/2880blocks',flush=True)
                        require(torch.cuda.max_memory_allocated()<=protocol['maximum_CUDA_allocated_bytes']
                                and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=protocol['maximum_RSS_bytes'], 'Registered guided resource ceiling exceeded')
                require(len(result['rows'])==2880 and outputs_checked==7200 and probe_control_exact
                        and inventory(model)==selected, 'Complete population,original probe and unchanged composition required')
                expected=read(ROOT/item['history']['path'])['selected_worst_score']['objective']
                require(math.isclose(objective,expected,rel_tol=protocol['original_training_objective_tolerance']['relative'],
                        abs_tol=protocol['original_training_objective_tolerance']['absolute']), 'Recomputed complete original selected training objective differs')
                if phase:
                    result['worst_field_capture']=prior['worst_field_capture'];result['retained_outputs']=prior['retained_outputs']
                else:
                    retained={(v[1],v[2]):v[3] for v in worst.values()};path=folder/'worst-fields.npz'
                    np.savez_compressed(path,**{f'case{index:04d}_lead{lead}_{n}':v for (index,lead),o in retained.items() for n,v in o.items()})
                    result['worst_field_capture']=dict(**binding(path),bytes=path.stat().st_size)
                    result['retained_outputs']=[dict(case_index=i,lead=l) for i,l in sorted(retained)]
                result.update(status='all2880original_guided_blocks_GPU_scored',completed_blocks=2880,physical_outputs=7200,
                    original_selected_training_objective=expected,recomputed_training_objective=objective,
                    complete_original_objective_reproduced=True,maximum_absolute_Torch_NumPy_metric_difference=maximum_metric_difference,
                    original_saved_GPU_probe_output_hashes_exact=True,selected_composition_inventory=selected,
                    residual_direct_CUDA_calls=direct,frozen_guide_direct_CUDA_calls=guide_direct,
                    captured_imports=runtime.imported_sources(),actual_model_module=item['actual_model_module'],
                    peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),maximum_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    GPU=torch.cuda.get_device_name(),Torch_version=torch.__version__,CUDA_version=torch.version.cuda)
            finally:
                if model is not None:model.close();model=None
                if bank is not None:bank.close();bank=None
        require(not runtime.denied and not view_root.exists(), 'No ambient access or retained temporary code view allowed')
        verify_sources(protocol)
    except BaseException as error:
        result.update(status='guided_full_accuracy_failed_requires_reconciliation',error_type=type(error).__name__,error=str(error));raise
    finally:
        if observer is not None:observer.close();result['module_forward_guard']=observer.evidence()
        if runtime is not None:
            result['ambient_access_attempts']=runtime.denied
            if hasattr(runtime,'root'):result['temporary_source_view_removed']=not runtime.root.exists()
        result['cleanup_complete']=True;result['resource_observations'].append(resources())
        result['finished_unix']=time.time();save(folder/'result.json',result)
    print(fit_id,'phase',phase,':2880blocks/7200outputs scored;no timing.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--fit-id',required=True);parser.add_argument('--phase',type=int,required=True)
    args=parser.parse_args();run(args.protocol.resolve(),args.fit_id,args.phase)
