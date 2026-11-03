"""Expanded scalar-only replay for the last eight accepted component fits.

All emitted fields must match original accepted hashes. Internal correction
requests, physical readout outputs and guided K-output compositions remain
different tasks. Only scalar diagnostics/hashes are written; no field captures.
"""
import argparse
from contextlib import ExitStack
import gzip
import hashlib
import importlib
import io
import math
import os
from pathlib import Path
import sys
import time
import zipfile

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require
from macro_readout_fields import FIELDS, digest, load_fields, physical_scores
from full_state_scalar_diagnostics import measure as field_statistics
from pressure_scalar_diagnostics import field_summary, error_summary
from guided_captured_runtime import GuidedCapturedRuntime, restore_network, guard_direct_methods
from guided_GPU_accuracy import predict_work
from guided_original_GPU_probe import inventory as guided_inventory
import review_direction_pressure_fits


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class ScalarSink:
    def __init__(self,folder,result):
        self.folder,self.result=folder,result
        self.stack=ExitStack();self.streams={};self.counts={}

    def write(self,fid,index,identity,statistics,metrics,extra=None):
        if fid not in self.streams:
            path=self.folder/(fid+'-statistics.jsonl.gz')
            raw=self.stack.enter_context(path.open('xb'))
            self.streams[fid]=self.stack.enter_context(gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=6,mtime=0))
            self.counts[fid]=0
        record=dict(fit_item_id=fid,case_index=index,identity=identity,statistics=statistics,accepted_metrics=metrics)
        if extra is not None:record['component_specific_statistics']=extra
        self.streams[fid].write((__import__('json').dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
        self.counts[fid]+=1

    def close(self):
        self.stack.close()
        self.result['scalar_populations']={fid:dict(records=count,payload=dict(**binding(self.folder/(fid+'-statistics.jsonl.gz')),
            bytes=(self.folder/(fid+'-statistics.jsonl.gz')).stat().st_size)) for fid,count in self.counts.items()}


def restore_tensors(np,torch,network,item):
    with np.load(ROOT/item['weights']['path'],allow_pickle=False) as archive:
        require(set(archive.files)==set(item['inventory'])==set(network.state_dict()),'Original tensor inventory required')
        values={}
        for name,spec in item['inventory'].items():
            value=archive[name]
            require(value.dtype.str==spec['dtype'] and list(value.shape)==spec['shape'] and digest(value)==spec['sha256'],
                'Original learned tensor differs')
            values[name]=torch.from_numpy(value.copy()).to('cuda')
        network.load_state_dict(values,strict=True)
    network.eval().requires_grad_(False)
    require(all(v.device.type=='cuda' for v in [*network.parameters(),*network.buffers()]),'All learned assets must reside onCUDA')


def imported(name,path):
    import importlib.util
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def readout_job(np,torch,old,accepted,result,sink,observer,folder):
    sys.path.insert(0,str(ROOT/'experiments/full-ai-data-diversity-v1'))
    from macrostep_backbone_v1 import FrozenMacroBackbone
    from diagnose_causal_composition import inventory
    backbone=None
    try:
        observer.set_phase('readout_CUDA_constructor')
        with torch.device('cuda'),torch.no_grad():backbone=FrozenMacroBackbone(*old['backbone_arguments'])
        initial=inventory(backbone)
        norm=read(ROOT/old['normalization']['path']);contract=read(ROOT/old['contract']['path'])
        models=[];result['models']=models
        for item in old['models']:
            fid=item['fit_item_id'];stages=item['stages']
            module=imported('scalar_readout_s'+str(stages),ROOT/item['source']['path'])
            with torch.device('cuda'):head=module.MacroReadout(stages,device='cuda')
            restore_tensors(np,torch,head,item)
            prior=next(m for m in accepted['models'] if m['fit_item_id']==fid)
            table={r['case_index']:r for r in prior['rows']}
            state=dict(fit_item_id=fid,stages=stages,rows=[],input_scope='Causal originalfrozenbackbone S+1stage fields from current state',
                output_scope='Onecompletephysical1/60soutput;internalstages arenotfutureframes')
            models.append(state)
            with torch.inference_mode():
                for index,case in enumerate(contract['cases']):
                    scene=read(ROOT/case['receipt'])['scene'];native=load_fields(case['current_state'])
                    current={n:torch.from_numpy(v).to('cuda') for n,v in native.items()}
                    observer.set_phase('causal_feature_generation')
                    features=backbone.feature_states(current,scene,case['start'],stages,1)[0]
                    require(len(features)==stages+1 and all(v.device.type=='cuda' for row in features for v in row.values()),'All causal stage fields CUDA')
                    require(all(v.device.type=='cuda' and v.dtype==torch.float64 for v in head.parameters()),'Direct readout parameters CUDA F64')
                    emitted=head.read(features)
                    require(all(v.device.type=='cuda' for v in emitted.values()),'Direct readout outputs must remain CUDA')
                    fields={n:v.detach().to('cpu',copy=True).numpy().reshape(-1) for n,v in emitted.items()};torch.cuda.synchronize()
                    prior_row=table[index]
                    require({n:digest(v) for n,v in fields.items()}==prior_row['output_hashes'],'Readout changed accepted fields')
                    truth=load_fields(case['target_state'],(*FIELDS,'rhs'))
                    metrics=physical_scores(fields,truth,scene,norm)
                    require(metrics==prior_row['metrics'],'Readout changed accepted metrics')
                    h=1/scene['cells_per_metre'];stats=field_statistics(fields,truth,native,scene['counts'],[h]*3,1000.)
                    identity=dict(prior_row['identity'],K=1,lead=1,output_role='endpoint',output_dt_s=1/60,
                        physical_lead_dt_s=1/60,block_horizon_s=1/60,actual_internal_advances=stages)
                    sink.write(fid,index,identity,stats,metrics)
                    state['rows'].append(dict(case_index=index,identity=identity,output_hashes=prior_row['output_hashes'],
                        direct_CUDA_readout_calls=1,causal_internal_advances=stages))
                    if index%14==0:save(folder/'result.json',result)
            require(inventory(head)==item['inventory'],'Readout changed selected coefficients')
        require(inventory(backbone)==initial,'Readout replay changed frozen learned assets')
    finally:
        if backbone is not None:backbone.close()


def stage_context_names(stats):
    """Do not call the supplied endpoint baseline an initial physical state."""
    for field in stats['fields'].values():
        for old,new in [('input','supplied_zero_correction_stage_baseline'),
            ('unchanged_input_baseline_error','supplied_stage_baseline_error'),
            ('prediction_minus_input','prediction_minus_supplied_stage_baseline'),
            ('reference_minus_input','reference_minus_supplied_stage_baseline'),
            ('RMS_improvement_vs_unchanged_input','RMS_improvement_vs_supplied_stage_baseline'),
            ('RMS_ratio_vs_unchanged_input','RMS_ratio_vs_supplied_stage_baseline')]:field[new]=field.pop(old)
        field.pop('prediction_minus_preceding_output');field.pop('reference_minus_preceding_reference')
    stats['combined_velocity_supplied_stage_baseline_error']=stats.pop('combined_velocity_unchanged_input_baseline_error')
    stats['supplied_stage_baseline_equation_residual']=stats.pop('unchanged_input_equation_residual')
    stats['state_quantities']['supplied_stage_baseline']=stats['state_quantities'].pop('input')
    return stats


def correction_job(np,torch,old,accepted,result,sink,observer,folder):
    contract=read(ROOT/old['contract']['path']);manifest=read(ROOT/contract['manifest']['path'])
    evaluation=read(ROOT/old['evaluation_normalization']['path']);normalization=read(ROOT/contract['normalization']['path'])
    sources=contract['models'][0]['sources']
    imported('internal_residual_network_v1',ROOT/sources['internal_residual_network_v1.py']['path'])
    network=imported('internal_residual_network_v2',ROOT/sources['internal_residual_network_v2.py']['path'])
    loss=imported('internal_residual_loss_v1',ROOT/sources['internal_residual_loss_v1.py']['path'])
    models={};states={};tables={}
    result['models']=[]
    for item in contract['models']:
        fid=item['fit_item_id']
        with torch.device('cuda'):model=network.InternalGroupResidual(device='cuda',normalization=normalization)
        restore_tensors(np,torch,model,item);models[fid]=model
        state=dict(fit_item_id=fid,rows=[],input_scope='42original current/fine-history feature channels plus suppliedzero-correctionstagecontext',
            output_scope='Fourscaledcarrier deltas andF64coupledPVTforoneinternalgroup;not acompletephysicalstep')
        result['models'].append(state);states[fid]=state
        tables[fid]={r['case_index']:r for r in next(m for m in accepted['models'] if m['fit_item_id']==fid)['rows']}
    geometry_specs={p['member']:p for p in manifest['geometry_parts']};responses={}
    with zipfile.ZipFile(ROOT/contract['archive']['path']) as archive,torch.inference_mode():
        def member(spec):
            raw=archive.read(spec['member']);require(len(raw)==spec['bytes'] and hashlib.sha256(raw).hexdigest()==spec['sha256'],'Original corpus member differs')
            stored=np.load(io.BytesIO(raw),allow_pickle=False);require(set(stored.files)==set(spec['arrays']),'Original member array inventory differs');return stored
        def take(stored,spec,name):
            value=stored[name];expected=spec['arrays'][name]
            require(value.dtype.str==expected['dtype'] and list(value.shape)==expected['shape'] and np.isfinite(value).all(),'Typed corpus array differs')
            return value
        with member(manifest['sparse_part']) as stored:sparse={n:take(stored,manifest['sparse_part'],n) for n in stored.files}
        for index,case in enumerate(manifest['parts']):
            if case['geometry_member'] not in responses:
                spec=geometry_specs[case['geometry_member']]
                with member(spec) as stored:geometry={n:take(stored,spec,n) for n in stored.files}
                responses[case['geometry_member']]=loss.FrozenCoupledTraining(sparse,geometry,device='cuda')
            response=responses[case['geometry_member']]
            require(all(v.device.type=='cuda' for v in [*response.axes,response.divisor,*(p[3] for p in response.sparse.values())]),'Learned coupled response assets CUDA')
            with member(case) as stored:
                scalar={n:float(take(stored,case,n)) for n in ('h','dt','rho')}
                require(math.isclose(scalar['dt'],case['width']/480,rel_tol=1e-14),'Exact original internal group duration required')
                input_arrays={'features':take(stored,case,'features'),**{'base_'+n:take(stored,case,'base_'+n) for n in FIELDS}}
                features=torch.from_numpy(input_arrays['features']).to('cuda')
                causal=dict(scalar,**{n:torch.from_numpy(v).to('cuda') for n,v in input_arrays.items() if n!='features'})
                outputs={};deltas={}
                observer.set_phase('original_CUDA_correction_and_response')
                for fid,model in models.items():
                    delta=model(features);emitted=response.emit(causal,delta)
                    require(all(v.device.type=='cuda' for v in [*delta,*emitted.values()]),'All head/response outputs CUDA')
                    outputs[fid]={n:v.detach().cpu().numpy().reshape(-1).copy() for n,v in emitted.items()}
                    deltas[fid]=[v.detach().cpu().numpy().copy() for v in delta]
                torch.cuda.synchronize()
                truth={n:take(stored,case,'target_'+n).reshape(-1) for n in (*FIELDS,'rhs')}
                target_delta={n:take(stored,case,'target_delta_'+n) for n in ('vx','vy','vz','tracer')}
                base={n:input_arrays['base_'+n].reshape(-1) for n in FIELDS}
                scene=dict(counts=case['counts_xyz'],cells_per_metre=case['cells_per_metre'])
                for fid,fields in outputs.items():
                    prior=tables[fid][index]
                    require({n:digest(v) for n,v in fields.items()}==prior['output_hashes'],'Correction changed accepted fields')
                    metrics=physical_scores(fields,truth,scene,evaluation);require(metrics==prior['metrics'],'Correction changed accepted metrics')
                    stats=stage_context_names(field_statistics(fields,truth,base,case['counts_xyz'],[scalar['h']]*3,scalar['rho']))
                    extras={}
                    for name,predicted in zip(('vx','vy','vz','tracer'),deltas[fid],strict=True):
                        target=target_delta[name];require(predicted.shape==target.shape,'Original scaled delta target shape required')
                        extras[name]=dict(prediction=field_summary(predicted),reference=field_summary(target),
                            error=error_summary(predicted.astype(np.float64)-target,target))
                    identity=dict(prior['identity'],physical_internal_group_dt_s=scalar['dt'],case_role='supplied_internal_stage_context')
                    sink.write(fid,index,identity,stats,metrics,dict(scaled_carrier_delta_errors=extras,
                        delta_units='Velocity deltas inoriginalcell-displacement/h convention;tracerdelta inoriginalconcentrationunits;output_rms scalesalreadyapplied',
                        input_feature_summary_before_network_RMS_normalization=field_summary(input_arrays['features'])))
                    states[fid]['rows'].append(dict(case_index=index,identity=identity,output_hashes=prior['output_hashes'],
                        observed_complete_head_and_response_requests=1))
            if index%40==0:
                result['completed_cases_per_fit']=index+1;save(folder/'result.json',result)
                print('corrections',index+1,'/783requests perfit',flush=True)
    for item in contract['models']:
        for name,value in models[item['fit_item_id']].state_dict().items():
            require(digest(value.detach().cpu().numpy())==item['inventory'][name]['sha256'],'Correction changed selected parameter/buffer')


def guided_job(np,torch,old,accepted,result,sink,observer,folder,fit_id,new):
    contract=read(ROOT/old['contract']['path']);item=next(m for m in contract['models'] if m['fit_item_id']==fit_id)
    meta=read(ROOT/item['manifest']['path']);normalization=read(ROOT/item['normalization']['path'])
    table={(r['case_index'],r['lead']):r for r in next(m for m in accepted['models'] if m['fit_item_id']==fit_id)['rows']}
    result['models']=[dict(fit_item_id=fit_id,rows=[],input_scope='CurrentnativePVTplusprescribedcontrols/events;capturedoriginalcomposition',
        output_scope='Everyphysicallead at30/60/120outputs_per_s,K1–4;endpoint andintermediates scoredseparately')]
    state=result['models'][0];model=bank=None;captured={}
    with GuidedCapturedRuntime(item,meta,contract,folder,extra_read_paths=new['admitted_lab_python']) as runtime:
        view_root=runtime.root
        try:
            core=importlib.import_module(item['actual_fit_core_module'])
            with torch.device('cuda'),torch.no_grad():model=core.make_model(item['configuration'],4)
            require(guided_inventory(model)==read(ROOT/item['original_configuration']['path'])['initial_model_inventory'],'Original full initial composition required')
            restore_network(torch,np,model,item);selected=guided_inventory(model)
            direct=guard_direct_methods(torch,model.network);guide_direct=guard_direct_methods(torch,model.guide.network)
            from captured_long_training_bank_v1 import CapturedLongTrainingBank
            bank=CapturedLongTrainingBank(ROOT/contract['data_archive']['path'],{n:contract['data_archive'][n] for n in ('bytes','sha256')})
            # Keep programs once, not the4800-event scene schedules in everycase.
            result['forcing_programs']={name:value['scene'].get('forcing_program') for name,value in bank.manifest['masters'].items()}
            with torch.inference_mode():
                for index,case in enumerate(contract['cases']):
                    require(time.monotonic()-result['began_monotonic']<new['worker_deadline_seconds'],'Scalar replay deadline exceeded')
                    scene=bank.manifest['masters'][case['identifier']]['scene'];native=bank.load(case['identifier'],case['start_fine_step'])
                    current={n:torch.from_numpy(native[n]).to('cuda') for n in FIELDS}
                    observer.set_phase('captured_original_causal_prediction')
                    outputs,work=predict_work(model,current,scene,case['start_fine_step'],case['output_steps_per_s'],case['K'])
                    require(len(outputs)==case['K'],'All Kphysical outputs required')
                    fields=[{n:v.detach().cpu().numpy().reshape(-1).copy() for n,v in output.items()} for output in outputs]
                    torch.cuda.synchronize();preceding_output=preceding_truth=native;identities=[]
                    for lead,(output,target_step) in enumerate(zip(fields,case['target_fine_steps'],strict=True),1):
                        prior=table[index,lead]
                        require({n:digest(v) for n,v in output.items()}==prior['output_hashes'],'Guided replay changed accepted physical fields')
                        truth=bank.load(case['identifier'],target_step)
                        metrics=physical_scores(output,truth,scene,normalization);require(metrics==prior['metrics'],'Guided replay changed accepted metrics')
                        h=1/scene['cells_per_metre'];stats=field_statistics(output,truth,native,scene['counts'],[h]*3,1000.,preceding_output,preceding_truth)
                        identity=dict(prior['identity'],output_role='endpoint' if lead==case['K'] else 'intermediate',case_role='physical_guided_output')
                        sink.write(fit_id,index,identity,stats,metrics)
                        identities.append(identity);preceding_output,preceding_truth=output,truth
                    state['rows'].append(dict(case_index=index,identities=identities,actual_work=work,
                        output_hashes=[table[index,lead]['output_hashes'] for lead in range(1,case['K']+1)]))
                    if index%60==0:
                        result['completed_blocks']=index+1;save(folder/'result.json',result)
                        print(fit_id,index+1,'/2880blocks',flush=True)
                require(guided_inventory(model)==selected,'Guided replay changed selected composition')
            result.update(residual_direct_CUDA_calls=direct,guide_direct_CUDA_calls=guide_direct,captured_imports=runtime.imported_sources())
            captured=result['captured_imports']
        finally:
            if model is not None:model.close()
            if bank is not None:bank.close()
    for name in captured:sys.modules.pop(name,None)
    require(not runtime.denied and not view_root.exists(),'No ambient access or retained code view allowed')
    result['temporary_code_view_removed']=True


def run(protocol_path,job_name):
    new=read(protocol_path);job=next(j for j in new['jobs'] if j['name']==job_name)
    old=read(ROOT/new['base_protocols'][job['kind']]['path']);accepted=read(ROOT/new['accepted_reports'][job['kind']]['path'])
    from component_material import current_publication
    current_publication(ROOT)
    for name,expected in old['source_sha256'].items():require(sha(ROOT/name)==expected,'Original scientificsource/input changed: '+name)
    for name,expected in new['admitted_lab_python'].items():require(sha(ROOT/name)==expected,'New scalar source changed: '+name)
    folder=ROOT/new['output_root']/job_name;require(not folder.exists(),'Fresh scalar worker required;no automatic retry')
    folder.mkdir(parents=True);os.sched_setaffinity(0,{job['host_cpu']});os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    result=dict(status='running_recent_component_scalar_replay',created_unix=time.time(),began_monotonic=time.monotonic(),pid=os.getpid(),
        job=job,protocol=binding(protocol_path),models=[],primary_timing_samples=0,latency_qualified=False,
        prediction_arrays_saved=False,new_parameter_fits=0,new_reference_generation=0,native_mutations=0)
    save(folder/'result.json',result);observer=None;sink=ScalarSink(folder,result)
    try:
        import numpy as np
        import torch
        from cuda_forward_observer import CUDAForwardObserver
        require(os.environ.get('TULDOK_TORCH_DEVICE')=='cuda','CUDA learned execution required')
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        require(torch.cuda.get_device_name()==new['expected_GPU'],'Expected GPU required')
        observer=CUDAForwardObserver(torch);observer.enable('CUDA_constructor_and_prediction')
        if job['kind']=='readout':readout_job(np,torch,old,accepted,result,sink,observer,folder)
        elif job['kind']=='correction':correction_job(np,torch,old,accepted,result,sink,observer,folder)
        else:guided_job(np,torch,old,accepted,result,sink,observer,folder,job['fit_item_id'],new)
        observer.close();require(not observer.forbidden_attempts,'No CPU/unclassified neural execution allowed')
        from python_source_coverage import collect
        coverage=collect(sys.modules,ROOT)
        for name in coverage['lab_python']:require(name in new['admitted_lab_python'] and sha(ROOT/name)==new['admitted_lab_python'][name],
            'Unbound actual lab source: '+name)
        result.update(status='all_original_component_outputs_exact_expanded_scalars_retained',source_coverage=coverage,
            device_observation=observer.evidence(),CPU_neural_forwards=0,GPU=torch.cuda.get_device_name(),Torch_version=torch.__version__,
            CUDA_version=torch.version.cuda,peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
            all_original_output_hashes_exact=True,all_original_accepted_metrics_exact=True)
    except BaseException as error:
        result.update(status='recent_component_scalar_replay_failed_requires_reconciliation',error_type=type(error).__name__,error=str(error));raise
    finally:
        sink.close()
        if observer is not None:observer.close();result['device_observation']=observer.evidence()
        result.update(wall_seconds=time.monotonic()-result['began_monotonic'],wall_is_accuracy_and_scoring_cost_not_inference_latency=True)
        save(folder/'result.json',result)
    print(job_name,'complete:',sink.counts,'scalar records,no fields.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--job',required=True)
    args=parser.parse_args();run(args.protocol.resolve(),args.job)
