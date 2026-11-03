"""Managed shared-model development results and training-only guide ablations."""
import hashlib
import math
import os
from pathlib import Path
import statistics

FIELDS=('equation','pressure','velocity','tracer')


def enrich_temporal_successors(data,base,lab,load):
    base,lab=Path(base),Path(lab);docs=base/'full-ai-data-diversity-v1';root=lab/'runs/full-ai-data-diversity-v1'
    report_path=docs/'shared-temporal-dualwidth-training-results-01.json'
    if not report_path.exists():return
    hashes={}
    def sha(path):
        path=Path(path)
        if path not in hashes:
            with path.open('rb') as stream:hashes[path]=hashlib.file_digest(stream,'sha256').hexdigest()
        return hashes[path]
    def source(path):return os.path.relpath(path,base)
    def witness(value):
        for key in ('source_sha256','source_evidence','evidence_sha256','reference_sources'):
            evidence=value.get(key) or {}
            if not isinstance(evidence,dict):continue
            for name,expected in evidence.items():
                if isinstance(expected,str) and len(expected)==64 and sha(lab/name)!=expected:
                    raise ValueError('Temporal successor evidence changed: '+name)
    def read(path):
        value=load(Path(path));witness(value);return value
    protocols={sha(p):p for p in docs.glob('*protocol*.json')}
    report=read(report_path);pair_dir=root/'shared-temporal-recovery-pair-01';pair=read(pair_dir/'receipt.json')
    if pair['status']!='completed_managed_dualwidth_recovery_and_CPUrestore' or pair['recovered_parameter_fits']!=2 or pair['recovered_original_updates']!=1920 or pair['new_parameter_fits'] or pair['model_promotion']:
        raise ValueError('Dualwidth recovery custody differs')
    for filename,parity in pair['artifact_parity'].items():
        if not parity or sha(pair_dir/'train'/filename)!=sha(pair_dir/'restore'/filename):raise ValueError('Recovered shared artifacts differ')
    restore=read(pair_dir/'restore/receipt.json');history=read(pair_dir/'restore/history.json')
    if not restore['typed_cpu_weight_restore_verified'] or restore['cpu_model_forward_executed'] or restore['execution']['cuda_initialized']:
        raise ValueError('Shared typed CPU restoration scope differs')
    if history['initial_inventory']!=history['final_inventory'] or sha(pair_dir/'restore/weights.npz')!=sha(pair_dir/'restore/initial-weights.npz'):
        raise ValueError('Selected shared checkpoints are not their initial states')
    for width in ('4','8'):
        fit=report['fit_details'][width]
        if fit['Adam_updates']!=960 or fit['selected_epoch']!=0 or not fit['selected_parameters_equal_initial'] or history['original_updates_by_width'][width]!=960:
            raise ValueError('Shared fit selection or work count differs')
    data['training']=dict(source=source(report_path),pair_source=source(pair_dir/'receipt.json'),fits=report['fit_details'],
        new_parameter_fits=2,original_training_state=report['original_training_Run_state'],recovery_new_parameter_fits=report['recovery_new_parameter_fits'],
        failure=report['failure'],detailed_window_history_available=history['detailed_epoch_window_history_available'],reserved_opened=report['reserved52_references_opened'])

    def checked(path):
        value=read(path)
        if value.get('new_parameter_fits',0) or value.get('reserved52_references_opened',0) or value.get('cpu_model_forward',False) or value.get('model_promotion',False) or value.get('full_generator_target_pass',False):
            raise ValueError('Successor evaluation scope differs')
        protocol=protocols.get(value['protocol_sha256'])
        if protocol is None:raise ValueError('No exact registered successor protocol')
        reg=load(protocol,value['protocol_sha256']);witness(reg)
        return value,reg
    def append_point(capture,row,path,metrics,mode=None,samples=None,accuracy_source=None,equivalence=None):
        k=row['K'];rate=row['output_steps_per_s'];wall=samples if samples is not None else row['block_wall_ms']
        if len(metrics)!=k or not wall or any(t<=0 or not math.isfinite(t) for t in wall):raise ValueError('Missing successor output or timing')
        errors={f:max(m['relative'][f] for m in metrics) for f in FIELDS}
        if not all(math.isfinite(e) and e>=0 for e in errors.values()):raise ValueError('Nonfinite successor errors')
        if not math.isclose(row.get('block_duration_s',k/rate),k/rate,rel_tol=1e-12):raise ValueError('Successor physical block duration differs')
        median=statistics.median(wall);maximum=max(wall)
        if samples is None:
            if not math.isclose(row.get('median_block_ms',row.get('median_ms_per_physical_output',median/k)*k),median,rel_tol=1e-12):raise ValueError('Successor median differs')
            if not math.isclose(row.get('maximum_block_ms',row.get('maximum_ms_per_physical_output',maximum/k)*k),maximum,rel_tol=1e-12):raise ValueError('Successor maximum differs')
            if not row['all_outputs_owned_and_caller_unchanged']:raise ValueError('Successor output ownership failed')
        width=row.get('width',4);mode=mode or row.get('mode',row.get('checkpoint'))
        point=dict(capture=capture,identifier=row['identifier'],split=row.get('split','train'),counts_xyz=row['counts_xyz'],cells_per_metre=row['cells_per_metre'],
            start_fine_step=row['start_fine_step'],cadence=rate,K=k,mode=mode,backend='recorded_gpu',hidden_width=width,
            neural_parameters=1724 if width==4 else 3432,internal_advances=None,coarse_endpoint_transports=None,encoder_calls=None,pressure_projections=None,
            field_worst=errors,accuracy_pass=all(m['all_below_1e5'] for m in metrics),scored_outputs=k,wall_samples_ms=wall,event_samples_ms=None,
            median_block_ms=median,maximum_block_ms=maximum,source=source(path),accuracy_source=source(accuracy_source or path),
            work_description='Shared forecast and recorded control/pressure guides; additional fine readouts count as work, not extra K outputs.')
        if capture in ('trainedDev','guidedDev'):
            if not row['managed_CPUrestore_certified'] or len(row['output_sha256'])!=k:raise ValueError('Development checkpoint/output custody differs')
        if capture=='trainedDev':
            point['work_description']='One coarse endpoint/shared encoder; all K physical output heads.'
            point['mode_label']='selected checkpoint · initial epoch 0' if mode=='selected' else 'last trained checkpoint · epoch 4'
        if equivalence:point['equivalence_source']=source(equivalence)
        data['points'].append(point)
    captures=[('trainedDev','Managed shared fits · all 12 development masters','trained-shared-development-evaluation-01',False),
        ('guideInitial','Control-guide integration · initial training windows','temporal-control-guide-integration-02',True),
        ('guideFull','Control-guide ablation · all 24 training masters','temporal-control-guide-full-training-01',True),
        ('fineInitial','Fine-readout integration · initial training windows','predicted-fine-readout-integration-01',True),
        ('fineFull','Fine-readout ablation · all 24 training masters','predicted-fine-readout-full-training-01',True)]
    fine_scores={}
    for identifier,label,folder,ablation in captures:
        path=root/folder/'results.json'
        if not path.exists():continue
        value,reg=checked(path)
        if not value['status'].startswith('completed_'):continue
        data['captures'].append(dict(id=identifier,label=label,source=source(path),qualification_source=source(path),status=value['status'],
            scope=value['scope'],event_scope='No CUDA-event samples recorded in this capture.',untrained=False,historical=False,
            resource_note='Training-design ablation on fixed selected checkpoint; no optimizer updates.' if ablation else 'Development data used for evaluation/selection; not untouched reserved confirmation.',
            gpu_model=value.get('gpu_name','NVIDIA GeForce RTX 5090 Laptop GPU'),ablation=ablation))
        for row in value['rows']:
            if len(row['block_wall_ms'])!=reg['repetitions']:raise ValueError('Incomplete successor repeats')
            append_point(identifier,row,path,row['native_metrics_per_output'])
            if identifier=='fineFull' and row['mode']=='preoutput_fine_fullstate_moment_tracer':
                fine_scores[(row['identifier'],row['start_fine_step'],row['output_steps_per_s'],row['K'])]=row['native_metrics_per_output']
    folded_path=root/'folded-fine-readout-parity-01/results.json'
    if folded_path.exists():
        value,reg=checked(folded_path)
        if value['status']!='passed_frozen_zero_head_complete_output_byte_parity' or not value['zero_head_certificate']['all_exact_zero'] or value['native_future_fields_read']:
            raise ValueError('Folded zero-head parity scope differs')
        accuracy_path=root/'predicted-fine-readout-full-training-01/results.json'
        # This qualification records actual original/folded byte equality on
        # every matching case of the independently native-scored composite.
        data['captures'].append(dict(
            id='folded',label='Paired zero-head folding · all 24 training masters',source=source(folded_path),qualification_source=source(accuracy_path),
            status=value['status'],scope=value['scope'],event_scope='No CUDA-event samples recorded.',untrained=False,historical=False,ablation=True,
            resource_note='Fixed zero-head checkpoint only. Accuracy inherited from the matched native-scored original under full output-byte parity.',gpu_model=value['gpu_name']))
        for row in value['rows']:
            if not row['all_complete_output_bytes_equal'] or len(row['output_sha256'])!=row['K']:raise ValueError('Folded actual output parity differs')
            key=(row['identifier'],row['start_fine_step'],row['output_steps_per_s'],row['K']);metrics=fine_scores[key]
            for mode,samples in row['block_wall_ms'].items():
                if len(samples)!=reg['repetitions']:raise ValueError('Folded repeat count differs')
                if not math.isclose(statistics.median(samples)/row['K'],row['median_GPU_ms_per_physical_output'][mode],rel_tol=1e-12):raise ValueError('Folded per-output time differs')
                append_point('folded',row,folded_path,metrics,mode=mode,samples=samples,accuracy_source=accuracy_path,equivalence=folded_path)
                data['points'][-1]['work_description']='Zero-head original/folded composite; K fine readouts and tracer moment transport retained. K physical outputs unchanged.'
    guided_path=docs/'guided-shared-qualification-results-01.json'
    if guided_path.exists():
        guided=read(guided_path)
        if guided['new_parameter_fits'] or guided['reserved52_references_opened'] or guided['model_promotion']:
            raise ValueError('Guided gradient qualification is not a model fit')
        data['guided_readiness']=dict(source=source(guided_path),rows=guided['rows'],new_parameter_fits=0,
            scope=guided['qualification_scope'],interpretation=guided['interpretation'])

    guided_fit_path=docs/'guided-shared-fit-results-01.json'
    if guided_fit_path.exists():
        fit=read(guided_fit_path);folder=root/'guided-shared-managed-pair-01';custody=read(folder/'receipt.json')
        if custody['status']!='completed_managed_guided_fit_and_four_checkpoint_CPUrestore' or custody['new_parameter_fits']!=1 or custody['optimizer_updates']!=8 or custody['accepted_steps']!=4 or custody['cpu_model_forward'] or custody['model_promotion']:
            raise ValueError('Guided managed fit custody differs')
        for filename,parity in custody['artifact_parity'].items():
            if not parity or sha(folder/'train'/filename)!=sha(folder/'restore'/filename):raise ValueError('Guided restored artifacts differ')
        restored=read(folder/'restore/receipt.json');history=read(folder/'restore/history.json');trained=read(folder/'train/receipt.json')
        if not restored['typed_cpu_weight_restore_verified'] or restored['cpu_model_forward_executed'] or restored['execution']['cuda_initialized'] or len(restored['restored_checkpoints'])!=4 or trained['model_family']!='guided_shared_residual_v3':
            raise ValueError('Guided typed restore/family differs')
        inventories=history['checkpoint_inventories'];selected=inventories['weights.npz']
        for filename in ('best-mean-weights.npz','last-evaluated-weights.npz'):
            if inventories[filename]!=selected:raise ValueError('Guided checkpoint alias inventory differs')
        if inventories['initial-weights.npz']==selected:raise ValueError('Guided fit did not change parameters')
        if fit['parameter_trial_updates']!=len(fit['training_trials']) or fit['accepted_steps']!=sum(i['accepted'] for i in fit['iterations']) or fit['full_generator_target_pass'] or fit['reserved52_references_opened']:
            raise ValueError('Guided fit trial counts/scope differ')
        data['guided_training']={k:fit[k] for k in ('new_parameter_fits','parameter_trial_updates','accepted_steps','initial_training_objective','mean_loss_percent_reduction','iterations','training_trials','checkpoint_aliases','alias_policy','interpretation')}
        data['guided_training'].update(source=source(guided_fit_path),pair_source=source(folder/'receipt.json'),model_family=trained['model_family'],
            selected_training_objective=fit['selected_training_score']['objective'],development={k:{a:b for a,b in v.items() if a!='stratified'} for k,v in fit['development'].items()})
        erratum=docs/'guided-source-family-annotation-erratum-01.json'
        if erratum.exists():data['guided_training']['family_annotation']=read(erratum)
        path=root/'guided-shared-development-evaluation-01/results.json';value,reg=checked(path)
        if value['status']!='completed_guided_all_development_native_accuracy_and_GPU_cost' or value['checkpoint_aliases']!=fit['checkpoint_aliases'] or set(value['representative_checkpoints'])!={'selected','initial'} or len(value['rows'])!=2880 or value['independent_confirmation']:
            raise ValueError('Guided development population/aliases differ')
        data['captures'].append(dict(id='guidedDev',label='Guided RMS fit · selected vs initial · all 12 development masters',source=source(path),qualification_source=source(guided_fit_path),
            status=value['status'],scope=value['scope'],event_scope='No CUDA-event samples recorded.',untrained=False,historical=False,ablation=False,
            resource_note='Two unique parameter states; best-mean and last are exact selected-checkpoint aliases, not additional measurements.',gpu_model=value['gpu_name']))
        for row in value['rows']:
            if len(row['block_wall_ms'])!=reg['repetitions'] or row['checkpoint'] not in ('selected','initial'):raise ValueError('Guided development measurement differs')
            append_point('guidedDev',row,path,row['native_metrics_per_output'])
            data['points'][-1].update(mode_label='guided RMS · '+('selected / best-mean / last (one parameter state)' if row['checkpoint']=='selected' else 'initial checkpoint'),
                work_description='Full coarse forecast, K fine/moment guides and trained RMS-scaled shared residual; K physical outputs.')

    minimal_path=root/'minimal-fine-readout-parity-01/results.json'
    if minimal_path.exists():
        value,reg=checked(minimal_path);summary_path=docs/'minimal-fine-readout-parity-results-01.json';summary=read(summary_path)
        if sha(minimal_path)!=summary['source_evidence_sha256'] or not summary['all_output_and_coarse_context_bytes_equal'] or value['native_future_fields_read'] or value['status']!='passed_fullstate_minimal_guide_complete_output_and_context_byte_parity' or not all(c['all_exact_zero'] for c in value['zero_head_certificates'].values()):
            raise ValueError('Minimal full-state guide parity differs')
        folded={(r['identifier'],r['start_fine_step'],r['output_steps_per_s'],r['K']):r for r in load(folded_path)['rows']}
        data['captures'].append(dict(id='minimal',label='Minimal full-state guide · paired GPU cost · all 24 training masters',source=source(minimal_path),qualification_source=source(summary_path),
            status=value['status'],scope=value['scope'],event_scope='No CUDA-event samples recorded.',untrained=False,historical=False,ablation=True,
            resource_note='Full-state + moment guide only; inherited native accuracy, identical outputs and coarse context. Not trained residual-model timing.',gpu_model=value['gpu_name']))
        data['minimal_guide']=dict(source=source(summary_path),median_paired_GPU_speedup=summary['median_paired_GPU_speedup'],scope=summary['scope'])
        for row in value['rows']:
            key=(row['identifier'],row['start_fine_step'],row['output_steps_per_s'],row['K'])
            if not row['all_complete_output_bytes_equal'] or not row['all_coarse_context_bytes_equal'] or row['output_sha256']!=folded[key]['output_sha256']:
                raise ValueError('Minimal guide output/context does not match scored parent')
            for mode,samples in row['block_wall_ms'].items():
                if len(samples)!=reg['repetitions'] or not math.isclose(statistics.median(samples)/row['K'],row['median_GPU_ms_per_physical_output'][mode],rel_tol=1e-12):raise ValueError('Minimal guide timing differs')
                append_point('minimal',row,minimal_path,fine_scores[key],mode=mode,samples=samples,accuracy_source=root/'predicted-fine-readout-full-training-01/results.json',equivalence=minimal_path)
                data['points'][-1]['work_description']='Full-state + moment guide only; unused pressure readouts removed in minimal. Fine/history/moment arithmetic and all K outputs retained.'
