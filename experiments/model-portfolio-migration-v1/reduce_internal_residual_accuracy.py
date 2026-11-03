"""Join both complete CUDA phases;replay retained worst fields without Torch."""
from collections import defaultdict
import hashlib
import io
import math
from pathlib import Path
import sys
import time
import zipfile

from import_common_generator_measurements import ROOT, DOC, RUN, read, save, binding, require
from component_material import observed_file
from macro_readout_fields import FIELDS, digest, physical_scores
from reduce_macro_readout_field_accuracy import summarize


def run():
    output=DOC/'reports/internal-residual-GPU-accuracy-01.json'
    require(not output.exists(), 'Preserve original correction reduction')
    protocol_path=RUN/'internal-residual-GPU-accuracy-protocol-02.json'
    protocol=read(protocol_path)
    contract=read(ROOT/protocol['contract']['path'])
    manifest=read(ROOT/contract['manifest']['path'])
    evaluation=read(ROOT/protocol['evaluation_normalization']['path'])
    paths=[ROOT/protocol['output_root']/f'phase{phase}/result.json' for phase in (0,1)]
    phases=[read(p) for p in paths]
    for phase,value in enumerate(phases):
        require(value['status']=='all_original_correction_requests_GPU_scored' and value['phase']==phase
                and value['cleanup_complete'] and not Path('/proc',str(value['pid'])).exists()
                and value['protocol']==binding(protocol_path) and value['completed_cases_per_fit']==783
                and value['CUDA_correction_requests']==value['frozen_CUDA_response_requests']==3132
                and value['original_weights_and_sources_unchanged']
                and value['module_forward_guard']['closed']
                and value['module_forward_guard']['forbidden_forward_attempts']==0
                and value['primary_timing_samples']==0 and not value['latency_qualified']
                and not value['future_targets_forwarded'] and value['CPU_neural_forwards']==0,
                'Both original CUDA component phases must be complete and terminal')
        require({m['fit_item_id'] for m in value['models']}==set(protocol['fit_ids'])
                and len(value['models'])==4, 'Four exact original component fits required')
    require(phases[1]['all_cross_process_fields_exact'], 'Every generated field must match reverse-order replay')
    for name,sha in protocol['source_sha256'].items():
        observed_file(ROOT,name,dict(sha256=sha))
    observed_file(ROOT,contract['archive']['path'],contract['archive'])
    import numpy as np
    models=[];maximum_difference=0.;worst_capture_cases=0
    with zipfile.ZipFile(ROOT/contract['archive']['path']) as corpus:
        for item in contract['models']:
            fid=item['fit_item_id']
            groups=[next(m for m in phase['models'] if m['fit_item_id']==fid) for phase in phases]
            require(groups[0]['weights']==groups[1]['weights']==item['weights']
                    and groups[0]['worst_field_capture']==groups[1]['worst_field_capture']
                    and groups[0]['retained_case_indices']==groups[1]['retained_case_indices'],
                    'Both workers must retain the same original weights and one worst-field capture')
            lookups=[{r['case_index']:r for r in g['rows']} for g in groups]
            require(all(len(g['rows'])==len(table)==783 and set(table)==set(range(783))
                        for g,table in zip(groups,lookups)), 'Complete distinct783request rows perfit required')
            rows=[]
            for index,case in enumerate(manifest['parts']):
                left,right=lookups[0][index],lookups[1][index]
                identity={name:case[name] for name in ('identifier','start','group_index','width','counts_xyz','cells_per_metre')}
                require(left==right and left['identity']==identity
                        and left['source_member_sha256']==case['sha256']
                        and math.isclose(left['physical_internal_group_dt_s'],case['width']/480,rel_tol=1e-14)
                        and left['targets_decoded_after_all_component_outputs'],
                        'Literal original identity,input/output hashes,metrics and label isolation must join both phases')
                for claimed in (left,right):
                    for name,original in claimed['original_Torch_relative'].items():
                        actual=claimed['metrics']['relative'][name]
                        difference=abs(actual-original)
                        maximum_difference=max(maximum_difference,difference)
                        require(math.isfinite(actual) and difference<=protocol['independent_metric_tolerance']['absolute']
                                +protocol['independent_metric_tolerance']['relative']*max(abs(actual),abs(original)),
                                'Independent full-population NumPy/original Torch scores disagree')
                rows.append(left)
            expected_worst=sorted({max(rows,key=lambda r:(r['metrics']['relative'][name],-r['case_index']))['case_index']
                                  for name in ('pressure','equation','velocity','tracer')})
            require(groups[0]['retained_case_indices']==expected_worst, 'Complete union of four actual worst cases must be retained')
            capture=groups[0]['worst_field_capture'];observed_file(ROOT,capture['path'],capture)
            with np.load(ROOT/capture['path'],allow_pickle=False) as fields:
                require(set(fields.files)=={f'case{index:03d}_{name}' for index in expected_worst for name in FIELDS},
                        'Exact retained worst-field array membership required')
                for index in expected_worst:
                    case=manifest['parts'][index]
                    with corpus.open(case['member']) as stream:raw=stream.read(case['bytes']+1)
                    require(len(raw)==case['bytes'] and hashlib.sha256(raw).hexdigest()==case['sha256'], 'Original worst-case corpus member changed')
                    with np.load(io.BytesIO(raw),allow_pickle=False) as stored:
                        target={name:stored['target_'+name].reshape(-1) for name in FIELDS}
                        target['rhs']=stored['target_rhs'].reshape(-1)
                        for name,value in target.items():
                            spec=case['arrays']['target_'+name]
                            require(value.dtype.str==spec['dtype'] and value.size==math.prod(spec['shape'])
                                    and np.isfinite(value).all(), 'Typed original target required')
                        predicted={name:fields[f'case{index:03d}_{name}'] for name in FIELDS}
                        require(all(v.dtype.str=='<f8' and v.shape==target[n].shape and np.isfinite(v).all()
                                    and digest(v)==lookups[0][index]['output_hashes'][n] for n,v in predicted.items()),
                                'Actual retained GPU worst-field hashes/ABI must match both processes')
                        measured=physical_scores(predicted,target,
                            dict(counts=case['counts_xyz'],cells_per_metre=case['cells_per_metre']),evaluation)
                        require(measured==lookups[0][index]['metrics'], 'Separate no-Torch worst-field reduction differs')
                    worst_capture_cases+=1
            selectors=dict(by_cells_per_metre=lambda r:str(r['identity']['cells_per_metre']),
                by_counts_xyz=lambda r:'x'.join(map(str,r['identity']['counts_xyz'])),
                by_cells_per_metre_and_counts=lambda r:str(r['identity']['cells_per_metre'])+'cpm/'+'x'.join(map(str,r['identity']['counts_xyz'])))
            grouped={}
            for name,select in selectors.items():
                partition=defaultdict(list)
                for row in rows:partition[select(row)].append(row)
                grouped[name]={key:summarize(value) for key,value in sorted(partition.items())}
            baseline=[dict(identity=row['identity'],metrics=row['archived_zero_correction_baseline']) for row in rows]
            history=read(ROOT/item['history']['path'])
            observed_file(ROOT,item['history']['path'],item['history'])
            with np.load(ROOT/item['weights']['path'],allow_pickle=False) as weights:
                output_head_tensors={name:int(np.count_nonzero(weights[name])) for name in weights.files
                                     if name.startswith(('velocity.','tracer.'))}
            models.append(dict(fit_item_id=fid,name=item['name'],weights=item['weights'],overall=summarize(rows),
                original_supplied_zero_correction_baseline=summarize(baseline),groups=grouped,rows=rows,
                cases_output_bytes_equal_supplied_baseline=sum(all(row['output_hashes'][name]
                    ==row['input_hashes']['base_'+name] for name in FIELDS) for row in rows),
                cases_metrics_equal_supplied_baseline=sum(row['metrics']==row['archived_zero_correction_baseline'] for row in rows),
                output_head_nonzero_tensor_elements=output_head_tensors,
                original_selection={name:history[name] for name in ('selected_epoch','selected_evaluation',
                    'selected_full148scene_objective','initial_full148scene_objective') if name in history},
                original_selection_objective_recomputed=False,
                paired_baseline_comparison={name:dict(improved=sum(row['metrics']['relative'][name]
                        <row['archived_zero_correction_baseline']['relative'][name] for row in rows),
                    worse=sum(row['metrics']['relative'][name]>row['archived_zero_correction_baseline']['relative'][name] for row in rows),
                    equal=sum(row['metrics']['relative'][name]==row['archived_zero_correction_baseline']['relative'][name] for row in rows))
                    for name in ('pressure','equation','velocity','tracer')},
                worst_field_capture=capture,retained_case_indices=expected_worst,
                supplied_internal_stage_context=True,complete_generator_accuracy_claim=False,
                original_raw_weights_and_F64_response_preserved=True,latency_qualified=False))
    require('torch' not in sys.modules, 'Final selected-field reducer must import no Torch/model runtime')
    result=dict(status='four_original_internal_residual_GPU_accuracies_independently_scored',created_unix=time.time(),
        producer=binding(Path(__file__)),protocol=binding(protocol_path),contract=protocol['contract'],
        failure_reconciliation=protocol['failure_reconciliation'],raw_results=[binding(p) for p in paths],models=models,
        fresh_GPU_processes=2,active_requests_per_fit=783,active_training_parents=29,
        active_current_states=261,fixed_bypass_parents=119,fixed_bypass_windows=1071,
        CUDA_correction_requests_across_processes=6264,frozen_CUDA_response_requests_across_processes=6264,
        generated_field_hashes_checked_across_processes=31320,
        full_population_independent_metric_pairs=25056,
        maximum_absolute_Torch_NumPy_metric_difference=maximum_difference,
        all_cross_process_input_output_and_metric_rows_exact=True,
        separate_no_Torch_worst_field_cases_reduced=worst_capture_cases,
        unique_worst_prediction_archive_bytes=sum(m['worst_field_capture']['bytes'] for m in models),
        full_prediction_corpus_retained=False,
        retention_scope='All request identities,source/input/output hashes and independent NumPy/original Torch physical scores retained. Only union of four metric-worst field cases perfit retained as arrays;second phase reuses these. Full population array retention is not claimed.',
        scoring_scope='Independent NumPy physical/operator scoring on every actual output within each GPU worker;captured original Torch loss independently agrees. This final no-Torch process verifies every score pair and independently reloads only retained worst fields.',
        scope=contract['comparison_scope'],field_precision='Original response pressure,velocity,tracer F64;raw correction subnet F32. No complete-generator carrier casts.',
        primary_timing_samples=0,latency_qualified=False,complete_generator_accuracy_claim=False,
        independent_holdout=False,own_history_drift=False,inference_device='cuda',CPU_neural_forwards=0,
        new_parameter_fits=0,new_reference_generation=0,model_dataset_or_project_copies=0,
        native_mutations=0,full_portfolio_complete=False)
    save(output,result)
    print(result['status'],flush=True)
    for model in models:print(model['fit_item_id'],model['overall']['worst_relative'],flush=True)


if __name__=='__main__':
    run()
