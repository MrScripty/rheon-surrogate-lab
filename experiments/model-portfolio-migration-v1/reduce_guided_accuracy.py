"""Verify both complete original guided repeats and re-score retained worst fields."""
from collections import defaultdict
import math
from pathlib import Path
import sys
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from component_material import observed_file
from guided_captured_runtime import GuidedCapturedRuntime
from macro_readout_fields import FIELDS, digest, physical_scores
from reduce_macro_readout_field_accuracy import summarize
import review_direction_pressure_fits


def run():
    output=DOC/'reports/guided-GPU-accuracy-01.json'
    require(not output.exists(), 'Preserve original guided accuracy reduction')
    protocol_path=RUN/'guided-GPU-accuracy-protocol-02.json';protocol=read(protocol_path)
    contract=read(ROOT/protocol['contract']['path'])
    for name,sha in protocol['source_sha256'].items():observed_file(ROOT,name,dict(sha256=sha))
    import numpy as np
    models=[];raw_paths=[];maximum_difference=0.;separate_cases=0
    for item in contract['models']:
        fid=item['fit_item_id']
        paths=[ROOT/protocol['output_root']/f'{fid}-phase{phase}/result.json' for phase in (0,1)]
        phases=[read(path) for path in paths];raw_paths.extend(paths)
        for phase,value in enumerate(phases):
            require(value['status']=='all2880original_guided_blocks_GPU_scored' and value['phase']==phase
                    and value['fit_item_id']==fid and value['protocol']==binding(protocol_path)
                    and value['cleanup_complete'] and not Path('/proc',str(value['pid'])).exists()
                    and value['completed_blocks']==2880 and value['physical_outputs']==7200
                    and value['original_saved_GPU_probe_output_hashes_exact']
                    and value['complete_original_objective_reproduced']
                    and value['module_forward_guard']['closed']
                    and value['module_forward_guard']['forbidden_forward_attempts']==0
                    and value['temporary_source_view_removed'] and not value['ambient_access_attempts']
                    and value['CPU_neural_forwards']==0 and value['primary_timing_samples']==0
                    and not value['latency_qualified'] and not value['future_targets_forwarded'],
                    'Both full original CUDA populations,probes,objectives and cleanup must qualify')
            meta=read(ROOT/item['manifest']['path'])
            for spec in value['captured_imports'].values():
                require(meta['source_manifest']['members'][spec['member']]==spec['sha256'], 'Every actual captured import must join original source archive')
        require(phases[0]['selected_composition_inventory']==phases[1]['selected_composition_inventory']
                and phases[0]['worst_field_capture']==phases[1]['worst_field_capture'], 'Same original composition and reused field capture required')
        tables=[{r['case_index']:r for r in p['rows']} for p in phases]
        require(all(len(p['rows'])==len(table)==2880 and set(table)==set(range(2880)) for p,table in zip(phases,tables)), 'Every2880case index must occur exactly once')
        rows=[];independent_objective=0.
        for index,case in enumerate(contract['cases']):
            left,right=tables[0][index],tables[1][index]
            require(left==right and left['future_targets_decoded_after_causal_outputs']
                    and len(left['outputs'])==case['K'], 'Every input/output/metric/work row must repeat exactly')
            identity=left['identity']
            require(identity['identifier']==case['identifier'] and identity['start_fine_step']==case['start_fine_step']
                    and identity['K']==case['K'] and identity['output_steps_per_s']==case['output_steps_per_s']
                    and math.isclose(identity['block_duration_s'],case['K']/case['output_steps_per_s'],rel_tol=1e-15),
                    'Original physical scene,start,rate,K and horizon must join')
            block_squared=0.
            for lead,value in enumerate(left['outputs'],1):
                require(value['lead']==lead and value['target_fine_step']==case['target_fine_steps'][lead-1]
                        and math.isclose(value['physical_lead_dt_s'],lead/case['output_steps_per_s'],rel_tol=1e-15),
                        'Each lead must advance one additional whole physical output interval')
                for name,original in value['original_Torch_relative'].items():
                    actual=value['metrics']['relative'][name];difference=abs(actual-original)
                    maximum_difference=max(maximum_difference,difference)
                    require(math.isfinite(actual) and difference<=protocol['independent_metric_tolerance']['absolute']+
                            protocol['independent_metric_tolerance']['relative']*max(abs(actual),abs(original)), 'Independent all-output physical metrics disagree')
                block_squared+=sum(v*v for v in value['metrics']['relative'].values())/case['K']
                rows.append(dict(case_index=index,lead=lead,identity=dict(identity,lead=lead,target_fine_step=value['target_fine_step'],
                    physical_lead_dt_s=value['physical_lead_dt_s']),
                    metrics=value['metrics'],output_hashes=value['output_hashes'],output_dtypes=value['output_dtypes'],
                    unchanged_guide_metrics=value['unchanged_guide_metrics']))
            starts={c['start_fine_step'] for c in contract['cases'] if c['identifier']==case['identifier']}
            independent_objective+=block_squared*(1/24)*(1/len(starts))/12
        require(len(rows)==7200, 'All7200physical outputs required')
        expected=phases[0]['original_selected_training_objective']
        require(math.isclose(independent_objective,expected,rel_tol=protocol['original_training_objective_tolerance']['relative'],
                abs_tol=protocol['original_training_objective_tolerance']['absolute']), 'Independent complete original equal-scene training objective differs')
        worst=sorted({(r['case_index'],r['lead']) for name in ('pressure','equation','velocity','tracer')
                      for r in [max(rows,key=lambda r:(r['metrics']['relative'][name],-r['case_index'],-r['lead']))]})
        require(phases[0]['retained_outputs']==[dict(case_index=i,lead=l) for i,l in worst], 'Retain complete union of four metric-worst outputs')
        capture=phases[0]['worst_field_capture'];observed_file(ROOT,capture['path'],capture)
        meta=read(ROOT/item['manifest']['path'])
        with GuidedCapturedRuntime(item,meta,contract,output,extra_read_paths=[*protocol['source_sha256'],capture['path']]) as runtime:
            from captured_long_training_bank_v1 import CapturedLongTrainingBank
            bank=CapturedLongTrainingBank(ROOT/contract['data_archive']['path'],{n:contract['data_archive'][n] for n in ('bytes','sha256')})
            try:
                with np.load(ROOT/capture['path'],allow_pickle=False) as fields:
                    require(set(fields.files)=={f'case{i:04d}_lead{l}_{n}' for i,l in worst for n in FIELDS}, 'Exact worst capture field membership required')
                    for index,lead in worst:
                        case=contract['cases'][index];scene=bank.manifest['masters'][case['identifier']]['scene']
                        truth=bank.load(case['identifier'],case['target_fine_steps'][lead-1])
                        row=tables[0][index]['outputs'][lead-1]
                        predicted={n:fields[f'case{index:04d}_lead{lead}_{n}'] for n in FIELDS}
                        require(all(v.dtype.str==row['output_dtypes'][n] and v.shape==truth[n].shape
                                    and np.isfinite(v).all() and digest(v)==row['output_hashes'][n] for n,v in predicted.items()),
                                'Original retained field hashes,types and shapes must match both fresh workers')
                        measured=physical_scores(predicted,truth,scene,read(ROOT/item['normalization']['path']))
                        require(measured==row['metrics'], 'Separate no-Torch retained-field score differs')
                        separate_cases+=1
                captured_data_modules=runtime.imported_sources()
            finally:bank.close()
        require(not runtime.denied and not runtime.root.exists(), 'Independent field reduction must remove its source-only view')
        for name in captured_data_modules:sys.modules.pop(name,None)
        selectors=dict(by_cells_per_metre=lambda r:str(r['identity']['cells_per_metre']),
            by_counts_xyz=lambda r:'x'.join(map(str,r['identity']['counts_xyz'])),
            by_output_steps_per_s=lambda r:str(r['identity']['output_steps_per_s']),by_K=lambda r:str(r['identity']['K']),
            by_lead=lambda r:str(r['lead']),
            by_rate_K_lead=lambda r:str(r['identity']['output_steps_per_s'])+'outputs_s/K'+str(r['identity']['K'])+'/lead'+str(r['lead']),
            by_spacing_grid_rate_K=lambda r:str(r['identity']['cells_per_metre'])+'cpm/'+'x'.join(map(str,r['identity']['counts_xyz']))
                +'/'+str(r['identity']['output_steps_per_s'])+'outputs_s/K'+str(r['identity']['K']))
        grouped={}
        for name,select in selectors.items():
            partition=defaultdict(list)
            for row in rows:partition[select(row)].append(row)
            grouped[name]={key:summarize(value) for key,value in sorted(partition.items())}
        original=read(ROOT/item['history']['path'])['selected_worst_score']
        overall=summarize(rows)
        for name,expected_worst in original['worst_relative'].items():
            require(math.isclose(overall['worst_relative'][name],expected_worst,rel_tol=1e-9,abs_tol=1e-12), 'Original selected training population worst score differs')
        target_groups=defaultdict(list)
        for row in rows:
            ident=row['identity'];target_groups[(ident['identifier'],ident['start_fine_step'],ident['output_steps_per_s'],row['lead'])].append(row)
        overlapping=[]
        for key,values in target_groups.items():
            if len(values)<2:continue
            require(len({v['identity']['target_fine_step'] for v in values})==1, 'Overlapping Kpredictions must have identical physical targets')
            overlapping.append(dict(identifier=key[0],start_fine_step=key[1],output_steps_per_s=key[2],lead=key[3],
                K_values=[v['identity']['K'] for v in values],all_output_hashes_equal=len({tuple(sorted(v['output_hashes'].items())) for v in values})==1,
                relative_error_by_K={str(v['identity']['K']):v['metrics']['relative'] for v in values},
                pressure_relative_error_range=max(v['metrics']['relative']['pressure'] for v in values)-min(v['metrics']['relative']['pressure'] for v in values)))
        models.append(dict(fit_item_id=fid,name=item['name'],weights=item['weights'],actual_model_module=item['actual_model_module'],
            captured_manifest_model_family=item['captured_model_family'],overall=overall,groups=grouped,rows=rows,
            endpoint_overall=summarize([r for r in rows if r['lead']==r['identity']['K']]),
            interior_overall=summarize([r for r in rows if r['lead']<r['identity']['K']]),
            same_physical_target_across_K=dict(groups=len(overlapping),groups_all_field_hashes_equal=sum(v['all_output_hashes_equal'] for v in overlapping),
                worst_pressure_error_spread_groups=sorted(overlapping,key=lambda v:v['pressure_relative_error_range'],reverse=True)[:10],
                scope='Same current state,scene,output rate andphysical target;different requested block lengths. Hash inequality is sensitivity,not itself an accuracy failure;error ranges compare each prediction against the same reference. No pairwise field-difference norm is claimed.'),
            unchanged_guide_overall=summarize([dict(identity=r['identity'],metrics=r['unchanged_guide_metrics']) for r in rows]),
            original_selected_training_objective=expected,independent_NumPy_training_objective=independent_objective,
            original_training_population_worst_reproduced=True,worst_field_capture=capture,retained_outputs=phases[0]['retained_outputs'],
            actual_work_by_block=[dict(case_index=r['case_index'],identity=r['identity'],actual_work=r['actual_work']) for r in phases[0]['rows']],
            original_captured_composition_unchanged=True,latency_qualified=False))
    require('torch' not in sys.modules, 'Independent final reduction must import no Torch/model runtime')
    save(output,dict(status='two_original_guided_full_populations_GPU_accuracy_independently_reduced',created_unix=time.time(),
        producer=binding(Path(__file__)),protocol=binding(protocol_path),contract=protocol['contract'],compaction=protocol['compaction'],
        failure_reconciliation=protocol['failure_reconciliation'],raw_results=[binding(p) for p in raw_paths],models=models,
        fresh_GPU_processes=4,training_parents=24,selected_current_windows=240,blocks_per_fit=2880,physical_outputs_per_fit=7200,
        CUDA_blocks_across_processes=11520,physical_outputs_across_processes=28800,
        generated_field_hashes_checked_across_processes=144000,all_cross_process_input_output_metric_and_work_rows_exact=True,
        original_probe_controls_exact=True,original_selected_training_objectives_and_worst_errors_reproduced=True,
        independent_original_Torch_NumPy_metric_pairs=115200,maximum_absolute_Torch_NumPy_metric_difference=maximum_difference,
        separate_no_Torch_worst_field_outputs_reduced=separate_cases,
        unique_worst_prediction_archive_bytes=sum(m['worst_field_capture']['bytes'] for m in models),
        full_prediction_corpus_retained=False,retention_scope=protocol['retained_prediction_policy'],
        work_observer_scope='Named Python phase_packet widths and selected _RestoredPressure.forward/emit entries observed during actual predictions. These are partial method counts,not all GPU kernel launches;do not sum overlapping pressure/emit entries as independent projections. Accuracy-only profiling overhead is excluded from any timing claim.',
        scoring_scope='Independent NumPy versus original captured Torch physical metrics on every generated output;final no-Torch process verifies every pair,reproduces whole original objectives/worst errors and reloads retained worst field outputs.',
        inference_device='cuda',CPU_neural_forwards=0,primary_timing_samples=0,latency_qualified=False,
        independent_holdout=False,own_history_drift=False,new_parameter_fits=0,new_reference_generation=0,
        model_dataset_or_project_copies=0,native_mutations=0,full_portfolio_complete=False))
    print('Both original guided fits:2880blocks/7200outputs each independently verified;no timing.',flush=True)
    for m in models:print(m['fit_item_id'],m['overall']['worst_relative'],flush=True)


if __name__=='__main__':run()
