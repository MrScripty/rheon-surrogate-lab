"""Independently score saved causal fields; never import/execute a model."""
from collections import defaultdict
import math
from pathlib import Path
import sys
import time

from import_common_generator_measurements import ROOT, DOC, RUN, read, save, binding, require
from component_material import observed_file
from macro_readout_fields import FIELDS, digest, load_fields, physical_scores


PROTOCOL = RUN/'macro-readout-field-GPU-accuracy-protocol-01.json'


def summarize(rows):
    metrics = ('pressure','equation','velocity','tracer')
    return dict(cases=len(rows),worst_relative={name:max(row['metrics']['relative'][name] for row in rows) for name in metrics},
        worst_case_by_metric={name:max(rows,key=lambda row:row['metrics']['relative'][name])['identity'] for name in metrics},
        target_floor_cases={name:sum(row['metrics']['target_floor_active'][name] for row in rows) for name in metrics},
        worst_absolute_rms={name:max(row['metrics']['absolute_rms'][name] for row in rows)
                            for name in ('pressure','velocity','tracer')},
        worst_pressure_cell_zero_absolute_error=max(row['metrics']['pressure_cell_zero_absolute_error'] for row in rows),
        original_diagnostic_1e5_passes=sum(all(v <= 1e-5 for v in row['metrics']['relative'].values()) for row in rows),
        diagnostic_threshold_is_application_acceptance=False)


def run():
    output = DOC/'reports/macro-readout-field-GPU-accuracy-01.json'
    require(not output.exists(), 'Preserve original independent field reduction')
    protocol = read(PROTOCOL)
    contract = read(ROOT/protocol['contract']['path'])
    require(binding(ROOT/protocol['contract']['path']) == protocol['contract'], 'Original field contract changed')
    import numpy as np
    phase_paths = [ROOT/protocol['output_root']/f'phase{phase}/result.json' for phase in (0,1)]
    phases = [read(path) for path in phase_paths]
    for phase,value in enumerate(phases):
        require(value['status'] == 'all58windows_per_original_readout_GPU_fields_collected'
                and value['phase'] == phase and value['protocol'] == binding(PROTOCOL)
                and not Path('/proc',str(value['pid'])).exists() and value['cleanup_complete']
                and value['learned_backbone_and_raw_heads_unchanged'] and value['all_model_parameters_and_buffers_CUDA']
                and value['total_physical_outputs'] == value['direct_CUDA_readout_calls'] == 116
                and value['total_causal_internal_advances'] == 406 and value['primary_timing_samples'] == 0
                and not value['latency_qualified'] and not value['references_passed_to_causal_model']
                and value['module_forward_guard']['forbidden_forward_attempts'] == 0
                and value['module_forward_guard']['closed'] and value['all_QR_factor_diagnostics_bitwise_equal'],
                'Complete original CUDA phases,source controls,cleanup and explicit timing limits required')
        require(all(name in protocol['source_sha256'] for name in value['source_coverage']['lab_python']),
                'Actual imported model/helper source closure must be bound')
    require(phases[0]['original_frozen_tensor_inventory'] == phases[1]['original_frozen_tensor_inventory']
            and phases[0]['original_frozen_backbone_identity'] == phases[1]['original_frozen_backbone_identity'],
            'Both fresh workers must restore the same frozen causal stage assets')
    for name,sha in protocol['source_sha256'].items():
        if Path(name).is_absolute():
            import hashlib
            with Path(name).open('rb') as stream:
                require(hashlib.file_digest(stream,'sha256').hexdigest() == sha, 'Pinned external Torch namespace changed')
        else:
            observed_file(ROOT,name,dict(sha256=sha))
    norm = read(ROOT/protocol['normalization']['path'])
    models = []
    maximum_score_difference = 0.
    for item in protocol['models']:
        fid = item['fit_item_id']
        original = [next(row for row in value['models'] if row['fit_item_id'] == fid) for value in phases]
        require(original[0]['prediction_archive'] == original[1]['prediction_archive']
                and all(len(row['rows']) == 58 and row['weights'] == item['weights']
                        and not row['readout_coefficient_quantization_applied'] for row in original),
                'All active windows and one reused original raw-head prediction archive required')
        archive_spec = original[0]['prediction_archive']
        observed_file(ROOT,archive_spec['path'],archive_spec)
        lookup = [{row['case_index']:row for row in group['rows']} for group in original]
        require(all(set(table) == set(range(58)) for table in lookup), 'Exact active case indices required')
        rows = []
        with np.load(ROOT/archive_spec['path'],allow_pickle=False) as archive:
            require(set(archive.files) == {f'case{index:03d}_{name}' for index in range(58) for name in FIELDS},
                    'Complete retained full-field prediction archive required')
            for index,case in enumerate(contract['cases']):
                expected_identity = {name:case[name] for name in ('identifier','scene_index','counts_xyz','cells_per_metre','start','teacher_substeps')}
                left,right = lookup[0][index],lookup[1][index]
                require(left['identity'] == right['identity'] == expected_identity
                        and left['output_hashes'] == right['output_hashes']
                        and left['input_hashes'] == right['input_hashes']
                        and left['targets_loaded_after_causal_output'] and right['targets_loaded_after_causal_output']
                        and left['original_QR_factor_bitwise_equal'] and right['original_QR_factor_bitwise_equal']
                        and case['teacher_substeps'] == 8 and case['trainable_readout_active'],
                        'Literal original active scene/start/input/output/target isolation must join both phases')
                require(case['target_state']['step'] == case['start']+1
                        and math.isclose(case['target_state']['time_s']-case['current_state']['time_s'],1/60,abs_tol=1e-15),
                        'Each prediction must advance one full original physical output interval')
                native = load_fields(case['current_state'])
                truth = load_fields(case['target_state'],(*FIELDS,'rhs'))
                require({name:digest(value) for name,value in native.items()} == left['input_hashes'],
                        'Independent original current-field identities differ')
                scene = read(ROOT/case['receipt'])['scene']
                require(scene['counts'] == case['counts_xyz'] and scene['cells_per_metre'] == case['cells_per_metre'],
                        'Actual native scene units/geometry differ')
                predicted = {name:archive[f'case{index:03d}_{name}'] for name in FIELDS}
                for name,value in predicted.items():
                    require(value.dtype.str == ('<f8' if name == 'pressure' else '<f4')
                            and value.shape == truth[name].shape and np.isfinite(value).all()
                            and digest(value) == left['output_hashes'][name], 'Original full-field bytes/ABI differ')
                metrics = physical_scores(predicted,truth,scene,norm)
                for worker_row in (left,right):
                    claimed = worker_row['original_Torch_physical_scores']
                    for group in ('relative','absolute_rms'):
                        for name,value in claimed[group].items():
                            actual = metrics[group][name]
                            difference = abs(actual-value)
                            maximum_score_difference = max(maximum_score_difference,difference)
                            require(difference <= protocol['independent_metric_tolerance']['absolute']+
                                    protocol['independent_metric_tolerance']['relative']*max(abs(actual),abs(value)),
                                    'Independent physical metric differs: '+group+'/'+name)
                rows.append(dict(case_index=index,identity=expected_identity,metrics=metrics,
                    output_hashes=left['output_hashes'],original_QR_factor_bitwise_equal=True))
        grouped = {}
        selectors = dict(by_cells_per_metre=lambda row:str(row['identity']['cells_per_metre']),
            by_counts_xyz=lambda row:'x'.join(map(str,row['identity']['counts_xyz'])),
            by_cells_per_metre_and_counts=lambda row:str(row['identity']['cells_per_metre'])+'cpm/'+'x'.join(map(str,row['identity']['counts_xyz'])))
        for name,select in selectors.items():
            groups = defaultdict(list)
            for row in rows:
                groups[select(row)].append(row)
            grouped[name] = {key:summarize(value) for key,value in sorted(groups.items())}
        models.append(dict(fit_item_id=fid,stages=item['stages'],weights=item['weights'],
            prediction_archive=archive_spec,overall=summarize(rows),groups=grouped,rows=rows,
            raw_published_coefficients_unchanged=True,original_frozen_features_reproduced_exactly=True,
            complete_readout_and_companion_latency_pending=True))
    require('torch' not in sys.modules, 'Independent reduction must import no Torch/model runtime')
    result = dict(status='two_original_macro_readout_causal_field_GPU_accuracy_independently_reduced',
        created_unix=time.time(),producer=binding(Path(__file__)),protocol=binding(PROTOCOL),
        contract=protocol['contract'],raw_results=[binding(path) for path in phase_paths],models=models,
        fresh_GPU_processes=2,active_cases_per_fit=58,active_training_parents=29,
        fixed_bypass_training_parents=119,fixed_bypass_windows_per_fit=238,
        physical_outputs_checked_across_processes=232,causal_internal_advances_across_processes=812,
        direct_CUDA_readout_calls=232,original_QR_factor_bitwise_checks=232,
        output_field_hashes_checked_across_processes=1160,unique_prediction_archives=2,
        new_prediction_archive_bytes=sum(row['prediction_archive']['bytes'] for row in models),
        independent_metric_backend='NumPy;no Torch,model import or CPU learned forward',
        maximum_absolute_Torch_NumPy_metric_difference=maximum_score_difference,
        all_cross_process_input_and_output_hashes_exact=True,latency_qualified=False,primary_timing_samples=0,
        inference_device='cuda',CPU_neural_forwards=0,source_or_architecture_changes=False,
        independent_holdout=False,own_history_drift=False,new_parameter_fits=0,new_reference_generation=False,
        physical_output_dt_s=1/60,field_precision=dict(pressure='F64',velocity_tracer='F32'),
        scope='Original raw9/12coefficient readouts on29consumed N8training scenes/two starts,causal original3/4stage features.119N1parents bypass the head,not counted as learned field predictions. Original QR reconstruction is a separate source-control check;physical errors score actual final casts.',
        model_dataset_or_project_copies=0,default_timed_benchmark_coverage_increased=False,full_portfolio_complete=False)
    save(output,result)
    print(result['status'],'116active cases/232process outputs;latency pending.',flush=True)


if __name__ == '__main__':
    run()
