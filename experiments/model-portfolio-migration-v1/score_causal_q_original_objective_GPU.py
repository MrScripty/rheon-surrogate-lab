"""Evaluate the original fixed training diagnostic on saved q outputs, CUDA only.

This evaluates no learned model. Its fixed pressure filter is an offline oracle,
not part of candidate inference or a native-F64 pressure accuracy measurement.
"""
from pathlib import Path
import math
import os
import statistics
import sys
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from modal_component_benchmark import imported, verify_sources
from component_material import observed_file
from cuda_forward_observer import CUDAForwardObserver


def summarize(rows):
    return dict(cases=len(rows),maximum_original_loss=max(row['original_loss'] for row in rows),
        mean_original_loss=statistics.mean(row['original_loss'] for row in rows),
        worst_teacher_filtered_relative_l2=max(row['teacher_filtered_relative_l2'] for row in rows),
        maximum_q_relative_l2_with_original_loss_floor=max(row['q_relative_l2_with_original_loss_floor'] for row in rows),
        pressure_reference_floor_cases=sum(row['pressure_reference_floor_active'] for row in rows),
        q_reference_floor_cases=sum(row['q_reference_floor_active'] for row in rows))


def run():
    output = DOC/'reports/causal-q-original-objective-GPU-01.json';require(not output.exists(), 'Preserve original objective diagnostic')
    benchmark_path = DOC/'reports/causal-request-component-GPU-benchmark-01.json';benchmark = read(benchmark_path)
    protocol_path = ROOT/benchmark['protocol']['path'];protocol = read(protocol_path);verify_sources(protocol)
    require(binding(protocol_path) == benchmark['protocol'] and os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda', 'Original accepted GPU output/protocol required')
    owner = read(ROOT/benchmark['owner']['path']);require(owner['active_child'] is None and not owner['pending']
        and not Path(f'/proc/{owner["pid"]}').exists(), 'Timed benchmark must be terminal before offlineGPUscoring')
    entries = {row['fit_item_id']:row for row in read(ROOT/benchmark['material']['path'])['models']}
    source = next(row for row in entries[benchmark['models'][0]['fit_item_id']]['sources'] if row['original_name'] == 'pressure_request_objective.py')
    for model in benchmark['models']:
        other = next(row for row in entries[model['fit_item_id']]['sources'] if row['original_name'] == 'pressure_request_objective.py')
        require(other['sha256'] == source['sha256'], 'Exact shared original training objective required')
    observed_file(ROOT,source['path'],source);producer = binding(Path(__file__))
    import numpy as np
    import torch
    from causal_request_component_data import load_original
    from modal_component_data import digest
    os.sched_setaffinity(0,{4});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.is_available() and torch.cuda.get_device_name() == protocol['expected_GPU'], 'Registered CUDA scoring device required')
    observer = CUDAForwardObserver(torch);observer.enable('offline_objective_no_model_forwards');models = [];began = time.monotonic()
    try:
        module = imported('original_causal_q_training_objective',ROOT/source['path'])
        _,_,norm,parents,caches = load_original(entries[benchmark['models'][0]['fit_item_id']]);del caches
        for parent in parents:
            del parent['velocity']
        with torch.inference_mode():
            for model in benchmark['models']:
                archive_spec = model['prediction_archive'];require(binding(ROOT/archive_spec['path']) == archive_spec, 'Accepted q fields changed')
                rows = []
                with np.load(ROOT/archive_spec['path'],allow_pickle=False) as archive:
                    for index,case in enumerate(model['rows']):
                        require(time.monotonic()-began < 600, 'Finite offline objective time bound exceeded')
                        parent,step = divmod(index,128)
                        target = parents[parent]['q'][step+1:step+2].unsqueeze(0)
                        saved = archive[f'case{index:04d}']
                        require(case['case_index'] == index and digest(saved) == case['output_sha256']
                                and digest(target.numpy()) == case['target_sha256'], 'Original saved q/target identities differ')
                        p = torch.from_numpy(saved.copy()).to('cuda');t = target.to('cuda');difference = p-t
                        loss = module.loss(p,t)
                        filtered = module.teacher_pressure(difference);reference = module.teacher_pressure(t)
                        floor = reference[0,0].numel()*1e-16
                        pressure_norm = reference.flatten(2).square().sum(2);q_norm = t.flatten(2).square().sum(2)
                        pressure_ratio = filtered.flatten(2).square().sum(2)/pressure_norm.clamp_min(floor)
                        q_ratio = difference.flatten(2).square().sum(2)/q_norm.clamp_min(floor)
                        values = dict(case_index=index,identity=case['identity'],original_loss=float(loss),
                            teacher_filtered_relative_l2=float(pressure_ratio.sqrt()[0,0]),
                            q_relative_l2_with_original_loss_floor=float(q_ratio.sqrt()[0,0]),
                            pressure_reference_floor_active=bool((pressure_norm < floor).any()),q_reference_floor_active=bool((q_norm < floor).any()))
                        require(all(math.isfinite(values[name]) for name in ('original_loss','teacher_filtered_relative_l2','q_relative_l2_with_original_loss_floor')),
                                'Finite original objective diagnostics required')
                        rows.append(values)
                seen = [row for row in rows if row['identity']['main_q_fit_seen']]
                unseen = [row for row in rows if not row['identity']['main_q_fit_seen']]
                visits = sum(row['identity']['main_q_fit_visits'] for row in rows);require(visits == 10000, 'Exact sampled main-loss weighting required')
                models.append(dict(fit_item_id=model['fit_item_id'],rows=rows,overall=summarize(rows),
                    sampled=summarize(seen),not_sampled=summarize(unseen),
                    current_checkpoint_loss_weighted_by_original_main_sample_visits=sum(row['original_loss']*row['identity']['main_q_fit_visits'] for row in rows)/visits,
                    original_main_sample_visits=visits))
                print(model['fit_item_id'],'originalfiltered worst',models[-1]['overall']['worst_teacher_filtered_relative_l2'],flush=True)
        torch.cuda.synchronize()
    finally:
        observer.close()
    require(not observer.calls and not observer.forbidden_attempts, 'Offline scoring must not execute learned modules')
    verify_sources(protocol);require(binding(Path(__file__)) == producer, 'Offline scoring producer changed')
    save(output,dict(status='original_q_training_objective_GPU_evaluated_on_all_saved_outputs',created_unix=time.time(),
        benchmark=binding(benchmark_path),original_objective_source=source,models=models,cases_per_fit=2048,
        original_loss_evaluations=4096,additional_pressure_filter_calls=8192,total_fixed_pressure_filter_calls=16384,
        inference_timing_includes_this_diagnostic=False,device='cuda',GPU=torch.cuda.get_device_name(),tf32=False,
        no_model_forward_evidence=observer.evidence(),CPU_neural_forwards=0,new_learned_GPU_forwards=0,new_parameter_fits=0,
        scope='OriginalF32fixedDCT/inverse training diagnostic onnormalizedF32q predictions/targets,includingoriginalRMS1e-8denominatorfloor. NotnativeF64pressure accuracy,notthecandidatepressuregenerator,notdrift,andnotinference latency.',
        sample_weighted_loss_scope='Currentselected checkpoint evaluatedwithoriginal10000main-sample visitweights;not its historical per-update training trajectory.',
        new_reference_simulations=0,weight_or_dataset_copies=0,producer=producer,full_portfolio_complete=False))


if __name__ == '__main__':
    run()
