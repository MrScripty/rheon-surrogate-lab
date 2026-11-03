"""Adopt35generator diagnostics andtwo supported-contract accuracy outcomes."""
import copy
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, read, save, binding, require


def run():
    output=DOC/'reports/common-fit-benchmark-coverage-15.json'
    plot_output=DOC/'reports/plot-statistics-03.json'
    require(not output.exists() and not plot_output.exists(),'Preserve first35generator/twofitadoption')
    old_path=DOC/'reports/common-fit-benchmark-coverage-14.json'
    generator_path=DOC/'reports/generator-field-statistics-01.json'
    bootstrap_path=DOC/'reports/bootstrap-pressure-GPU-accuracy-01.json'
    old,generators,bootstrap=(read(p) for p in (old_path,generator_path,bootstrap_path))
    require(generators['fit_populations']==35 and generators['physical_outputs']==5184
        and bootstrap['unique_case_outputs']==1540 and bootstrap['independent_processes']==4,
        'Complete accepted scalar/accuracy populations required')
    rows=copy.deepcopy(old['rows']);table={r['fit_item_id']:r for r in rows}
    for model in generators['models']:
        row=table[model['fit_item_id']]
        require(row['default_benchmark']['group']=='complete_state','Only already accepted full-state fits gain diagnostics')
        row['retained_field_statistics']=dict(source=binding(generator_path),output_records=model['physical_outputs'],
            scope='All64matched nativeblocks/everyphysicallead:fullP/XYZvelocity/tracer distributions,baselines,changes,physicaldiagnostics,source-scopedwork',
            scalar_case_payload=model['scalar_statistics'],full_population_spatial_statistics=True,
            per_lead_statistics=True,instrumented_replay_latency_qualified=False,retirement_ready=False,
            remaining='Exact source/work semantics,portableconsumer closure andown-history drift;original clean timing retained separately')
    for number,model in enumerate(bootstrap['models'],1):
        row=table[model['fit_item_id']]
        require(row['default_benchmark'] is None and row.get('accuracy_benchmark') is None,
            'Only original two pending fit accuracies may be newly accepted')
        row.update(status='supported_original_contract_GPU_accuracy_available_latency_pending',
            display_name='Legacy anisotropic GroupNorm pressure · cohort'+str(number),
            original_name_preserved=True,
            accuracy_benchmark=dict(group='bootstrap_anisotropic_pressure',cases=770,native_matched_cases=768,
                common_water_supported_cases=2,common_water_unsupported_cases=126,
                source='original_selected_GN_weights_device_boundary_only_CUDA_variant',
                independent_reduction=binding(bootstrap_path),scope='Bothoriginalnativecohorts;separateapplicablewater subset. Pressureonly;no common128complete comparison.',
                latency_qualified=False,independent_holdout=False,own_history_drift=False,native_publication_pending=True,
                worst_native=model['native_union']['metrics'],worst_water_subset=model['common_water_applicable_subset']['metrics']),
            retained_field_statistics=dict(source=binding(bootstrap_path),scalar_case_payload=model['scalar_statistics'],output_records=770,
                scope='Nativeanisotropicpressureplusseparatewater subset;fullscalarbody,baseline andsourceconditioning metadata',
                retirement_ready=False,prediction_arrays_saved=False))
    accuracy={r['fit_item_id'] for r in rows if r['default_benchmark'] is not None or r.get('accuracy_benchmark') is not None}
    timing={r['fit_item_id'] for r in rows if r['default_benchmark'] is not None}
    additional={r['fit_item_id'] for r in rows if r.get('retained_field_statistics') is not None}
    require(len(rows)==len(accuracy)==205 and len(timing)==195 and len(additional)==197,
        '205supportedaccuracy/195timed/197expanded-stat fit counts required')
    previous_plot=DOC/'reports/plot-statistics-02.json'
    save(plot_output,dict(status='205selected_fit_accuracy_contracts_197expanded_field_statistic_populations',
        created_unix=time.time(),producer=binding(Path(__file__)),base_plot_manifest=binding(previous_plot),
        supplements=[dict(report=binding(generator_path),join=['fit_item_id','case_index','lead'],
                units='Everyphysicaloutput;dt1/60,K1–4;joinidentitiesand source-scopeworkfromeachrawresult',
                scalar_payload_owner='models[].scalar_statistics'),
            dict(report=binding(bootstrap_path),join=['fit_item_id','case_index'],
                units='Pressure-only requests;nativeunion768andapplicablewatersubset2aredifferentphysics/inputpopulations',
                scalar_payload_owner='models[].scalar_statistics')],
        qualified_accuracy_fits=205,qualified_timing_fits=195,additional_field_statistics_fits=197,
        remaining_expanded_statistic_fit_ids=sorted(set(table)-additional),remaining_expanded_statistic_fits=8,
        every_generator_lead_retained=True,joint_latency_not_assigned_to_leads=True,
        bootstrap_disposition='126watergeometriesunsupportedperfit;32cubedeclaredbutnotyetexercised. Thesearevisiblecapabilitylimits,notzeroerrorpoints.',
        final_holdout_or_600step_drift_claim=False,native_publication_pending=True,
        prediction_files_deleted=0,deletion_paused=True,full_portfolio_complete=False))
    evidence=[*old['evidence'],binding(generator_path),binding(bootstrap_path),binding(plot_output)]
    result=dict(old,created_unix=time.time(),producer=binding(Path(__file__)),previous_coverage=binding(old_path),rows=rows,
        common_GPU_accuracy_fits=205,common_accuracy_pending_fits=0,pending_accuracy_fit_ids=[],pending_accuracy_by_task={},
        bootstrap_supported_contract_accuracy_fits=2,bootstrap_not_full128_shared_request_comparisons=True,
        new_bootstrap_accuracy_result=binding(bootstrap_path),
        additional_field_statistics_fits=197,
        additional_field_statistics_outputs=old['additional_field_statistics_outputs']+5184+1540,
        plotting_manifest=binding(plot_output),native_component_outcomes_pending=20,
        current_scope='Every205selecteddefaultfit nowhasGPUaccuracyonitsadmittedsupportedtask/population,withunsupportedcasesexplicit.195havequalifiedlatency. Thisdoesnotqualifyevery524state/variant/shape/rollout.',
        full524_state_or_variant_coverage_complete=False,
        supported_contract_accuracy_complete_for_selected_fit_identities=True,
        additional_field_statistics=[*old['additional_field_statistics'],binding(generator_path),binding(bootstrap_path)],
        full_generator_expanded_diagnostics=binding(generator_path),
        evidence=evidence,new_model_forwards=0,
        adoption_creates_no_new_model_forwards=True,
        native_statistics_publication_pending=True,native_mutations=0,full_portfolio_complete=False)
    save(output,result)
    print('Coverage15:205selectedfit supported-contract accuracies,195timings,197expanded-stat populations;8diagnosticpopulations remain.',flush=True)


if __name__=='__main__':run()
