"""Make all205selected-fit expanded scalar populations discoverable together."""
import copy
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, read, save, binding, require


def run():
    output=DOC/'reports/common-fit-benchmark-coverage-16.json';plot_output=DOC/'reports/plot-statistics-04.json'
    require(not output.exists() and not plot_output.exists(),'Preserve first complete selected-fit scalar adoption')
    old_path=DOC/'reports/common-fit-benchmark-coverage-15.json';recent_path=DOC/'reports/recent-component-field-statistics-01.json'
    old,recent=read(old_path),read(recent_path)
    require(recent['fit_populations']==8 and recent['output_records']==17648
        and recent['additional_RMS_checks']==70592 and recent['independent_spatial_RMS_checks']==88240,
        'All eight complete accepted diagnostic populations required')
    rows=copy.deepcopy(old['rows']);table={r['fit_item_id']:r for r in rows}
    for model in recent['models']:
        row=table[model['fit_item_id']]
        require(row.get('retained_field_statistics') is None and row.get('accuracy_benchmark') is not None,
            'Only eight accepted accuracy-only components gain expanded scalar coverage')
        row['retained_field_statistics']=dict(source=binding(recent_path),scalar_case_payload=model['scalar_statistics'],
            output_records=model['output_records'],input_scope=model['input_scope'],output_scope=model['output_scope'],
            per_physical_lead_statistics=model['task']!='correction',
            internal_stage_is_not_physical_lead=model['task']=='correction',
            accepted_metric_floors_preserved=True,retirement_ready=False,
            remaining='Source/interface/internal-work andconsumer/portableclosure audit;own-historydrift andunqualifiedvariants remain separate')
    require(len(rows)==205 and sum(r.get('retained_field_statistics') is not None for r in rows)==205
        and sum(r['default_benchmark'] is not None for r in rows)==195,'Selected-fit scientific/timing counts must remain exact')
    payloads=[]
    for row in rows:
        retained=row['retained_field_statistics']
        if 'scalar_case_payload' in retained:
            spec=retained['scalar_case_payload'];require(binding(ROOT/spec['path'])['sha256']==spec['sha256'],'Scalar payload changed')
            payloads.append(dict(fit_item_id=row['fit_item_id'],payload=spec))
    save(plot_output,dict(status='all205selected_fit_expanded_scalar_populations_indexed',created_unix=time.time(),
        producer=binding(Path(__file__)),base_plot_manifest=binding(DOC/'reports/plot-statistics-03.json'),
        supplement=dict(report=binding(recent_path),join=['fit_item_id','case_index','lead_when_physical'],
            scalar_payload_owner='models[].scalar_statistics'),
        selected_fit_accuracy_populations=205,selected_fit_expanded_statistic_populations=205,qualified_timing_populations=195,
        expanded_task_specific_output_records=49700,
        additional_internal_head_delta_error_records=12528,
        roles='Outputrecords includephysicaloutputs,pressure/operatorrequests,andlocalsampler/internaltargets. Theyarenot49700simulationsteps.',
        comparison_requirements=['Task andrequiredinputcontext','Originalreferencepopulation/physics/integrationscheme',
            'Metricunits,gauge,denominator/floor','GridXYZandspacing/density','Physicaloutputinterval,K,lead,horizon',
            'Actualinternalwork versus outputcount','Device,precision,timingboundary andqualification',
            'Train/development exposure andcoverage/failures'],
        source_metric_policy='Acceptedrelative metricskeeporiginalfloors. Supplementarynumeric1e-30relativeL2andphysicalstate statistics remainseparatelydefined.',
        future_graphs=['Within-taskerror/latencyfrontiers','Pairedcase/baselineimprovement','Per-leadendpoint/intermediateerrors',
            'Density/grid/aspectratio scaling','Cadence/K/horizon effects','Forcing/event breakdowns','Physicalconsistency diagnostics'],
        not_yet_complete=['Tenqualifiedtimings','All524states/intendedvariants/capabilities','Own-history600outputdrift',
            'Nativepublication/portablescalar-sourceclosure','Source/work/consumer audit beforefurtherpredictionretirement'],
        native_publication_pending=True,prediction_files_deleted=0,deletion_paused=True,new_model_forwards=0,full_portfolio_complete=False))
    result=dict(old,created_unix=time.time(),producer=binding(Path(__file__)),previous_coverage=binding(old_path),rows=rows,
        additional_field_statistics_fits=205,additional_field_statistics_outputs=49700,
        selected_fit_expanded_scalar_collection_complete=True,remaining_expanded_statistic_fit_ids=[],
        additional_field_statistics=[*old['additional_field_statistics'],binding(recent_path)],
        plotting_manifest=binding(plot_output),recent_component_expanded_diagnostics=binding(recent_path),
        evidence=[*old['evidence'],binding(recent_path),binding(plot_output)],
        further_prediction_deletion_paused=True,expanded_metric_contract_complete=False,
        native_statistics_publication_pending=True,new_model_forwards=0,native_mutations=0,full_portfolio_complete=False)
    save(output,result)
    print('Coverage16:205supportedfitaccuracies/205expandedscalarpopulations/195timings;49700task-specificoutputrecords.',flush=True)


if __name__=='__main__':run()
