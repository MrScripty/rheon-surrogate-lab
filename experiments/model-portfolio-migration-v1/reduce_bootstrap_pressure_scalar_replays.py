"""Verify both original GN fits, independent phases and scalar-only objectives."""
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from portfolio_statistics import distribution


def summarize(rows):
    names=('pressure_mean_free_relative_l2','pressure_absolute_rms','pressure_absolute_max','equation_relative_l2',
        'original_gauged_normalized_objective')
    return dict(cases=len(rows),metrics={name:dict(distribution(r['metrics'][name] for r in rows),
        minimum_case=min(rows,key=lambda r:r['metrics'][name])['identity'],
        maximum_case=max(rows,key=lambda r:r['metrics'][name])['identity']) for name in names})


def run():
    output=DOC/'reports/bootstrap-pressure-GPU-accuracy-01.json';require(not output.exists(),'Preserve first accepted bootstrap reduction')
    owner_path=RUN/'bootstrap-pressure-CUDA-owner-01/status.json';owner=read(owner_path)
    protocol_path=RUN/'bootstrap-pressure-CUDA-protocol-01.json';protocol=read(protocol_path)
    require(owner['status']=='all_four_bootstrap_pressure_CUDA_workers_terminal' and not owner['pending']
        and owner['active_child'] is None and len(owner['qualified'])==4 and owner['protocol']==binding(protocol_path),
        'Four terminal independent worker processes required')
    material=read(ROOT/protocol['material']['path']);entries={m['fit_item_id']:m for m in material['models']}
    phases={}
    for item in owner['qualified']:
        require(binding(ROOT/item['result']['path'])==item['result'],'Worker evidence changed')
        value=read(ROOT/item['result']['path'])
        require(value['status']=='bootstrap_pressure_CUDA_all_applicable_cases_qualified' and value['device_observation']['closed']
            and not value['device_observation']['forbidden_forward_attempts'] and not value['CPU_neural_forwards']
            and value['forward_variant']['all_operation_and_other_guard_AST_exact'],'Original guarded CUDA boundary required')
        phases[item['fit_item_id'],item['phase']]=(value,item['result'])
    models=[];objective_checks=0;all_inputs=None;hash_checks=0
    for fid in protocol['fit_ids']:
        first,first_spec=phases[fid,0];second,second_spec=phases[fid,1]
        a={r['case_index']:r for r in first['rows']};b={r['case_index']:r for r in second['rows']}
        require(len(a)==len(b)==770 and set(a)==set(b)==set(range(770)) and a==b,
            'All original inputs/targets/predictions/metrics must repeat exactly in reverse-order process')
        hash_checks+=770
        identities=[dict(case_index=i,input_hashes=a[i]['input_hashes'],reference_pressure_sha256=a[i]['reference_pressure_sha256'],
            reference_rhs_sha256=a[i]['reference_rhs_sha256']) for i in range(770)]
        require(all_inputs is None or all_inputs==identities,'Both original fits must see the exact same768native and2water requests')
        all_inputs=identities
        norm=read(ROOT/entries[fid]['outputs']['normalization.json']['path'])
        spec=first['scalar_statistics'];require(binding(ROOT/spec['path'])['sha256']==spec['sha256'],'Scalar body changed')
        seen=set();max_difference=0.
        with gzip.open(ROOT/spec['path'],'rt') as stream:
            for line in stream:
                row=json.loads(line);index=row['case_index'];s=row['statistics'];original=a[index]
                require(row['fit_item_id']==fid and index not in seen,'Scalar fit/case identity required')
                seen.add(index)
                count=s['reference_pressure']['values']
                # The original objective uses cell-zero gauge,not mean-free
                # pressure. Recover its raw squared norm from centred error
                # plus the separately retained scalar gauge offset.
                p2=s['pressure_mean_free_error']['l2']**2+count*s['pressure_gauge_offset']**2
                ep2=(1e-6*math.sqrt(count)*norm['scales']['pressure']['value'])**2
                eb2=(1e-6*math.sqrt(count)*norm['scales']['rhs']['value'])**2
                loss=.5*p2/max(s['reference_pressure']['l2']**2,ep2)+.5*s['equation_residual']['l2']**2/max(s['equation_residual']['reference_l2']**2,eb2)
                expected=original['metrics']['original_gauged_normalized_objective']
                require(math.isclose(loss,expected,rel_tol=1e-10,abs_tol=1e-12),'Independent NumPy/scalar original objective differs')
                max_difference=max(max_difference,abs(loss-expected));objective_checks+=1
        require(seen==set(a),'All scalar case records must roundtrip')
        native=[r for r in a.values() if r['identity']['population']=='bootstrap_native_union768']
        water=[r for r in a.values() if r['identity']['population']=='common_water128_applicable_subset']
        groups=defaultdict(list)
        for row in native:groups[row['identity']['training_exposure']].append(row)
        require(len(native)==768 and len(water)==2 and len(first['water_unsupported'])==126,'Complete supported andunsupported populations required')
        models.append(dict(fit_item_id=fid,parameters=372377,tensor_count=38,
            execution_variant=first['execution_variant'],selected_weights=first['selected_weights'],
            native_union=summarize(native),by_training_exposure={k:summarize(v) for k,v in groups.items()},
            common_water_applicable_subset=summarize(water),water_unsupported=first['water_unsupported'],
            scalar_statistics=spec,rows=[a[i] for i in range(770)],raw_results=[first_spec,second_spec],
            original_forward_AST_preserved_except_two_device_guard_literals=True,
            both_fresh_phase_rows_exact=True,
            current_CUDA_own_validation_objective=first['current_CUDA_own_validation_objective'],
            historical_CPU_own_validation_objective=first['historical_CPU_own_validation_objective'],
            historical_objective_absolute_difference=first['historical_objective_absolute_difference'],
            maximum_independent_original_objective_difference=max_difference,
            independent_original_objective_checks=770,latency_qualified=False,
            declared32cube_support_not_yet_exercised=True,own_history_drift=False))
    require(objective_checks==hash_checks==1540,'Both models/full independent objective andphase populations required')
    for fid in protocol['fit_ids']:
        for spec in [*entries[fid]['inputs'].values(),*entries[fid]['outputs'].values(),*entries[fid]['sources']]:
            require(binding(ROOT/spec['path'])['sha256']==spec['sha256'],'Original source/data/weights changed during qualification')
    save(output,dict(status='both_original_bootstrap_GN_pressure_fits_CUDA_qualified_all_applicable_populations',
        created_unix=time.time(),producer=binding(Path(__file__)),owner=binding(owner_path),protocol=binding(protocol_path),
        forward_AST_qualification=binding(DOC/'reports/bootstrap-device-boundary-AST-qualification-01.json'),models=models,
        independent_processes=4,unique_case_outputs=1540,physical_field_hash_checks=1540,
        exact_crossphase_complete_row_comparisons=1540,independent_original_objective_checks=1540,
        native_union_cases_per_fit=768,shared_water_supported_cases_per_fit=2,shared_water_unsupported_cases_per_fit=126,
        future_32cube_contract_qualification_pending=True,
        applicability='Originalsource permits exact16/32cubes only. All126otherwater shapes areexplicitlyunsupported;no inventedpressurefield orfull128comparison isclaimed.',
        physics='Twooriginal384frame nativecohorts preserve independentXYZspacing andfluiddensity1.25;thetwoapplicablewater requests use1000andstayaseparatepopulation.',
        exposure=protocol['exposure'],CPU_history_bitwise_equivalence_claim=False,
        metric_definitions='Mean-freepressurefieldL2/RMS/max,equationresidual,andoriginalgaugednormalizedobjectivearedistinct.SupplementaryL2floors1e-30areexplicitintheper-casescalars.',
        primary_timing_samples=0,latency_qualified=False,CPU_neural_forwards=0,new_parameter_fits=0,
        prediction_arrays_saved=False,new_reference_generation=0,prediction_files_deleted=0,
        deletion_paused=True,native_mutations=0,full_portfolio_complete=False))
    print('Two originalGNfits qualified;1540exactcrossphase cases and1540independentobjectivechecks;126unsupportedwatergeometries perfit explicit.',flush=True)


if __name__=='__main__':run()
