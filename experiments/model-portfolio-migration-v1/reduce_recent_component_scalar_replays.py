"""Verify every retained scalar record for the final eightfit populations."""
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from portfolio_statistics import distribution


def summary(rows):
    result={}
    for name in sorted(set().union(*(r['values'] for r in rows))):
        population=[r for r in rows if name in r['values']]
        measured=distribution(r['values'][name] for r in population)
        measured['minimum']=measured.pop('best');measured['maximum']=measured.pop('worst')
        measured.update(minimum_case=min(population,key=lambda r:r['values'][name])['identity'],
            maximum_case=max(population,key=lambda r:r['values'][name])['identity'])
        result[name]=measured
    return dict(output_records=len(rows),metrics=result)


def run():
    output=DOC/'reports/recent-component-field-statistics-01.json';require(not output.exists(),'Preserve first final-eight population reduction')
    owner_path=RUN/'recent-component-scalar-replay-owner-01/status.json';owner=read(owner_path)
    protocol_path=RUN/'recent-component-scalar-replay-protocol-01.json';protocol=read(protocol_path)
    require(owner['status']=='all_four_recent_component_scalar_jobs_terminal' and not owner['pending'] and not owner['active']
        and len(owner['qualified'])==4 and owner['protocol']==binding(protocol_path),'All four owned scientific jobs must be terminal')
    models=[];total=0;norm_checks=0;spatial_checks=0;hash_checks=0;metric_fields=0;delta_records=0
    for qualified in owner['qualified']:
        spec=qualified['result'];require(binding(ROOT/spec['path'])==spec,'Worker result changed')
        result=read(ROOT/spec['path']);kind=result['job']['kind']
        accepted=read(ROOT/protocol['accepted_reports'][kind]['path'])
        old_models={m['fit_item_id']:m for m in accepted['models']}
        require(result['status']=='all_original_component_outputs_exact_expanded_scalars_retained'
            and result['all_original_output_hashes_exact'] and result['all_original_accepted_metrics_exact']
            and result['device_observation']['closed'] and not result['device_observation']['forbidden_forward_attempts']
            and result['CPU_neural_forwards']==0,'All output bytes/scores and guarded CUDA lifecycle must qualify')
        if kind=='guided':require(result['temporary_code_view_removed'],'Temporary scientific Python view must be removed')
        for model in result['models']:
            fid=model['fit_item_id'];old=old_models[fid]
            expected={(r['case_index'],r['lead'] if kind=='guided' else None):r for r in old['rows']}
            observed_hashes={(r['case_index'],identity['lead']):hashes for r in model['rows']
                for identity,hashes in zip(r['identities'],r['output_hashes'],strict=True)} if kind=='guided' else {
                    (r['case_index'],None):r['output_hashes'] for r in model['rows']}
            population=result['scalar_populations'][fid];payload=population['payload']
            require(binding(ROOT/payload['path'])['sha256']==payload['sha256'],'Retained scalar payload changed')
            seen=set();observations=[];lead_groups=defaultdict(list);grid_groups=defaultdict(list);role_groups=defaultdict(list)
            with gzip.open(ROOT/payload['path'],'rt') as stream:
                for line in stream:
                    row=json.loads(line);identity=row['identity'];key=(row['case_index'],identity.get('lead') if kind=='guided' else None)
                    require(row['fit_item_id']==fid and key not in seen and key in expected,'Exact source-case membership required')
                    seen.add(key);prior=expected[key];stats=row['statistics'];measured=row['accepted_metrics']
                    require(measured==prior['metrics'] and observed_hashes[key]==prior['output_hashes'],'Every accepted field hash/metric must join')
                    hash_checks+=5;metric_fields+=len(measured['relative'])
                    for field in ('pressure','tracer'):
                        require(math.isclose(stats['fields'][field]['error']['rms'],measured['absolute_rms'][field],rel_tol=1e-11,abs_tol=1e-12),
                            'New field RMS disagrees with accepted score')
                        norm_checks+=1
                    for name,path in [('velocity',stats['combined_velocity_error']),('equation',stats['equation_residual'])]:
                        expected_rms=measured['absolute_rms'][name] if name=='velocity' else measured['equation_absolute_rms']
                        require(math.isclose(path['rms'],expected_rms,rel_tol=1e-10,abs_tol=1e-12),'New vector/equation RMS differs')
                        norm_checks+=1
                    for field in ('pressure','vx','vy','vz','tracer'):
                        require(math.isclose(stats['fields'][field]['error']['rms'],stats['spatial'][field+'_absolute_rms_error'],rel_tol=1e-11,abs_tol=1e-12),
                            'Independent spatial/field RMS mismatch')
                        spatial_checks+=1
                    values={name+'_accepted_relative_l2':value for name,value in measured['relative'].items()}
                    values.update({name+'_absolute_MAE':stats['fields'][name]['error']['mae'] for name in ('pressure','vx','vy','vz','tracer')})
                    values.update(divergence_error_rms_per_s=stats['spatial']['divergence_error_rms_per_s'],
                        boundary_normal_velocity_maximum_absolute=stats['spatial']['boundary_normal_velocity_maximum_absolute'],
                        tracer_volume_integral_error=stats['spatial']['tracer_volume_integral_error'],
                        kinetic_energy_error_joule=stats['state_quantities']['kinetic_energy_error_joule'])
                    if kind=='correction':
                        require('lead' not in identity and 'K' not in identity
                            and identity['case_role']=='supplied_internal_stage_context'
                            and 'input' not in stats['fields']['pressure']
                            and 'prediction_minus_preceding_output' not in stats['fields']['pressure'],
                            'Supplied stage context must not become a physical input/lead claim')
                        deltas=row['component_specific_statistics']['scaled_carrier_delta_errors']
                        require(set(deltas)=={'vx','vy','vz','tracer'},'All four actual head output targets retained')
                        for name,delta in deltas.items():values['scaled_delta_'+name+'_relative_l2']=delta['error']['relative_l2']
                        delta_records+=4
                        baseline=prior['archived_zero_correction_baseline']
                        for name in measured['relative']:values[name+'_accepted_improvement_vs_supplied_baseline']=baseline['relative'][name]-measured['relative'][name]
                    if kind=='guided':
                        require(identity['lead'] in range(1,identity['K']+1)
                            and identity['output_role']==('endpoint' if identity['lead']==identity['K'] else 'intermediate'),
                            'Every physical lead and endpoint/intermediate role retained')
                        baseline=prior['unchanged_guide_metrics']
                        for name in measured['relative']:values[name+'_accepted_improvement_vs_guide']=baseline['relative'][name]-measured['relative'][name]
                    observation=dict(identity=identity,values=values);observations.append(observation)
                    grid_groups[str(identity['cells_per_metre'])+'cpm/'+'x'.join(map(str,identity['counts_xyz']))].append(observation)
                    if 'lead' in identity:lead_groups[str(identity.get('output_steps_per_s',60))+'outputs_s/K'+str(identity['K'])+'/lead'+str(identity['lead'])].append(observation)
                    if 'output_role' in identity:role_groups[identity['output_role']].append(observation)
            require(seen==set(expected) and len(seen)==population['records'],'Full expected scientific population must roundtrip')
            models.append(dict(fit_item_id=fid,task=kind,output_records=len(seen),scalar_statistics=payload,
                raw_worker_result=spec,original_accuracy_source=protocol['accepted_reports'][kind],
                overall=summary(observations),by_density_and_grid={k:summary(v) for k,v in grid_groups.items()},
                by_physical_rate_K_lead={k:summary(v) for k,v in lead_groups.items()},
                by_endpoint_intermediate_role={k:summary(v) for k,v in role_groups.items()},
                input_scope=model['input_scope'],output_scope=model['output_scope'],
                unchanged_original_fields_and_metrics=True,latency_qualified=False))
            total+=len(seen)
    require(len(models)==8 and total==17648 and norm_checks==70592 and spatial_checks==88240
        and delta_records==12528,'Exact complete eightfit diagnostic population required')
    save(output,dict(status='all_eight_remaining_fit_field_statistic_populations_verified',created_unix=time.time(),
        producer=binding(Path(__file__)),owner=binding(owner_path),protocol=binding(protocol_path),models=models,
        fit_populations=8,output_records=total,original_field_hashes_reproduced=hash_checks,
        original_accepted_relative_metric_values_exact=metric_fields,additional_RMS_checks=norm_checks,
        independent_spatial_RMS_checks=spatial_checks,scaled_carrier_delta_error_records=delta_records,
        definitions=dict(relative_metrics='Use accepted_metrics.relative for original train-derived RMS/equation floors. Supplementary fields.error.relative_l2 uses explicit numeric1e-30L2floor;not interchangeable.',
            correction='Suppliedzero-correctionstage baseline isnotinitialphysicalstate. NoK/physicallead orstandaloneadvance claim. Fouractualscaledhead deltas separatelyscored.',
            readout='Original causalS+1stagefeatures,onephysical1/60soutput. S=3/4not3/4futureframes.',
            guided='EveryphysicalKoutput at30/60/120outputs_per_s,initialandprecedingphysicalstatechanges,endpoint/intermediategroups;unchangedguide comparisons retained.',
            work='OriginalpartialPython/modulecounter scopes;notallkernels orindependentpasses;nostagecostinvented.',
            physics='All original referenceepochs,geometry,cells/metre andscenecontrols preserved;no newreferencegeneration orholdoutclaim.'),
        primary_timing_samples=0,latency_qualified=False,prediction_arrays_saved=False,prediction_files_deleted=0,
        new_parameter_fits=0,new_reference_generation=0,deletion_paused=True,native_mutations=0,full_portfolio_complete=False))
    print('Eightfits/17648records verified;',norm_checks,'accepted RMS and',spatial_checks,'independent spatial checks;no field files.',flush=True)


if __name__=='__main__':run()
