"""Verify35complete scalar-only replays and summarize each physical lead."""
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from portfolio_statistics import distribution


def summarize(population):
    result={}
    for name,items in population.items():
        values=[v for v,_ in items]
        measured=distribution(values)
        measured['minimum']=measured.pop('best');measured['maximum']=measured.pop('worst')
        measured['population_std']=statistics.pstdev(values)
        measured['minimum_case']=min(items,key=lambda item:item[0])[1]
        measured['maximum_case']=max(items,key=lambda item:item[0])[1]
        result[name]=measured
    return result


def run():
    output=DOC/'reports/generator-field-statistics-01.json';require(not output.exists(),'Preserve first whole-generator scalar adoption')
    owner_path=RUN/'generator-scalar-replay-owner-01/status.json';owner=read(owner_path)
    require(owner['status']=='all35_generator_scalar_replays_terminal' and len(owner['qualified'])==35
        and not owner['pending'] and owner['active_child'] is None,'All35worker results must be terminal')
    protocol_path=RUN/'generator-scalar-replay-protocol-01.json';protocol=read(protocol_path)
    require(owner['protocol']==binding(protocol_path),'Owner protocol changed')
    original=read(ROOT/protocol['accepted_accuracy']['path']);accepted={m['fit_item_id']:m for m in original['models']}
    models=[];total=0;checks=0;old_scalars=0;spatial_checks=0
    for item in owner['qualified']:
        require(binding(ROOT/item['result']['path'])==item['result'],'Worker result changed')
        result=read(ROOT/item['result']['path']);fid=item['fit_item_id'];prior=accepted[fid]
        require(result['status']=='all64_generator_cases_exact_original_outputs_expanded_scalars_retained'
            and result['all_original_physical_output_hashes_exact'] and result['all_original_metric_scalars_exact']
            and result['device_observation']['closed'] and not result['device_observation']['forbidden_forward_attempts']
            and not result['CPU_neural_forwards'],'Complete exact guarded replay required')
        spec=result['scalar_statistics'];require(binding(ROOT/spec['path'])['sha256']==spec['sha256'],'Scalar body changed')
        rows={(r['case_index'],i['lead']):i for r in result['rows'] for i in r['identities']}
        expected={(index,m['lead']):m['values'] for index,r in enumerate(prior['pooled_rows']) for m in r['metrics']}
        overall=defaultdict(list);leads=defaultdict(lambda:defaultdict(list));geometry=defaultdict(lambda:defaultdict(list));seen=set()
        with gzip.open(ROOT/spec['path'],'rt') as stream:
            for line in stream:
                row=json.loads(line);identity=row['identity'];key=(row['case_index'],identity['lead'])
                require(row['fit_item_id']==fid and key not in seen and identity==rows[key],'Exact output identity/lead membership required')
                seen.add(key);values=row['statistics'];old=expected[key]
                for field in ('pressure','vx','vy','vz','tracer'):
                    error=values['fields'][field]['error']
                    for metric,name in (('absolute_rms','rms'),('relative_l2','relative_l2')):
                        previous=old[field+'_'+metric]
                        if previous is None:
                            require(error['denominator_floor_active'],'Original undefined relative error must remain explicitly floor-limited')
                        else:
                            require(math.isclose(error[name],previous,rel_tol=1e-11,abs_tol=1e-14),'New diagnostic field norm differs from accepted metric')
                        checks+=1
                    require(math.isclose(values['spatial'][field+'_absolute_rms_error'],error['rms'],rel_tol=1e-11,abs_tol=1e-14),
                        'Independent spatial andnewfield RMS disagree')
                    spatial_checks+=1
                for field,oldname in (('equation_residual','equation_relative_l2'),('combined_velocity_error','velocity_relative_l2')):
                    if old[oldname] is not None:
                        require(math.isclose(values[field]['relative_l2'],old[oldname],rel_tol=1e-10,abs_tol=1e-13),'New diagnostic norm differs from accepted equation/vector metric')
                    checks+=1
                focus={field+'_relative_l2':values['fields'][field]['error']['relative_l2'] for field in ('pressure','vx','vy','vz','tracer')}
                focus.update({field+'_absolute_mae':values['fields'][field]['error']['mae'] for field in ('pressure','vx','vy','vz','tracer')})
                focus.update(velocity_relative_l2=values['combined_velocity_error']['relative_l2'],
                    equation_relative_l2=values['equation_residual']['relative_l2'],
                    divergence_error_rms_per_s=values['spatial']['divergence_error_rms_per_s'],
                    boundary_normal_velocity_maximum_absolute=values['spatial']['boundary_normal_velocity_maximum_absolute'],
                    tracer_volume_integral_error=values['spatial']['tracer_volume_integral_error'],
                    kinetic_energy_error_joule=values['state_quantities']['kinetic_energy_error_joule'])
                group=str(identity['cells_per_metre'])+'cpm/'+'x'.join(map(str,identity['counts_xyz']))
                for name,value in focus.items():
                    overall[name].append((value,identity));leads[identity['lead']][name].append((value,identity));geometry[group][name].append((value,identity))
        require(seen==set(expected)==set(rows) and len(seen)==64*result['K'],'Every physical output must survive scalar roundtrip')
        models.append(dict(fit_item_id=fid,identifier=result['identifier'],K=result['K'],physical_outputs=len(seen),
            scalar_statistics=spec,raw_result=item['result'],overall=summarize(overall),
            by_physical_lead={str(lead):summarize(population) for lead,population in leads.items()},
            by_density_and_grid={name:summarize(population) for name,population in geometry.items()},
            source_work_evidence='Raw result rows containpartial source/Python andCUDA module counts;not independentpasses orper-lead latency',
            all_original_outputs_exact=True,latency_qualified=False))
        total+=len(seen);old_scalars+=result['published_metric_checks']
    require(len(models)==35 and total==5184 and checks==62208 and spatial_checks==25920,'Exact whole-generator diagnostic population required')
    save(output,dict(status='35original_generators_all5184physical_outputs_expanded_scalars_verified',created_unix=time.time(),
        producer=binding(Path(__file__)),owner=binding(owner_path),protocol=binding(protocol_path),
        original_benchmark=protocol['accepted_accuracy'],models=models,fit_populations=35,blocks=2240,physical_outputs=5184,
        original_field_hashes_reproduced=25920,original_metric_scalars_reproduced_exactly=old_scalars,
        additional_field_equation_vector_norm_checks=checks,independent_spatial_RMS_checks=spatial_checks,
        scalar_oracles_passed=7,scalar_test_source=binding(Path(__file__).with_name('test_full_state_scalar_diagnostics.py')),
        definitions=dict(pressure='Mean-free field error;raw gauge offset separately retained',
            velocity='Combined originalstaggered XYZandseparate axiserrors;cellcenteredspeed/direction supplementary',
            direction='Angle valid onlywhere bothcellcenteredspeeds exceed1e-30m/s;undefined counts retained',
            vorticity='Curl ofcellcenteredvelocity usingcentral interior/one-sided edge differences;not an additional solver output',
            energy='Cellcenteredvelocity,constant1000kg/m3 andcellvolume;referenceenergy retained,noconservation assumption underforcing',
            temporal='Everyphysicallead separately;changesfrominitialandpreviousphysicaloutput retained;single-block diagnostic,not own-history drift',
            denominator='SupplementaryrelativeL2uses1e-30normfloor,withflags.Originalpublishedzero-targetundefined policies remain inoriginalscalars.',
            timing='No qualified primarytiming intheseinstrumented/shared-GPU accuracyreplays;originalaccepted timing remains separate'),
        exposure=protocol['exposure'],reference_binary_sha256=protocol['reference_binary_sha256'],
        primary_timing_samples=0,new_parameter_fits=0,new_reference_generation=0,prediction_arrays_saved=False,
        prediction_files_deleted=0,deletion_paused=True,native_mutations=0,full_portfolio_complete=False))
    print('Verified35fits/5184outputs/25920exactfieldhashes;',checks,'norm checks and',spatial_checks,'independent spatial checks.',flush=True)


if __name__=='__main__':run()
