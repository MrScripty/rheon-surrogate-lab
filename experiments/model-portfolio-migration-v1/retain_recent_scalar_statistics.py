"""Retain complete available scalar statistics for eight recent components."""
import json
from pathlib import Path
import time
import zipfile

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from portfolio_statistics import summarize, grouped, physical_metrics


def run():
    output=DOC/'reports/recent-component-scalar-statistics-01.json'
    require(not output.exists(), 'Preserve original scalar statistics authority')
    sources=[DOC/'reports'/name for name in ('macro-readout-field-GPU-accuracy-01.json',
        'internal-residual-GPU-accuracy-01.json','guided-GPU-accuracy-01.json')]
    guided_contract=read(RUN/'guided-component-contract-02.json')
    with zipfile.ZipFile(ROOT/guided_contract['data_archive']['path']) as archive:
        manifest=json.loads(archive.read('manifest.json'))
    guided_scenes={name:value['scene'] for name,value in manifest['masters'].items()}
    readout=read(RUN/'macro-readout-component-contract-01.json')
    cases={r['identifier']:r for r in readout['cases']}
    ordinary_scenes={name:read(ROOT/value['receipt'])['scene'] for name,value in cases.items()}
    models=[];predictions={}
    for source_path in sources:
        report=read(source_path)
        for model in report['models']:
            observations=[];missing_frequency=0
            for row in model['rows']:
                ident=dict(row['identity']);scene=(guided_scenes if 'start_fine_step' in ident else ordinary_scenes)[ident['identifier']]
                program=scene.get('forcing_program',{})
                ident.update(forcing_profile=program.get('profile','not_recorded'),
                    configured_pulse_frequency_hz=program.get('pulse_frequency_hz'),
                    motion_frequency_hz=program.get('motion_frequency_hz'),
                    variance_frequency_hz_components=program.get('variance_frequencies_hz'))
                if ident['configured_pulse_frequency_hz'] is None:missing_frequency+=1
                if source_path.name.startswith('macro-readout'):
                    ident.update(output_steps_per_s=60,output_dt_s=1/60,K=1,lead=1,internal_stages=model['stages'])
                observations.append(dict(identity=ident,values=physical_metrics(row)))
            selectors=dict(by_cells_per_metre=lambda r:r['identity']['cells_per_metre'],
                by_counts_xyz=lambda r:'x'.join(map(str,r['identity']['counts_xyz'])),
                by_cells_per_metre_and_counts=lambda r:str(r['identity']['cells_per_metre'])+'cpm/'+'x'.join(map(str,r['identity']['counts_xyz'])),
                by_output_steps_per_s=lambda r:r['identity'].get('output_steps_per_s'),
                by_K=lambda r:r['identity'].get('K'),by_lead=lambda r:r['identity'].get('lead'),
                by_forcing_profile=lambda r:r['identity']['forcing_profile'],
                by_configured_pulse_frequency_hz=lambda r:r['identity']['configured_pulse_frequency_hz'],
                by_motion_frequency_hz=lambda r:r['identity']['motion_frequency_hz'],
                by_variance_frequency_hz_components=lambda r:json.dumps(r['identity']['variance_frequency_hz_components'])
                    if r['identity']['variance_frequency_hz_components'] is not None else None)
            overall=summarize(observations)
            for name,value in model['overall']['worst_relative'].items():
                require(overall['metrics'][name+'_relative_l2']['worst']==value, 'Scalar summaries must exactly preserve every accepted worst error')
            spec=model.get('prediction_archive',model.get('worst_field_capture'))
            require(spec is not None, 'Original prediction disposition must be explicit')
            predictions[spec['path']]=spec
            models.append(dict(fit_item_id=model['fit_item_id'],source_report=binding(source_path),
                source_scalar_rows=len(observations),overall=overall,groups=grouped(observations,selectors),
                forcing_frequency_not_recorded_rows=missing_frequency,
                pressure_relative_error_definition='Mean-free relativeL2 with original diagnostic denominator floors',
                velocity_relative_error_definition='Combined staggered VX/VY/VZ relativeL2 with original diagnostic denominator floor',
                unavailable_full_population_metrics=['Maximum cell errors','Separate velocity-axis errors',
                    'Velocity divergence','Boundary violations','Tracer integral/overshoot','Own-history drift'],
                unavailable_policy='Unavailable metrics are not fabricated. Future prediction passes compute applicable scalar diagnostics in memory before discarding fields.',
                new_inference=False))
    require(len(models)==8 and len(predictions)==8, 'Exactly eight recent component fits and unique prediction files required')
    save(output,dict(status='eight_recent_components_scalar_statistics_verified_prediction_retirement_pending',
        created_unix=time.time(),producer=binding(Path(__file__)),statistics_helper=binding(Path(__file__).with_name('portfolio_statistics.py')),
        source_reports=[binding(p) for p in sources],models=models,prediction_archives_to_retire=list(predictions.values()),
        statistics=['best','worst','mean','median','p95','samples','best_case_identity','worst_case_identity'],
        population='Original per-output/per-request observations;not voxel-weighted andnot independent scenes. Independent scene counts reported separately.',
        frequency_semantics='Output cadence,configured pulse frequency,motion frequency and variance spectral components are separate. Irregular pulses do not imply one measured periodic frequency. Missing metadata remains not_recorded.',
        prediction_arrays_embedded=False,prediction_retirement_complete=False,
        model_dataset_or_project_copies=0,new_model_forwards=0,CPU_neural_forwards=0,
        native_mutations=0,full_portfolio_complete=False))
    print('Eight complete scalar populations summarized;best/worst/mean/median/p95 and density/grid/frequency/cadence groups verified.',flush=True)


if __name__=='__main__':run()
