"""Join scalar sidecars to plotting identities without duplicating the tables."""
import csv
import gzip
import json
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, read, save, binding, require


def run():
    output = DOC/'reports/plot-statistics-02.json'
    require(not output.exists(), 'Preserve original scalar-sidecar adoption')
    base_path = DOC/'reports/plot-statistics-01.json'
    pressure_path = DOC/'reports/pressure-field-statistics-01.json'
    other_path = DOC/'reports/other-component-field-statistics-02.json'
    base,pressure,other = (read(p) for p in (base_path,pressure_path,other_path))
    for report in (base,pressure,other):
        spec = report['scalar_case_payload']
        require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Scalar sidecar changed')
    # Match exact request identities, not just a number that happens to be the
    # same case index in unrelated populations.
    expected = {m['fit_item_id'] for m in pressure['models']}
    pressure_cases = pressure['case_metadata']
    joined = set()
    all_base_models = {m['subject_id'] for m in base['models'] if m['subject_kind'] == 'fit_default_state'}
    base_case_keys = set()
    with gzip.open(ROOT/base['scalar_case_payload']['path'],'rt') as stream:
        for line in stream:
            row = json.loads(line)
            base_case_keys.add((row['subject_id'],row['case_index']))
            if row['subject_id'] not in expected:
                continue
            case = pressure_cases[row['case_index']]
            dims = row['dimensions']
            require(dims['identifier'] == case['identity']['trajectory_id']
                and dims['native_frame'] == case['identity']['native_frame']
                and dims['counts_xyz'] == case['identity']['counts_xyz']
                and dims['cells_per_metre'] == case['dimensions']['cells_per_metre']
                and dims['input_dt_s'] == case['dimensions']['input_dt_s'], 'Plot/scalar sidecar pressure case join differs')
            joined.add((row['subject_id'],row['case_index']))
    require(len(joined) == 19200, 'Every150pressure fit/case must join')
    for report in (pressure,other):
        count = 0
        with gzip.open(ROOT/report['scalar_case_payload']['path'],'rt') as stream:
            for line in stream:
                row = json.loads(line)
                require((row['fit_item_id'],row['case_index']) in base_case_keys,
                    'Supplementary scalar record has no original accepted case')
                if report is other and row['stage'] is not None:
                    require(row['stage'] in ('middle','end'), 'Local stage is not a physical lead')
                count += 1
        require(count == (19200 if report is pressure else 6128), 'Complete sidecar membership required')
    covered = expected | set(other['model_descriptions'])
    require(len(covered) == 160 and covered <= all_base_models, '160existing fits; no new fit or benchmark coverage claim')
    csv_path = DOC/'reports/C106-C107-pressure-by-density-01.csv'
    with csv_path.open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == 4 and {r['identifier'] for r in csv_rows} == {'C106','C107'}
        and {float(r['density']) for r in csv_rows} == {8.,16.}
        and all(int(r['samples']) == 64 for r in csv_rows), 'Actual joint model/density CSV must preserve64cases per point')
    save(output,dict(status='verified_scalar_sidecars_join160fit_populations_deletion_remains_paused',
        created_unix=time.time(),producer=binding(Path(__file__)),base_plot_manifest=binding(base_path),
        supplements=[dict(report=binding(pressure_path),join=['fit_item_id','case_index'],
            metrics='Pressure distributions, scales, equation residual, supplied-base comparison, correction and gradient errors',
            metadata='Join case_index to report.case_metadata for geometry,fluid mass density,input timestep,original force frequency/motion/amplitude and source exposure'),
            dict(report=binding(other_path),join=['fit_item_id','case_index','stage'],
                metrics='Task-specific distributions, target scales, local masks/carriers, input norms and per-stage sampler vectors',
                metadata='Join fit_item_id through model_descriptions.population to case_contexts;stage is an internal local target,not physical lead')],
        example_inspectable_CSV=binding(csv_path),scalar_query_tool=binding(Path(__file__).with_name('compare_pressure_statistics.py')),
        pressure_case_identity_joins=19200,additional_component_output_joins=6128,
        additional_field_statistics_fit_ids=sorted(covered),additional_field_statistics_fits=160,
        remaining_fit_ids_without_this_full_field_statistic_extraction=sorted(all_base_models-covered),
        remaining_count=43,
        coverage_limits='These160fits have additional task-specific field summaries,not automatic completion of all interface/pass/consumer/drift requirements. The35full generators and8recent components require further in-memory statistics/replay;some earlier spatial summaries already exist.',
        prediction_retirement_inventory=binding(DOC/'reports/prediction-capture-retention-inventory-02.json'),
        reference_epoch='Original old-Rheon populations only; no new solver reference or inferred new physical timestamp',
        old_default_qualified_accuracy_fits=203,old_default_qualified_timing_fits=195,
        new_model_forwards=0,field_copies_saved=0,prediction_files_deleted=0,deletion_paused=True,
        native_mutations=0,full_portfolio_complete=False))
    print('Verified160existing-fit scalar sidecars;19200pressure and6128component output joins;four64case CSVpoints.',flush=True)


if __name__ == '__main__':
    run()
