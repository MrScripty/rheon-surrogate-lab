"""Adopt verified scalar sidecars without changing benchmark qualifications."""
import copy
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, read, save, binding, require


def run():
    output = DOC/'reports/common-fit-benchmark-coverage-14.json'
    require(not output.exists(), 'Preserve original field-statistic adoption')
    previous_path = DOC/'reports/common-fit-benchmark-coverage-13.json'
    pressure_path = DOC/'reports/pressure-field-statistics-01.json'
    other_path = DOC/'reports/other-component-field-statistics-02.json'
    plot_path = DOC/'reports/plot-statistics-02.json'
    previous,pressure,other,plot = (read(p) for p in (previous_path,pressure_path,other_path,plot_path))
    inventory_path = DOC/'reports/prediction-capture-retention-inventory-02.json'
    inventory = read(inventory_path)
    for report in (pressure,other):
        for spec in (report['producer'],report['helper'],report['scalar_case_payload']):
            require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Accepted scalar extraction material changed')
    for spec in inventory['candidates']:
        path = ROOT/spec['path']
        require(path.stat().st_size == spec['bytes'] and binding(path)['sha256'] == spec['sha256'],
            'Existing prediction capture was changed/deleted during scalar extraction')
    require(pressure['fits'] == 150 and pressure['evaluated_case_outputs'] == 19200
        and pressure['accepted_metric_comparisons'] == 76800 and other['output_records'] == 6128
        and sum(other['accepted_metric_comparisons'].values()) == 19744
        and plot['additional_field_statistics_fits'] == 160, 'Complete accepted field statistic coverage required')
    rows = copy.deepcopy(previous['rows'])
    table = {r['fit_item_id']:r for r in rows}
    for model in pressure['models']:
        table[model['fit_item_id']]['retained_field_statistics'] = dict(source=binding(pressure_path),cases=128,
            scope='Pressure-only request: distributions/scales,pressure gradient,equation residual and supplied-base improvement',
            input_output_metadata='Captured input argument names,original force configuration,grid/cells_per_metre/fluid density/dt',
            prediction_capture_retained=True,retirement_ready=False,
            remaining='Per-model derived-feature/pass and downstream consumer audit;full-state effects require an identified generator composition')
    for fid,description in other['model_descriptions'].items():
        table[fid]['retained_field_statistics'] = dict(source=binding(other_path),output_records=other['output_records_by_fit'][fid],
            scope=description,local_stage_is_not_physical_lead=True,prediction_capture_retained=True,retirement_ready=False,
            remaining='Per-source interface/internal-work and downstream consumer audit;unsupported full-state quantities remain not_applicable')
    require(len(rows) == 205 and sum(r['default_benchmark'] is not None for r in rows) == 195
        and sum('retained_field_statistics' in r for r in rows) == 160, 'Scientific benchmark/fit counts must remain unchanged')
    result = dict(previous,created_unix=time.time(),producer=binding(Path(__file__)),previous_coverage=binding(previous_path),rows=rows,
        additional_field_statistics_fits=160,additional_field_statistics_outputs=25328,
        additional_field_statistics_metric_checks=96544,plotting_manifest=binding(plot_path),
        additional_field_statistics=[binding(pressure_path),binding(other_path)],
        prediction_retirement_inventory=binding(inventory_path),
        current_retention_policy='Statistics-only future benchmark adapters. Further prediction retirement paused pending expanded contract/consumer audit;154current captures unchanged after extraction. No copies of training/reference/model artifacts.',
        further_prediction_deletion_paused=True,prediction_files_deleted_this_adoption=0,
        source_contract=binding(DOC/'statistics-retention-contract.md'),
        native_statistics_publication_pending=True,
        evidence=[*previous['evidence'],binding(pressure_path),binding(other_path),binding(plot_path),binding(inventory_path)],
        new_model_forwards=0,native_mutations=0,full_portfolio_complete=False)
    save(output,result)
    print('Coverage14:203accuracy/195timed unchanged;160fits additional field statistics;all154captures unchanged and retained.',flush=True)


if __name__ == '__main__':
    run()
