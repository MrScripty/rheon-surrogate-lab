"""Small inspectable CSV views of verified statistics; no prediction arrays."""
import csv
from pathlib import Path

from import_common_generator_measurements import ROOT, DOC, read, save, binding, require


def write_csv(path, rows):
    require(not path.exists(), 'Preserve original CSV export')
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    output_dir = DOC/'reports'
    qualification = read(output_dir/'plot-statistics-qualification-01.json')
    require(qualification['fit_populations_exact'] == 203 and qualification['scalar_aggregate_checks'] == 6960,
        'Complete verified scalar export required')
    source = output_dir/'plot-statistics-01.json'
    manifest = read(source)
    models = {m['subject_id']:m for m in manifest['models']}
    overall_source = output_dir/'portfolio-scalar-statistics-01.json'
    overall_rows = []
    for model in read(overall_source)['models']:
        meta = models[model['fit_item_id']]
        for metric,summary in model['accuracy']['metrics'].items():
            overall_rows.append(dict(subject_id=model['fit_item_id'], name=model['name'],
                task=model['task'], comparison_group=meta['comparison_group'], case_population=meta['case_population'],
                inputs=meta['inputs'], outputs=meta['outputs'], metric=metric,
                samples=summary['samples'], best=summary['best'], worst=summary['worst'],
                mean=summary['mean'], median=summary['median'], p95=summary['p95'],
                qualified_GPU_timing=meta['qualified_timing'], timing_units=meta['timing_units'],
                source_path=model['accuracy_source']['path']))
    lead_source = output_dir/'multistep-lead-statistics-01.json'
    lead_rows = []
    for row in read(lead_source)['rows']:
        meta = models[row['subject_id']]
        summary = row['statistics']
        lead_rows.append(dict(subject_id=row['subject_id'],name=meta['name'],
            task=meta['task'], comparison_group=meta['comparison_group'],case_population=meta['case_population'],
            K=row['K'],output_dt_s=row['output_dt_s'],block_horizon_s=row['block_horizon_s'],
            lead=row['lead'],physical_lead_dt_s=row['physical_lead_dt_s'],output_role=row['output_role'],
            metric=row['metric'],samples=summary['samples'],best=summary['best'],worst=summary['worst'],
            mean=summary['mean'],median=summary['median'],p95=summary['p95'],
            worst_scene=summary['worst_case']['identifier'],
            source_path=meta['accuracy_source']['path']))
    overall_path = output_dir/'model-error-statistics-01.csv'
    lead_path = output_dir/'multistep-lead-statistics-01.csv'
    write_csv(overall_path,overall_rows)
    write_csv(lead_path,lead_rows)
    for path,expected in ((overall_path,len(overall_rows)),(lead_path,len(lead_rows))):
        with path.open(newline='') as stream:
            require(sum(1 for _ in csv.DictReader(stream)) == expected, 'CSV record membership differs')
    save(output_dir/'plot-statistics-csv-01.json',dict(status='inspectable_scalar_csv_exports_verified',
        producer=binding(Path(__file__)),sources=[binding(source),binding(overall_source),binding(lead_source)],
        overall_csv=binding(overall_path),overall_rows=len(overall_rows),lead_csv=binding(lead_path),lead_rows=len(lead_rows),
        units='Errors retain original metric definitions; horizons and intervals are seconds. No per-lead timing is inferred.',
        filtering='Filter comparison_group/case_population and metric definition before ranking; lead CSV pools original grids/densities. Use case payload for joint geometry/density filters.',
        deletion_paused=True,prediction_fields_opened=0,prediction_files_deleted=0,new_model_forwards=0))
    print('Verified inspectable CSV rows:',len(overall_rows),'overall,',len(lead_rows),'per-lead.',flush=True)


if __name__ == '__main__':
    run()
