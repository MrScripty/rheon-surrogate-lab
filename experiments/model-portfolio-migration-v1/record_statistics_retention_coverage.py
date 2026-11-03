"""Make scalar retention current without rewriting historical benchmark evidence."""
import copy
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, read, save, binding, require


def run():
    output=DOC/'reports/common-fit-benchmark-coverage-13.json'
    require(not output.exists(), 'Preserve original scalar retention coverage adoption')
    previous_path=DOC/'reports/common-fit-benchmark-coverage-12.json';previous=read(previous_path)
    statistics_path=DOC/'reports/recent-component-scalar-statistics-01.json';statistics=read(statistics_path)
    spatial_path=DOC/'reports/recent-component-spatial-statistics-01.json'
    retirement_path=DOC/'reports/recent-prediction-retirement-01.json';retirement=read(retirement_path)
    require(retirement['status']=='eight_verified_prediction_captures_retired_scalar_statistics_preserved'
            and retirement['deleted_prediction_files']==8 and retirement['deleted_prediction_bytes']==178352994
            and retirement['primary_statistics']==binding(statistics_path)
            and retirement['all_protected_material_identities_unchanged']
            and statistics['prediction_retirement_complete'], 'Verified statistics and completed derivative retirement required')
    require(all(not (ROOT/s['path']).exists() for s in retirement['captures']), 'Retired fields must remain absent')
    rows=copy.deepcopy(previous['rows']);lookup={r['fit_item_id']:r for r in rows}
    for model in statistics['models']:
        row=lookup[model['fit_item_id']]
        require(row.get('accuracy_benchmark') is not None, 'Only independently accepted accuracy rows may acquire scalar retention')
        row['accuracy_benchmark'].update(current_retention_policy='Scalar results and provenance only;prediction arrays intentionally retired',
            scalar_statistics=binding(statistics_path),spatial_statistics=binding(spatial_path),
            spatial_statistics_scope=model['spatial_statistics_scope'],prediction_retirement=binding(retirement_path),
            retained_prediction_arrays=False)
    require(len(rows)==205 and sum(r['default_benchmark'] is not None for r in rows)==195,
            'Retention changes must not alter scientific benchmark coverage')
    result=dict(previous,created_unix=time.time(),producer=binding(Path(__file__)),previous_coverage=binding(previous_path),
        rows=rows,current_retention_policy='No new bulk inference prediction arrays. Retain per-case scalar metrics,aggregate best/worst/mean/median/p95/counts and device/geometry/frequency/cadence/K/lead identities. Training/reference/model assets remain protected.',
        scalar_statistics=binding(statistics_path),spatial_statistics=binding(spatial_path),prediction_retirement=binding(retirement_path),
        new_prediction_arrays_retained=False,evidence=[*previous['evidence'],binding(statistics_path),binding(spatial_path),binding(retirement_path)],
        native_mutations=0,new_model_forwards=0,full_portfolio_complete=False)
    save(output,result)
    print('Coverage203accuracy/195timed preserved;current eight-fit retention is scalar-only.',flush=True)


if __name__=='__main__':run()
