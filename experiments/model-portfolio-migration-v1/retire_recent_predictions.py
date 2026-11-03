"""Retire eight verified derivative field captures after scalar preservation."""
import os
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from component_material import observed_file


def identity(path):
    value=path.lstat()
    return [value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns]


def run():
    report_path=DOC/'reports/recent-prediction-retirement-01.json'
    require(not report_path.exists(), 'Preserve original prediction retirement owner')
    primary_path=DOC/'reports/recent-component-scalar-statistics-01.json';primary=read(primary_path)
    spatial_path=DOC/'reports/recent-component-spatial-statistics-01.json';spatial=read(spatial_path)
    require(len(primary['models'])==len(spatial['models'])==8 and spatial['total_spatial_outputs']==133
            and not primary['prediction_arrays_embedded'] and not spatial['prediction_arrays_saved'],
            'All available scalar populations and spatial diagnostics must be retained before deleting fields')
    protected={}
    material=read(RUN/'remaining-component-material-01.json')
    for model in material['models']:
        for spec in [*model['inputs'].values(),*model['outputs'].values()]:
            p=ROOT/spec['path']
            if p.exists():protected[str(p.resolve())]=identity(p)
    for contract in ('macro-readout-component-contract-01.json','internal-residual-component-contract-01.json','guided-component-contract-02.json'):
        data=read(RUN/contract)
        spec=data.get('archive',data.get('data_archive'))
        if spec:protected[str((ROOT/spec['path']).resolve())]=identity(ROOT/spec['path'])
    captures=primary['prediction_archives_to_retire'];require(len(captures)==8, 'Exactly eight audited derivative captures required')
    decisions=[]
    for spec in captures:
        path=ROOT/spec['path']
        require(path.is_relative_to(RUN) and path.resolve()==path and path.suffix=='.npz'
                and str(path) not in protected and 'weights' not in path.name
                and ('fields' in path.name), 'Only ordinary task-owned derivative field captures may retire')
        observed=observed_file(ROOT,spec['path'],spec)
        require(any(m['source_prediction_identity']['path']==spec['path'] for m in spatial['models']),
                'Every retired capture must have completed scalar extraction')
        decisions.append(dict(**observed,before_identity=identity(path),deleted=False))
    primary['spatial_statistics']=binding(spatial_path)
    for model in primary['models']:
        spatial_model=next(m for m in spatial['models'] if m['fit_item_id']==model['fit_item_id'])
        model['spatial_statistics_scope']=spatial_model['scope']
        if spatial_model['spatial_outputs']==model['source_scalar_rows']:
            model['unavailable_full_population_metrics']=['Own-history drift']
    primary['status']='eight_recent_components_scalar_statistics_verified_fields_retired'
    primary['prediction_retirement_complete']=True
    # Decision is saved before deletion. An interrupted owner must reconcile
    # these exact paths/inodes and its existing receipt,never start another one.
    result=dict(status='retiring_verified_recent_prediction_arrays',created_unix=time.time(),pid=os.getpid(),
        producer=binding(Path(__file__)),user_instruction='Retain derived statistics;do not retain per-model predictions or gigabytes of field outputs.',
        primary_statistics_before_retirement=binding(primary_path),spatial_statistics=binding(spatial_path),
        captures=decisions,deleted_prediction_files=0,deleted_prediction_bytes=0,
        weights_deleted=0,training_or_reference_files_deleted=0,native_mutations=0,
        broader_prediction_inventory_complete=False,full_portfolio_complete=False)
    save(report_path,result)
    for item in decisions:
        path=ROOT/item['path']
        require(identity(path)==item['before_identity'], 'Derivative capture changed since its audited decision')
        path.unlink();require(not path.exists(), 'Prediction capture must be absent after retirement')
        item['deleted']=True;result['deleted_prediction_files']+=1;result['deleted_prediction_bytes']+=item['bytes']
        save(report_path,result)
    require(all(identity(Path(p))==value for p,value in protected.items()), 'Training/reference/model artifacts must remain untouched')
    save(primary_path,primary)
    result.update(status='eight_verified_prediction_captures_retired_scalar_statistics_preserved',
        primary_statistics=binding(primary_path),all_protected_material_identities_unchanged=True,
        protected_material_paths=len(protected),finished_unix=time.time())
    save(report_path,result)
    print('Retired',result['deleted_prediction_files'],'prediction captures:',result['deleted_prediction_bytes'],
          'bytes;scalar reports retained;weights/reference/training artifacts unchanged.',flush=True)


if __name__=='__main__':run()
