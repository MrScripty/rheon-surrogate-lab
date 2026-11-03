"""Find report-declared derivative arrays;do not delete or inspect model payloads."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require


def run():
    output=DOC/'reports/prediction-capture-retention-inventory-01.json'
    require(not output.exists(), 'Preserve first report-based prediction inventory')
    found={};reports=[]
    keys={'prediction_archive','prediction_archives','worst_field_capture','output_archive','predictions_archive','field_capture'}
    def candidate(value,source,location):
        if isinstance(value,list):
            for i,item in enumerate(value):candidate(item,source,location+'/'+str(i))
        if not isinstance(value,dict) or not isinstance(value.get('path'),str):return
        path=ROOT/value['path']
        if not path.is_relative_to(RUN) or '/sources/' in str(path) or path.suffix not in ('.npz','.npy'):return
        if 'weight' in path.name or 'training' in path.name:return
        entry=found.setdefault(str(path.relative_to(ROOT)),dict(path=str(path.relative_to(ROOT)),
            expected_sha256=value.get('sha256'),report_references=[],exists=path.is_file(),
            bytes=path.stat().st_size if path.is_file() else None,classification='Report-declared derivative-array candidate;consumer/statistics audit required before deletion'))
        entry['report_references'].append(dict(report=source,location=location))
    def walk(node,source,location=''):
        if isinstance(node,dict):
            for key,value in node.items():
                if key in keys:candidate(value,source,location+'/'+key)
                walk(value,source,location+'/'+key)
        elif isinstance(node,list):
            for i,value in enumerate(node):walk(value,source,location+'/'+str(i))
    for path in sorted((DOC/'reports').glob('*.json')):
        if path.name in ('prediction-capture-retention-inventory-01.json',):continue
        data=read(path);walk(data,str(path.relative_to(ROOT)));reports.append(binding(path))
    rows=list(found.values())
    save(output,dict(status='report_declared_prediction_arrays_inventoried_not_deleted',created_unix=time.time(),
        producer=binding(Path(__file__)),reports_scanned=len(reports),candidates=rows,
        existing_files=sum(r['exists'] for r in rows),existing_logical_bytes=sum(r['bytes'] or 0 for r in rows),
        already_absent_paths=sum(not r['exists'] for r in rows),
        scope='Top-level task reports and their explicit prediction/field/output archive declarations only. Does not cover unreferenced files,other lab histories or native payloads;does not authorize candidate deletion.',
        model_payloads_opened=False,files_deleted=0,new_model_forwards=0,CPU_neural_forwards=0,
        native_mutations=0,full_portfolio_complete=False))
    print('Prediction candidates:',sum(r['exists'] for r in rows),'existing files;',sum(r['bytes'] or 0 for r in rows),'bytes;no deletion.',flush=True)


if __name__=='__main__':run()
