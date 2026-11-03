"""Keep a concise statistics index and one lossless compressed scalar payload."""
import gzip
import hashlib
import time
from pathlib import Path

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require


def run():
    index_path=DOC/'reports/portfolio-scalar-statistics-01.json'
    report_path=DOC/'reports/portfolio-statistics-storage-01.json'
    payload_path=RUN/'portfolio-scalar-statistics-01.json.gz'
    require(not report_path.exists() and not payload_path.exists(), 'Preserve original scalar storage owner')
    original=read(index_path)
    require(original['status']=='all203qualified_fit_scalar_populations_summarized'
            and len(original['models'])==203, 'Complete original scalar population required')
    original_identity=binding(index_path);before=index_path.stat().st_size
    with index_path.open('rb') as source,payload_path.open('wb') as destination:
        with gzip.GzipFile(filename='',mode='wb',fileobj=destination,compresslevel=6,mtime=0) as compressed:
            for chunk in iter(lambda:source.read(1024**2),b''):compressed.write(chunk)
    digest=hashlib.sha256();decoded_bytes=0
    with gzip.open(payload_path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):
            digest.update(chunk);decoded_bytes+=len(chunk)
    require(digest.hexdigest()==original_identity['sha256'] and decoded_bytes==before,
            'Compressed statistics must preserve every original byte and scalar')
    payload=dict(**binding(payload_path),bytes=payload_path.stat().st_size,
        decoded_json_sha256=digest.hexdigest(),decoded_json_bytes=decoded_bytes,format='gzip-compressed UTF8 JSON;scalar metrics and metadata only')
    models=[]
    for model in original['models']:
        concise={k:v for k,v in model.items() if k not in ('groups','timing_groups')}
        concise['full_strata_in_scalar_payload']=True;models.append(concise)
    concise={k:v for k,v in original.items() if k!='models'}
    concise.update(status='all203qualified_fit_scalar_statistics_index_and_lossless_payload',
        models=models,statistics_payload=payload,calculation_producer=original['producer'],
        storage_producer=binding(Path(__file__)),
        group_lookup='Decode statistics_payload with gzip/json;select models by fit_item_id,then groups or timing_groups. Index retains every overall statistic and case-median summary.',
        original_expanded_statistics_identity=original_identity)
    save(index_path,concise)
    save(report_path,dict(status='scalar_statistics_losslessly_packed_no_prediction_arrays',created_unix=time.time(),
        producer=binding(Path(__file__)),current_index=binding(index_path),payload=payload,
        original_expanded_json=original_identity,original_expanded_json_bytes=before,
        current_index_bytes=index_path.stat().st_size,total_current_bytes=index_path.stat().st_size+payload_path.stat().st_size,
        exact_roundtrip_verified=True,model_populations=203,qualified_timing_populations=195,
        prediction_arrays_embedded=False,new_model_forwards=0,model_dataset_or_project_copies=0,
        native_mutations=0,full_portfolio_complete=False))
    print('All scalar statistics preserved:',before,'bytes →',index_path.stat().st_size+payload_path.stat().st_size,
          'bytes including index;exact roundtrip.',flush=True)


if __name__=='__main__':run()
