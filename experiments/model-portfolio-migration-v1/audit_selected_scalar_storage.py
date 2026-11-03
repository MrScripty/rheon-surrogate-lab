"""Verify205fit scalar bodies, membership and actual retained disk footprint."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require


def scalar_dictionary(value):
    if isinstance(value,dict):return all(isinstance(k,str) and scalar_dictionary(v) for k,v in value.items())
    if isinstance(value,list):return False
    return value is None or type(value) in (str,bool,int,float)


def run():
    output=DOC/'reports/selected-scalar-storage-audit-01.json';require(not output.exists(),'Preserve first full selected-fit scalar storage audit')
    coverage_path=DOC/'reports/common-fit-benchmark-coverage-16.json';coverage=read(coverage_path)
    datasets={};expected={};source_cache={}
    for row in coverage['rows']:
        fid=row['fit_item_id'];retained=row['retained_field_statistics']
        expected[fid]=retained.get('output_records',retained.get('cases'))
        require(type(expected[fid]) is int and expected[fid]>0,'Explicit record membership required')
        spec=retained.get('scalar_case_payload')
        if spec is None:
            source=retained['source'];name=source['path']
            if name not in source_cache:
                require(binding(ROOT/name)['sha256']==source['sha256'],'Scalar index source changed')
                source_cache[name]=read(ROOT/name)
            spec=source_cache[name]['scalar_case_payload']
        dataset=datasets.setdefault(spec['path'],dict(identity=spec,fit_item_ids=[]))
        require(dataset['identity']['sha256']==spec['sha256'],'Conflicting scalar payload authority')
        dataset['fit_item_ids'].append(fid)
    counts=Counter();pvt_results=0;pvt_bytes=0;stage_results=0;total_disk=0
    for name,dataset in datasets.items():
        path=ROOT/name;spec=dataset['identity']
        require(path.is_relative_to(RUN) and path.suffix=='.gz','Task-owned compressed scalar result required')
        with path.open('rb') as stream:require(hashlib.file_digest(stream,'sha256').hexdigest()==spec['sha256'],'Scalar payload changed')
        dataset['bytes']=path.stat().st_size;total_disk+=dataset['bytes'];dataset_count=0
        with gzip.open(path,'rt') as stream:
            for line in stream:
                row=json.loads(line);fid=row['fit_item_id']
                require(fid in dataset['fit_item_ids'] and scalar_dictionary(row['statistics']),
                    'Only scalar dictionaries,not raw prediction arrays,may appear in statistics')
                if 'component_specific_statistics' in row:
                    require(scalar_dictionary(row['component_specific_statistics']),'Only scalar component diagnostics allowed')
                counts[fid]+=1;dataset_count+=1
                values=row['statistics']
                if 'fields' in values and set(values['fields'])=={'pressure','vx','vy','vz','tracer'}:
                    dtypes=values['output_dtypes'];sizes={'<f4':4,'<f8':8}
                    pvt_bytes+=sum(values['fields'][n]['prediction']['values']*sizes[dtypes[n]] for n in dtypes)
                    pvt_results+=1
                    stage_results+=int(row.get('identity',{}).get('case_role')=='supplied_internal_stage_context')
        dataset['output_records']=dataset_count
    require(dict(counts)==expected and len(counts)==205 and sum(counts.values())==49700,'Exact complete205fit scalar population required')
    require(pvt_results==22832 and stage_results==3132,'PVTstate andpartialstage populations must remain distinct')
    newer_root=RUN/'recent-component-scalar-replay-01'
    ordinary_output_files=[]
    for path in newer_root.rglob('*'):
        if path.is_file():
            require(path.suffix in ('.json','.gz'),'Unexpected non-scalar file in new component replay outputs')
            ordinary_output_files.append(str(path.relative_to(ROOT)))
    save(output,dict(status='all205selected_fit_scalar_payloads_rehashed_and_array_free',created_unix=time.time(),
        producer=binding(Path(__file__)),coverage=binding(coverage_path),datasets=list(datasets.values()),
        distinct_compressed_payloads=len(datasets),compressed_payload_disk_bytes=total_disk,
        fit_populations=205,task_specific_output_records=49700,records_by_fit=dict(counts),
        PVT_state_results_scored_in_memory=pvt_results,PVT_internal_stage_results=stage_results,
        PVT_physical_outputs=pvt_results-stage_results,
        uncompressed_generated_PVT_array_bytes=pvt_bytes,
        uncompressed_byte_scope='Only actual generatedPVTarrays represented byretainedscalarrecords. Notcompressedarchive size orameasured disksaving claim. Pressure/localoperator outputs excluded.',
        statistics_dictionary_contains_raw_arrays=False,new_component_output_files=ordinary_output_files,
        datasets_or_weights_duplicated=False,prediction_fields_saved=False,prediction_files_deleted=0,
        deletion_paused=True,native_mutations=0,full_portfolio_complete=False))
    print('Audited205fits/49700records in',len(datasets),'scalar payloads;',total_disk,'compressedbytes;',pvt_bytes,'generatedPVTbytes neverarchived.',flush=True)


if __name__=='__main__':run()
