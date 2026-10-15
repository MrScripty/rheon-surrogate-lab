"""Normal Tuldok field records; each whole trajectory has independent lineage."""
import io
import re
from pathlib import Path
import sys
import uuid
import zipfile
from policy import LAB,MIB,canonical,decode,fresh,read,require,sha,write

FIELDS=[('pressure','<f8','cell','Pa'),('tracer','<f4','cell','dimensionless'),
        ('corrected_u','<f4','face_x','m/s'),('corrected_v','<f4','face_y','m/s'),('corrected_w','<f4','face_z','m/s')]


def bundles(capture_root,tuldok,output,population,source_name):
    require(re.fullmatch(r"[a-z][a-z0-9-]{0,23}", source_name), "collection_scope", "bounded source namespace")
    require(population in ('train_validation','reserved_test'),'population','explicit collection population')
    native_raw=read(capture_root/'manifest.json',4*MIB);native=decode(native_raw)
    require(native['status']=='verified_development','capture','qualified full corpus')
    output=fresh(output);sys.path.insert(0,str(tuldok))
    from adapters.rheon.generate import npy
    import scientific_fields
    rows=[]
    for trajectory in native['trajectories']:
        entry=trajectory['entry']
        if (entry['split']=='reserved_test')!=(population=='reserved_test'):continue
        if population=='train_validation' and entry['split']!='train':continue
        identity=source_name+'--'+entry['trajectory_id']
        geom=trajectory['geometry'];counts=geom['counts_xyz'];h=1/entry['cells_per_metre'];members={}
        manifest={'schema_version':1,'format':'scientific_field_sequence_v1','grid':{'shape_xyz':counts,'origin_xyz':geom['origin_m'],
            'spacing_xyz':[h]*3,'orientation':[[1,0,0],[0,1,0],[0,0,1]],'units':'m'},'time_units':'s',
            'fields':[{'id':name,'label':name.replace('_',' '),'dtype':dtype,'centering':center,'units':units,
                'stage':'accepted native water reference; passive tracer'} for name,dtype,center,units in FIELDS],
            'frames':[],'family':{'namespace':'sealed-water-forcing-v1','id':native['reservation_sha256']+'--'+entry['trajectory_id']},'sources':[],
            'provenance':{'producer':'Qualified sealed viscous water capture','source_id':trajectory['fingerprint']}}
        for frame in trajectory['frames']:
            row={'time':str(frame['time_s']),'arrays':[]}
            for name,dtype,_center,_units in FIELDS:
                descriptor=frame['arrays'][name];raw=read(capture_root/descriptor['path'],MIB)
                require(sha(raw)==descriptor['sha256'],'raw_drift',name)
                path=f'arrays/{frame["index"]:03d}-{name}.npy';value=npy(raw,dtype,descriptor['shape_xyz'][::-1]);members[path]=value
                row['arrays'].append({'field':name,'path':path,'bytes':len(value),'sha256':sha(value)})
            manifest['frames'].append(row)
        source=canonical({'schema':'rheon-water-field-provenance-v1','capture_manifest_sha256':sha(native_raw),
            'reservation_sha256':native['reservation_sha256'],'scene':entry,'source_name':source_name,'geometry':geom,'trajectory_fingerprint':trajectory['fingerprint']})
        members['scene-provenance.json']=source
        manifest['sources']=[{'path':'scene-provenance.json','bytes':len(source),'sha256':sha(source),'description':'Measured physical scene, material, forcing and capture lineage'}]
        members['manifest.json']=canonical(manifest);stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as archive:
            for path,value in sorted(members.items()):archive.writestr(path,value)
        raw=stream.getvalue();scientific_fields.parse_bundle(raw)
        path=output/(identity+'.zip');path.write_bytes(raw)
        rows.append({'trajectory_id':identity,'split':entry['split'],'counts_xyz':counts,'path':path.relative_to(LAB).as_posix(),'sha256':sha(raw),'bytes':len(raw)})
    project={'schema_version':1,'id':uuid.uuid4().hex,'name':'Water learning · '+source_name+' · '+population,'activity_path':'runs/live-project/activity.json',
        'data':[{'id':r['trajectory_id'],'label':r['trajectory_id'],'path':r['path'],'sha256':r['sha256']} for r in rows],
        'recipes':[],'evidence':[],'requirements':[]}
    # Project references are bounded; partition the catalog-facing index.
    projects=[]
    for offset in range(0,len(rows),64):
        selected={**project,'id':uuid.uuid4().hex,'data':project['data'][offset:offset+64]}
        path=output/f'project-{offset//64:02d}.json';write(path,selected);projects.append(path.relative_to(LAB).as_posix())
    write(output/'exports.json',{'population':population,'rows':rows,'projects':projects});return {'records':len(rows),'projects':projects}


def publish(client,exports,output,observer=None):
    exports=decode(read(exports,2*MIB));records=[]
    if Path(output).exists():
        previous=decode(read(Path(output),2*MIB))
        require(previous['catalog_id']==client.catalog_id, 'collection_catalog', 'resume in the same catalog')
        records=previous['records']
        require(all(any(r['trajectory_id']==e['trajectory_id'] and r['sha256']==e['sha256'] for e in exports['rows']) for r in records), 'collection_resume', 'same exact exports')
        for row in records:
            saved=client.request('GET',f'/api/workbench/records/{row["record_id"]}')
            require(saved['kind']=='field_sequence' and saved['content_hash']==row['sha256'], 'collection_resume', 'reopened exact prior records')
    for path in exports['projects']:
        manifest=decode(read(LAB/path,MIB));opened=client.request('POST','/api/workspace/projects',{'root':'lab','manifest_path':path})
        for reference in manifest['data']:
            if any(r['trajectory_id']==reference['id'] for r in records):continue
            row=next(r for r in exports['rows'] if r['trajectory_id']==reference['id'])
            record=client.request('POST',f'/api/workspace/projects/{opened["id"]}/import-data/{reference["id"]}',
                {'expected_sha256':reference['sha256'],'name':f'Water · {row["split"]} · {row["trajectory_id"]} · {row["counts_xyz"]}',
                 'rights':'Locally generated synthetic sealed-water reference; independent snapshot physics verified.',
                 'groups':[row['trajectory_id']],'parents':[]})
            records.append({'record_id':record['id'],**row})
            write_path=Path(output);write_path.write_bytes(canonical({'status':'importing','catalog_id':client.catalog_id,'records':records}))
            if observer:observer.publish('Publishing water datasets',f'{len(records)}/{len(exports["rows"])} whole-scene field records visible in Tuldok',len(records),len(exports['rows']))
    result={'status':'draft_reference_fields_imported','catalog_id':client.catalog_id,'population':exports['population'],'records':records,'review_or_release_admitted':False}
    Path(output).write_bytes(canonical(result));return result
