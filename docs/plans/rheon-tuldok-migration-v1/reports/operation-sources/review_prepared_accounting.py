"""Independent filesystem-only review of explicitly selected actual M3 preparation.

Reads exact archived SQL and bounded staged evidence; never imports application or
converter code, opens a database, starts a service or publishes catalog changes.
The authoritative owner retains responsibility for complete stage tree hashing.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def main():
    if not __debug__:
        raise RuntimeError('Run this independent review without Python optimization.')
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','spec','prepare-receipt','bookend','schema-receipt','source-backup','source-selection','target-catalog','application-source','converter-source','receipt'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--bookend-sha256',required=True)
    parser.add_argument('--prepared-sha256',required=True)
    parser.add_argument('--target-connection-sha256',required=True)
    args=parser.parse_args()
    start=time.monotonic()
    reviewer_initial_sha=sha(Path(__file__))
    # Gate first: no partial/intended prepare artifact can receive a review PASS.
    owner=load(args.prepare_receipt)
    assert owner['schema']=='rheon-m3-orchestrator-v1' and owner['mode']=='prepare' and owner['status']=='prepared_read_only', 'Actual original prepare is not durably complete'
    assert owner['authoritative_db_writes']==0 and not owner['automatic_retry']
    assert owner['prepared_sha256']==args.prepared_sha256
    stage=args.stage.resolve(strict=True);backup=args.source_backup.resolve(strict=True);target=args.target_catalog.resolve(strict=True)
    assert owner['stage']==str(stage) and owner['spec']==str(args.spec.resolve(strict=True)) and owner['catalog']==str(target)
    assert stage!=target and not stage.is_relative_to(target) and not target.is_relative_to(stage)
    assert not args.receipt.exists(), 'Review receipt already exists'
    bookend=load(args.bookend);schema=load(args.schema_receipt)
    assert sha(args.bookend)==args.bookend_sha256==schema['bookend']['sha256']
    assert schema['bookend']['path']==str(args.bookend.resolve(strict=True))
    assert bookend['status']=='exact_backup_revalidated' and bookend['schema_version']==8
    assert schema['status']=='schema10_verified_reentry' and schema['catalog']==str(target)
    assert schema['upgrade']['status']=='upgraded' and schema['reentry']['status']=='already_current'
    assert bookend['complete_backup']==str(backup) and bookend['domain_tables']==31 and bookend['domain_rows']==6870 and bookend['managed_entries']==38598
    for key in ('all_domain_tables_equal','all_sequences_equal','all_managed_hashes_equal','all_file_and_directory_modes_equal','metadata_pre_post_stable','original_port_closed','original_source_command_absent'):
        assert bookend[key] is True,key
    assert bookend['authoritative_db_writes']==bookend['files_rewritten_or_replaced']==0 and not bookend['services_changed_by_helper']
    prepared_bytes=(stage/'prepared.json').read_bytes();prepared_initial_sha=hashlib.sha256(prepared_bytes).hexdigest();p=json.loads(prepared_bytes)
    spec=load(args.spec);selection=load(args.source_selection);s=load(backup/'metadata.json');b=load(backup/'backup-receipt.json')
    assert p['spec']==spec and digest(p)==args.prepared_sha256
    assert spec['target_connection_sha256']==args.target_connection_sha256
    assert spec['archive_receipt_sha256']==bookend['backup_receipt_sha256']==sha(backup/'backup-receipt.json')==selection['archive_receipt_sha256']
    assert sha(backup/'metadata.json')==b['metadata_sha256']
    assert spec['dataset_id']==owner['catalog_id']==schema['catalog_id']==bookend['dataset_id']==s['dataset_id']==b['dataset_id']
    assert spec['roots']==selection['roots'] and spec['target_schema_version']==10 and spec['target_definition_sha256']==schema['reentry']['definition_sha256']
    app_files={path.name:sha(path) for path in sorted(args.application_source.glob('*.py'))}
    assert app_files and digest(app_files)==spec['target_application_sha256']
    assert p['converter_sha256']==sha(args.converter_source)
    assert p['roots']=={'historical':str(stage),**{alias:str(stage/'external'/alias) for alias in selection['roots']}}
    assert p['catalog_files']==b['files'] and len(p['catalog_files'])==38598, 'Actual original managed namespace must have no clone restore-marker extra'
    assert p['sequences']==s['sequences']==b['sequences']
    for table,state in s['tables'].items():
        assert {key:state[key] for key in ('columns','count','row_sha256','sha256')}==b['tables'][table]==p['before'][table]
    assert all(not state['count'] for name,state in p['before'].items() if name not in s['tables'])
    items={a['source']:a for a in p['accounting']};assert len(items)==len(p['accounting'])==len(p['source_keys']) and set(items)==set(p['source_keys'])
    expected=set();bytable={};run_ids={};large_rows=0;fragment_count=0
    for table,t in s['tables'].items():
        counts=Counter()
        for ordinal,(row,rsha) in enumerate(zip(t['rows'],t['row_sha256'])):
            key=f'row:{table}:{rsha}:{ordinal}';expected.add(key);a=items[key]
            assert a['source_sha256']==rsha and digest(row)==rsha
            raw=(stage/a['archive']['path']).read_bytes();record=json.loads(raw)
            assert record['table']==table and record['columns']==t['columns'] and record['values']==row and record['original_row_sha256']==rsha
            assert hashlib.sha256(raw).hexdigest()==a['archive']['sha256']
            parts=[]
            for part in a['inspection']:
                r=(stage/part['path']).read_bytes();assert len(r)<=60*1024 and hashlib.sha256(r).hexdigest()==part['sha256'];parts.append(r);fragment_count+=1
            assert b''.join(parts)==raw
            if len(raw)>64*1024:large_rows+=1
            counts[a['disposition']]+=1
            if table=='workspace_runs':
                v=dict(zip(t['columns'],row));run_ids[v['id']]=a['disposition']
            if a['disposition']=='archived-visible':assert a['destination']==a['archive']
        bytable[table]=dict(counts)
    for prefix,inventory in [('catalog',b['files'])]+[('external/'+alias,sel['files']) for alias,sel in selection['roots'].items()]:
        for name,entry in inventory.items():
            key='file:'+prefix+'/'+name;expected.add(key);a=items[key]
            path='original-archive/catalog/'+name if prefix=='catalog' else prefix+'/'+name
            assert a['destination']=={'stage_path':path} and a.get('source_sha256')==entry.get('sha256')
            assert p['stage_files'][path]==entry
    for alias,sel in selection['roots'].items():
        for name,obs in sel['unavailable'].items():
            key='reference-unavailable:'+alias+'/'+name;expected.add(key);a=items[key];record=json.loads((stage/a['destination']['stage_path']).read_bytes())
            assert record['path']==name and record['root']==alias and record['observation']==obs
        for r in sel['recoveries']:
            key='recovered:'+alias+'/'+r['original_path']+':'+r['sha256'];expected.add(key);a=items[key]
            assert a['recovery_provenance']==r['source'] and a['destination']=={'stage_path':'recovered/'+r['sha256']}
            assert sha(stage/a['destination']['stage_path'])==r['sha256']
    assert expected==set(items)
    refs={};datarefs=set();expected_evidence=set();expected_data=set()
    for project in p['projects']:
        raw=(Path(p['roots'][project['root']])/project['manifest_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==project['manifest_sha256'] and json.loads(raw)==project['manifest']
        info=Path(p['roots'][project['root']]).stat();assert project['root_identity']==[info.st_dev,info.st_ino]
        if project['root']=='historical':
            for r in project['manifest']['evidence']:
                key=(r['path'],r['sha256']);assert key not in refs;refs[key]=project['id'];assert sha(stage/r['path'])==r['sha256']
            datarefs.update((r['path'],r['sha256']) for r in project['manifest']['data'])
    for n,a in enumerate(p['accounting']):
        for part in a.get('inspection',[]):
            expected_evidence.add((part['path'],part['sha256']));assert (part['path'],part['sha256']) in refs
        if a['source'].startswith(('file:','recovered:')) and a.get('source_sha256'):
            descriptor=json.loads((stage/f'files/{n}.json').read_bytes());expected_evidence.add((f'files/{n}.json',sha(stage/f'files/{n}.json')));expected_data.add((a['destination']['stage_path'],a['source_sha256']))
            assert descriptor['source']==a['source'] and descriptor['original_sha256']==a['source_sha256'] and descriptor['archive_path']==a['destination']['stage_path'] and descriptor['recovery_provenance']==a.get('recovery_provenance')
            assert (a['destination']['stage_path'],a['source_sha256']) in datarefs
        if a['source'].startswith('reference-unavailable:'):
            expected_evidence.add((a['destination']['stage_path'],a['destination']['sha256']));assert (a['destination']['stage_path'],a['destination']['sha256']) in refs
    assert set(refs)==expected_evidence and datarefs==expected_data
    
    source=s;bykey={a['source']:a for a in p['accounting']};projects={v['id']:v for v in p['projects']};converted=moved_recipes=0;retained_artifacts=[];runstates=Counter()
    for table in ('workspace_projects','workspace_runs'):
        t=source['tables'][table]
        for ordinal,(row,h) in enumerate(zip(t['rows'],t['row_sha256'])):
            v=dict(zip(t['columns'],row));doc=json.loads(v['document']);key=f'row:{table}:{h}:{ordinal}';a=bykey[key]
            assert a['source_id']==v['id']
            if table=='workspace_runs':
                runstates[(a['disposition'],doc['state'])]+=1
                if a['disposition']=='already-current':
                    assert a['destination']=={'table':table,'row_sha256':h}
                    for art in doc.get('artifacts',[]):retained_artifacts.append({'run_id':v['id'],**art})
                continue
            if a['disposition']=='archived-visible':assert a.get('source_observation')
            else:
                assert a['disposition']=='converted-current';saved=projects[a['destination']['id']];new=saved['manifest'];original=doc['manifest'];base_manifest=copy.deepcopy(original);activity=base_manifest.pop('activity_path',None);base_manifest['schema_version']=2;base_manifest.setdefault('extensions',[])
                if activity and not any(r['id']=='activity' for r in base_manifest['evidence']):base_manifest['evidence'].append({'id':'activity','label':'Live activity','path':activity})
                assert new['id']==digest({'source':key,'manifest':base_manifest,'archive':p['spec']['archive_receipt_sha256']})[:32]
                assert saved['root']==doc['root']
                for name in ('name','data','requirements','extensions'):assert new[name]==base_manifest[name]
                expected_evidence=list(base_manifest['evidence']);retained_recipes=[]
                for recipe in original['recipes']:
                    if recipe in new['recipes']:retained_recipes.append(recipe)
                    else:
                        archived_definition=json.loads((Path(p['roots'][saved['root']])/recipe['path']).read_bytes())
                        reason='obsolete runtime descriptor' if archived_definition['runtime']=='managed_torch_python_v1' else 'unsupported current Project recipe'
                        expected_evidence.append(dict(recipe,id='historical-'+recipe['id'],label='Historical recipe ('+reason+'): '+recipe['label']));moved_recipes+=1
                assert new['recipes']==retained_recipes and new['evidence']==expected_evidence
                assert a['destination']=={'table':'workspace_projects','id':new['id'],'manifest_sha256':saved['manifest_sha256'],'document_sha256':digest(saved)}
                converted+=1
    
    profiles=Counter()
    for project in p['projects']:
        if project['root']=='historical':continue
        assert not project['manifest']['recipes'], 'Historical recipe remained active'
        for ref in project['manifest']['evidence']:
            if ref['id'].startswith('historical-'):
                path=Path(p['roots'][project['root']])/ref['path']
                assert sha(path)==ref['sha256'], 'Historical recipe bytes changed'
                profiles[json.loads(path.read_bytes())['runtime']]+=1
    assert profiles=={'managed_torch_python_v1':231,'captured_cpu_python_v1':4}
    assert converted==83 and moved_recipes==235 and len(retained_artifacts)==72
    assert bytable['workspace_projects']=={'archived-visible':37,'converted-current':83}
    assert bytable['workspace_runs']=={'archived-visible':156,'already-current':4}
    assert len(expected)==49192 and len(p['projects'])==874 and large_rows==165 and fragment_count==10456
    assert len(refs)==47457 and len(datarefs)==36864
    assert runstates=={('archived-visible','completed'):154,('archived-visible','failed'):1,('already-current','completed'):4,('archived-visible','interrupted'):1}
    for artifact in retained_artifacts:
        rel=Path('workspace-runs')/artifact['run_id']/artifact['name']
        for location in (backup/'catalog',stage/'original-archive/catalog',target):
            path=location/rel
            assert not path.is_symlink() and path.stat().st_size==artifact['bytes'] and sha(path)==artifact['sha256'], 'Retained artifact differs'
    
    assert sha(stage/'prepared.json')==prepared_initial_sha and digest(load(stage/'prepared.json'))==args.prepared_sha256
    assert app_files=={path.name:sha(path) for path in sorted(args.application_source.glob('*.py'))}
    assert sha(args.converter_source)==p['converter_sha256']
    assert sha(Path(__file__))==reviewer_initial_sha
    result={'schema':'rheon-independent-actual-original-prepared-review-v1','status':'passed','scope':'actual-original-prepublication-review','accepted_for_root_publication_decision':True,'finished_at':datetime.now(timezone.utc).isoformat(),'command':[sys.executable,*sys.argv],
        'paths':{'stage':str(stage),'spec':str(args.spec.resolve()),'target_catalog':str(target),'source_backup':str(backup)},
        'prepared':{'canonical_sha256':args.prepared_sha256,'file_sha256':prepared_initial_sha,'owner_receipt_sha256':sha(args.prepare_receipt)},
        'bookend':{'path':str(args.bookend.resolve()),'sha256':sha(args.bookend)},'schema_upgrade_receipt_sha256':sha(args.schema_receipt),
        'source_backup_receipt_sha256':sha(backup/'backup-receipt.json'),'source_metadata_sha256':b['metadata_sha256'],'source_selection_sha256':sha(args.source_selection),
        'current_application_sha256':digest(app_files),'converter_source_sha256':p['converter_sha256'],'target_connection_sha256':spec['target_connection_sha256'],
        'source_keys_verified':len(expected),'source_row_count':sum(t['count'] for t in s['tables'].values()),'row_dispositions':bytable,'source_run_states_and_dispositions':{d+':'+state:n for (d,state),n in runstates.items()},
        'all_source_row_wrappers_and_fragments_exact':True,'row_fragments_hashed':fragment_count,'large_original_rows_reconstructed':large_rows,'historical_evidence_paths_hashed':len(refs),'historical_original_data_paths_verified':len(datarefs),
        'converted_projects':converted,'archived_source_projects':37,'historical_project_pages':len(p['projects'])-converted,'active_recipe_references':0,'historical_recipe_references_hashed':moved_recipes,'historical_recipe_profiles':dict(profiles),
        'preexisting_unavailable_records_verified':sum(len(root['unavailable']) for root in selection['roots'].values()),'exact_recoveries_verified':sum(len(root.get('recoveries',[])) for root in selection['roots'].values()),
        'retained_current_runs':len({a['run_id'] for a in retained_artifacts}),'retained_artifacts_hashed_three_locations':len(retained_artifacts),'retained_artifact_bytes':sum(a['bytes'] for a in retained_artifacts),
        'actual_original_catalog_inventory_exact_backup_no_clone_extra':True,'actual_stage_root_envelopes_and_reference_namespaces_verified':True,
        'reviewer_effects':{'database_access':False,'application_or_converter_imports':False,'private_authority_access':False,'service_actions':False,'catalog_writes':False},
        'limits':{'authoritative_owner_full_stage_tree_hash_not_repeated':True,'publication_authorized_or_executed_by_reviewer':False,'schema_or_service_cutover_completed_by_this_review':False},
        'reviewer_source_sha256':sha(Path(__file__)),'elapsed_seconds':time.monotonic()-start}
    args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('command','row_dispositions')},indent=2))


if __name__=='__main__':
    main()
