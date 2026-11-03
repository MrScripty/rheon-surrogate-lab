"""One serial owner for the qualified150fit pressure comparison; no retries."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import uuid

from import_common_generator_measurements import ROOT,RUN,DOC,PROJECT,CONTROLLER,header,payloads,read,save,binding,digest,encode,require

PACKET = RUN/'all150-pressure-GPU-measurements-01'
PROOF = DOC/'reports/all150-pressure-GPU-measurements-qualification-01.json'
JOB = RUN/'native-all150-pressure-import-01'
BEFORE = JOB/'before-custody.json'
STATUS = JOB/'status.json'
REPORT = DOC/'reports/native-all150-pressure-import-custody-01.json'


def qualified():
    prep,proof = read(PACKET/'preparation.json'),read(PROOF)
    require(proof['status'] == 'all150_pressure_GPU_measurements_independently_qualified_not_imported'
            and proof['fits'] == prep['subjects'] == 150 and proof['exact_scientific_observations'] == prep['observations'] == 96000
            and proof['original_primary_samples_checked'] == 384000 and proof['original_fresh_process_bindings_checked'] == 300
            and proof['native_filter_queries'] == 335 and len(prep['packets']) == 2,'Complete150pressure packet qualification required')
    for spec in [proof['producer'],proof['preparation'],proof['source_manifest'],proof['native_owner'],*proof['packets'],
                 prep['producer'],*prep['evidence'],prep['original_fit_source_coverage']]:
        require(binding(ROOT/spec['path']) == spec,'Qualified pressure publication evidence changed')
    return prep,proof


def before():
    from original_train_pressure_accuracy_02 import publication_gate
    publication_gate()
    require(not JOB.exists() and not REPORT.exists(),'Fresh native pressure owner required; reconcile original attempts')
    prep,_ = qualified()
    for mode in ('pilot','batch'):
        state = read(RUN/f'original-pressure-GPU-timing-{mode}-01/status.json')
        require(state['status'] == 'all_original_pressure_GPU_timing_phases_terminal' and state['all_qualified']
                and state['active_child'] is None and not Path(f'/proc/{state["pid"]}').exists(),'All GPU timing owners must be terminal')
    raw,current = header()
    prior = read(DOC/'reports/native-matched-generator-outcome-adoption-01.json')
    keep = payloads()
    sources = [e['id'].split('.',1) for e in current['inventory'] if e['family'] == 'measurement_source']
    hashes = [prep['source_manifest']['sha256'],*[part['sha256'] for part in prep['packets']]]
    require(keep == prior['payload_preservation'] and len(sources) == 29
            and len(set(hashes)) == 3 and all(not (PROJECT/'artifacts/sha256'/h).exists() for h in hashes),
            'Exact previous custody and three genuinely new chart metadata blobs required')
    JOB.mkdir()
    save(BEFORE,dict(status='compact_all150_pressure_before_custody_captured',created_unix=time.time(),checkpoint=current['checkpoint'],
        project_id=current['project_id'],header_sha256=digest(raw),canonical_records=len(current['inventory']),payloads=keep,
        preexisting_sources=sources,expected_new_blobs=hashes,preparation=binding(PACKET/'preparation.json'),
        no_full_header_project_or_payload_backup=True))
    print('Compact150pressure before-custody captured; no native mutation dispatched.',flush=True)


def adopt():
    require(BEFORE.exists() and not STATUS.exists() and not list(JOB.glob('*.request.json')),
            'Original before-custody required; never replay native imports')
    prep,proof = qualified()
    previous = read(BEFORE)
    raw,current_header = header()
    require(current_header['checkpoint'] == previous['checkpoint'] and digest(raw) == previous['header_sha256'],
            'Current project changed before import')
    controller = read(CONTROLLER/'controller.json')
    source = Path(controller['source'])
    require(controller['status'] == 'ready' and controller['source_commit'] == 'f8ba4cedd22580f273f8c4fcbc06bca53797a7a7'
            and all(Path(f'/proc/{controller[k]}').exists() for k in ('launcher_pid','app_pid'))
            and subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() == controller['source_commit']
            and not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
            and binding(source/'retained_measurements.py') == proof['native_owner'],'Exact clean installed UI source required')
    state = dict(status='starting_original_all150_pressure_import',pid=os.getpid(),created_unix=time.time(),
        project_id=current_header['project_id'],starting_checkpoint=previous['checkpoint'],requests=[],imports=[],
        source_commit=controller['source_commit'],new_model_forwards=0,model_or_dataset_copies=0,
        raw_process_bodies_copied=0,automatic_retries=False,native_UI_accepted=False,full_portfolio_complete=False)
    save(STATUS,state)
    sys.path.insert(0,str(source))
    import agent_api
    client = agent_api.ProjectFilesOwnerClient(controller['origin'],(CONTROLLER/'private/owner-session.txt').read_text().strip(),
        controller['catalog_id'],timeout=7200)
    try:
        client.bootstrap()
        response = client.call('project_files.status',{})
        save(JOB/'before-status.response.json',response)
        selected = response['result']
        require(response['status'] == 200 and selected['state'] == 'open'
                and selected['project']['project_id'] == current_header['project_id']
                and selected['project']['checkpoint'] == previous['checkpoint'],'Actual native selection changed')
        current = dict(binding=selected['binding'],checkpoint=previous['checkpoint'])

        def call(action,body):
            operation = 'project_files.'+action
            arguments = dict(binding=current['binding'],**body)
            if action in ('source_register','measurement_import'):
                arguments.update(expected_checkpoint=current['checkpoint'],request_key=uuid.uuid4().hex)
            else:
                arguments['checkpoint'] = current['checkpoint']
            stem = f'{len(state["requests"])+1:03d}-{action}'
            save(JOB/(stem+'.request.json'),dict(operation=operation,arguments=arguments,
                origin=controller['origin'],controller_catalog_id=controller['catalog_id']))
            item = dict(operation=operation,request=stem+'.request.json',outcome='dispatched_once')
            state['requests'].append(item)
            state.update(status='original_all150_pressure_request_in_flight',active_request=stem,updated_unix=time.time())
            save(STATUS,state)
            tick = time.monotonic()
            reply = client.call(operation,arguments)
            save(JOB/(stem+'.response.json'),reply)
            item.update(status=reply['status'],response=stem+'.response.json',wall_seconds=time.monotonic()-tick)
            save(STATUS,state)
            require(reply['status'] == 200,'Original native outcome requires reconciliation; no automatic retry')
            envelope = reply['result']
            require(envelope['binding'] == current['binding'] and envelope['project_id'] == current_header['project_id'],
                    'Actual project/binding changed')
            current['checkpoint'] = envelope['checkpoint']
            state['last_acknowledged_checkpoint'] = current['checkpoint']
            save(STATUS,state)
            print(action,'HTTP200',stem,flush=True)
            return envelope['value']

        require((PROJECT/'.tuldok/project.json').read_bytes() == raw,'Checkpoint changed before first mutation')
        registered = call('source_register',dict(root='lab',manifest_path=str((PACKET/'source-project.json').relative_to(ROOT))))
        require(registered['project_id'] == read(PACKET/'source-project.json')['id']
                and registered['manifest_sha256'] == prep['source_manifest']['sha256'],'Registered source differs')
        snapshot = {key:registered[key] for key in ('origin_catalog_id','project_id','manifest_sha256')}
        state['source_snapshot'] = snapshot
        save(STATUS,state)
        for number,part in enumerate(prep['packets'],1):
            admitted = call('measurement_import',dict(snapshot=snapshot,reference_id=f'all150-pressure-{number}',expected_sha256=part['sha256']))
            require(admitted['blob'] == dict(sha256=part['sha256'],bytes=part['bytes']),'Imported chart blob differs')
            state['imports'].append(dict(part=number,**admitted))
            save(STATUS,state)
        sources = call('measurement_sources',dict(offset=0,limit=40))
        expected = {tuple(s) for s in previous['preexisting_sources']}|{(s['origin_catalog_id'],s['id']) for s in state['imports']}
        require(sources['total'] == 31 and {(s['origin_catalog_id'],s['id']) for s in sources['items']} == expected,
                'Native source population differs')
        locators = []
        values = {}
        for part,admitted in zip(prep['packets'],state['imports'],strict=True):
            locator = dict(origin_catalog_id=admitted['origin_catalog_id'],packet_sha256=part['sha256'])
            locators.append(locator)
            metadata = call('measurement_metadata',locator)['value']
            require((metadata['subject_count'],metadata['case_count'],metadata['observation_count']) == (75,128,48000),
                    'Imported metadata counts differ')
            packet = read(PACKET/part['path'])
            for observation in packet['observations']:
                case = packet['cases'][int(observation['case_id'].removeprefix('case-'))]
                values.setdefault((observation['subject_id'],observation['metric_id']),[]).append((case['dimensions'],observation['value']))
        selections = [{},dict(cells_per_metre=8),dict(cells_per_metre=16),
                      dict(cells_per_metre=8,grid_x=16,grid_y=16,grid_z=16)]
        for filters in selections:
            for metric in ('pressure_mean_free_relative_l2','equation_relative_l2','pressure_call_ms'):
                query = call('measurement_compare',dict(sources=locators,metric_id=metric,filters=filters,offset=0,limit=1))['value']['query']
                require(len(query['summary']) == 150,'Native combined view must contain all150fits')
                total = 0
                for summary in query['summary']:
                    selected_values = [value for dims,value in values[(summary['subject_id'],metric)] if all(dims[k] == v for k,v in filters.items())]
                    require(selected_values and summary['states'] == dict(measured=len(selected_values),failed=0,unsupported=0,unavailable=0)
                            and struct.pack('!d',summary['maximum']['value']) == struct.pack('!d',max(selected_values)),
                            'Actual native filtered count or maximum differs')
                    total += len(selected_values)
                require(query['total'] == total,'Native query total differs')
        state.update(status='all150_pressure_imports_native_metadata_and_maxima_passed',finished_unix=time.time(),
            active_request=None,measured_scalars=96000,original_fits=150,source_count=31,native_metadata_reads=2,
            native_combined_maxima_queries=12,project_left_open=True)
    except BaseException as error:
        state.update(status='original_all150_pressure_outcome_requires_reconciliation',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        save(STATUS,state)


def after():
    require(not REPORT.exists(),'Preserve original pressure import custody')
    prep,_ = qualified()
    previous,state = read(BEFORE),read(STATUS)
    require(state['status'] == 'all150_pressure_imports_native_metadata_and_maxima_passed'
            and not Path(f'/proc/{state["pid"]}').exists() and len(state['imports']) == 2,'Original imports and queries must be terminal')
    raw,current = header()
    require(current['checkpoint'] == state['last_acknowledged_checkpoint'],'Actual checkpoint differs from acknowledgement')
    keys = [read(JOB/item['request'])['arguments']['request_key'] for item in state['requests']
            if item['operation'] in ('project_files.source_register','project_files.measurement_import')]
    require(len(keys) == len(set(keys)) == 3 and all(item.get('status') == 200 for item in state['requests']),
            'Exactly three acknowledged mutations required')
    additions = {('project_operation',current['project_id']+'.'+digest(key.encode())) for key in keys}
    source_hash = prep['source_manifest']['sha256']
    additions.update({('project_snapshot',current['project_id']+'.'+source_hash),('project_artifact',source_hash)})
    for part,admitted in zip(prep['packets'],state['imports'],strict=True):
        additions.update({('project_artifact',part['sha256']),('measurement_source',admitted['origin_catalog_id']+'.'+part['sha256'])})
    indexed = {(entry['family'],entry['id']):entry for entry in current['inventory']}
    require(len(additions) == 9 and additions <= indexed.keys() and len(indexed) == previous['canonical_records']+9,
            'Unexpected pressure import record additions')
    old = dict(current,checkpoint=previous['checkpoint'])
    for name in ('inventory','roots'):
        old[name] = [entry for entry in current[name] if (entry['family'],entry['id']) not in additions]
    old['receipts'] = [row for row in current['receipts'] if row['request_key'] not in keys]
    require(digest(encode(old)) == previous['header_sha256'],'Original header/records/receipts changed')
    total = 0
    for entry in current['inventory']:
        path = PROJECT/'.tuldok'/entry['path']
        body = path.read_bytes()
        require(path.is_file() and not path.is_symlink() and len(body) == entry['bytes'] and digest(body) == entry['sha256'],
                'Canonical pressure custody body differs')
        total += len(body)
    require(payloads(exclude=previous['expected_new_blobs']) == previous['payloads']
            and payloads()['files'] == previous['payloads']['files']+3,'Prior scientific payload identity changed')
    manifest = read(PACKET/'source-project.json')
    for path in [PACKET/'source-project.json',*[PACKET/part['path'] for part in prep['packets']]]:
        body = path.read_bytes();sha = digest(body)
        require((PROJECT/'artifacts/sha256'/sha).read_bytes() == body and
                read(PROJECT/'.tuldok'/indexed[('project_artifact',sha)]['path']) == dict(format='project_artifact_v1',sha256=sha,bytes=len(body)),
                'Imported chart metadata/declaration differs')
    snapshot = read(PROJECT/'.tuldok'/indexed[('project_snapshot',current['project_id']+'.'+source_hash)]['path'])
    require(snapshot['origin_catalog_id'] == current['project_id'] and snapshot['value']['manifest'] == manifest
            and snapshot['value']['id'] == manifest['id'],'Native pressure source snapshot differs')
    for number,(part,admitted) in enumerate(zip(prep['packets'],state['imports'],strict=True),1):
        reference = next(ref for ref in manifest['data'] if ref['id'] == f'all150-pressure-{number}')
        doc = read(PROJECT/'.tuldok'/indexed[('measurement_source',admitted['origin_catalog_id']+'.'+part['sha256'])]['path'])
        expected = dict(origin_catalog_id=current['project_id'],project_id=manifest['id'],manifest_sha256=source_hash,
            reference_id=reference['id'],path=reference['path'],measured_sha256=part['sha256'])
        require(doc['source'] == expected and doc['blob'] == dict(sha256=part['sha256'],bytes=part['bytes']),
                'Native pressure table lost original source binding')
    require((PROJECT/'.tuldok/project.json').read_bytes() == raw,'Project changed during custody')
    save(REPORT,dict(status='all150_pressure_tables_native_custody_exact',created_unix=time.time(),checkpoint=current['checkpoint'],
        canonical_header_sha256=digest(raw),canonical_records=len(indexed),canonical_record_bytes=total,
        original_header_reconstructed_exactly=True,all_canonical_bodies_freshly_hashed=True,
        prior_scientific_payload_identities_exact=True,old_payload_bodies_rehashed=False,new_chart_metadata_blobs_verified=3,
        pressure_fits=150,measured_scalars=96000,native_sources=31,project_left_open=True,native_UI_pending=True,
        payloads=payloads(),producer=binding(Path(__file__)),original_owner=binding(STATUS),before=binding(BEFORE),full_portfolio_complete=False))
    print('Native150pressure custody exact;96,000scalars and prior history/payload identities retained;UI pending.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('before','import','after'))
    args = parser.parse_args()
    {'before':before,'import':adopt,'after':after}[args.phase]()
