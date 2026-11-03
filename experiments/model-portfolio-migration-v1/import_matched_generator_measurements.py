"""One retained native owner for the readable, qualified matched GPU tables.

Original requests are stored before dispatch. Unknown or rejected outcomes stop
this owner for reconciliation; no automatic retry occurs. Custody reconstructs
the prior header without storing a duplicate project or complete header backup.
"""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import uuid

from import_common_generator_measurements import ROOT,RUN,DOC,PROJECT,CONTROLLER,header,payloads,read,digest,encode,binding,require,save

PACKET = RUN/'matched-generator-GPU-measurements-02'
JOB = RUN/'native-matched-generator-GPU-import-01'
BEFORE = JOB/'before-custody.json'
STATUS = JOB/'status.json'
REPORT = DOC/'reports/native-matched-generator-GPU-import-custody-01.json'
PROOF = DOC/'reports/readable-matched-generator-GPU-measurements-qualification-01.json'


def qualified():
    prep,proof = read(PACKET/'preparation.json'),read(PROOF)
    require(prep['exact_fit_count'] == proof['exact_fits'] == 35
            and prep['F64_assembly_executions'] == proof['assembly_executions'] == 4
            and prep['measured_scalars'] == proof['measured_scalars'] == 24960
            and proof['raw_process_blobs_exact'] == 78 and proof['native_filter_queries'] == 600
            and proof['incompatible_K_pair_rejections'] == 6 and len(prep['packets']) == 4,
            'Complete readable matched-table qualification required')
    for spec in [proof['producer'],proof['oracle'],proof['preparation'],proof['native_owner'],*proof['packet_bindings'],
            prep['producer'],prep['source_manifest'],prep['presentation_inverse']]:
        require(binding(ROOT/spec['path']) == spec,'Qualified matched authority changed')
    for part in prep['packets']:
        raw = (PACKET/part['path']).read_bytes()
        require(digest(raw) == part['sha256'] and len(raw) == part['bytes'],'Qualified matched table changed')
    return prep,proof


def before():
    require(not JOB.exists() and not REPORT.exists(),'Preserve original native matched import owner')
    prep,_ = qualified()
    old_audit = read(DOC/'reports/native-common-generator-GPU-import-custody-01.json')
    old_UI = read(DOC/'reports/native-common-generator-GPU-charts-UI-01.json')
    raw,current = header()
    require(old_audit['status'] == 'four_common_generator_GPU_tables_native_custody_exact'
            and current['checkpoint'] == old_audit['checkpoint'] == old_UI['checkpoint']
            and digest(raw) == old_audit['canonical_header_sha256'] and old_UI['actual_UI_actions'] == 516,
            'Previous native import custody and actual UI must match current checkpoint')
    for name,status in [('complete-generator-GPU-benchmark-01','all70_original_workers_complete_reduction_pending'),
                        ('same-case-F64-GPU-benchmark-02','all8_original_matched_F64_workers_complete_reduction_pending')]:
        owner = read(RUN/name/'status.json')
        require(owner['status'] == status and owner['active'] is None and not Path(f'/proc/{owner["pid"]}').exists(),
                'All controlled GPU workers must be terminal before publication')
    hashes = [prep['source_manifest']['sha256'],*[p['sha256'] for p in prep['packets']]]
    require(len(set(hashes)) == 5 and all(not (PROJECT/'artifacts/sha256'/h).exists() for h in hashes),
            'Five genuinely new metadata blobs required')
    sources = [e['id'].split('.',1) for e in current['inventory'] if e['family'] == 'measurement_source']
    keep = payloads()
    require(len(sources) == 25 and keep == old_audit['payloads'],'Prior sources or scientific payload identities changed')
    JOB.mkdir()
    save(BEFORE,dict(status='compact_before_custody_captured',created_unix=time.time(),checkpoint=current['checkpoint'],
        project_id=current['project_id'],header_sha256=digest(raw),canonical_records=len(current['inventory']),
        payloads=keep,preexisting_sources=sources,preparation=binding(PACKET/'preparation.json'),
        expected_new_blobs=hashes,no_full_header_or_payload_backup=True))
    print('Compact matched-table before-custody captured; no native request dispatched.',flush=True)


def adopt():
    require(BEFORE.exists() and not STATUS.exists() and not list(JOB.glob('*.request.json')),
            'Original before-custody required; never replay an import owner')
    prep,proof = qualified()
    previous = read(BEFORE)
    raw,current_header = header()
    require(current_header['checkpoint'] == previous['checkpoint'] and digest(raw) == previous['header_sha256'],
            'Checkpoint changed before matched import')
    controller = read(CONTROLLER/'controller.json')
    source = Path(controller['source'])
    require(controller['status'] == 'ready' and controller['source_commit'] == 'f8ba4cedd22580f273f8c4fcbc06bca53797a7a7'
            and all(Path(f'/proc/{controller[k]}').exists() for k in ('launcher_pid','app_pid'))
            and subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() == controller['source_commit']
            and not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
            and binding(source/'retained_measurements.py') == proof['native_owner'],'Exact clean live source required')
    state = dict(status='starting_original_matched_import',pid=os.getpid(),created_unix=time.time(),
        project_id=current_header['project_id'],starting_checkpoint=previous['checkpoint'],requests=[],imports=[],
        source_commit=controller['source_commit'],new_model_forwards=0,new_reference_generation=False,
        dataset_or_weight_copies=0,native_UI_accepted=False,full_portfolio_complete=False)
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
            arguments = dict(body,binding=current['binding'])
            if agent_api.operation(operation)['side_effects']:
                arguments.update(expected_checkpoint=current['checkpoint'],request_key=uuid.uuid4().hex)
            else:
                arguments['checkpoint'] = current['checkpoint']
            stem = str(len(state['requests'])+1).zfill(3)+'-'+action
            save(JOB/(stem+'.request.json'),dict(operation=operation,arguments=arguments,
                origin=controller['origin'],controller_catalog_id=controller['catalog_id']))
            receipt = dict(operation=operation,request=stem+'.request.json',outcome='dispatched_once')
            state['requests'].append(receipt)
            state.update(status='original_native_matched_request_in_flight',active_request=stem,updated_unix=time.time())
            save(STATUS,state)
            tick = time.monotonic()
            response = client.call(operation,arguments)
            save(JOB/(stem+'.response.json'),response)
            receipt.update(status=response['status'],response=stem+'.response.json',wall_seconds=time.monotonic()-tick)
            save(STATUS,state)
            require(response['status'] == 200,'Original native outcome requires reconciliation; no retry')
            envelope = response['result']
            require(envelope['binding'] == current['binding'] and envelope['project_id'] == current_header['project_id'],
                    'Native binding/project changed')
            current['checkpoint'] = envelope['checkpoint']
            state['last_acknowledged_checkpoint'] = current['checkpoint']
            save(STATUS,state)
            print(action,'HTTP200',stem,flush=True)
            return envelope['value']

        require((PROJECT/'.tuldok/project.json').read_bytes() == raw,'Checkpoint changed before first mutation')
        registered = call('source_register',dict(root='lab',manifest_path=str((PACKET/'source-project.json').relative_to(ROOT))))
        require(registered['project_id'] == read(PACKET/'source-project.json')['id']
                and registered['manifest_sha256'] == prep['source_manifest']['sha256'],'Registered matched source differs')
        snapshot = {k:registered[k] for k in ('origin_catalog_id','project_id','manifest_sha256')}
        state['source_snapshot'] = snapshot
        save(STATUS,state)
        for part in prep['packets']:
            admitted = call('measurement_import',dict(snapshot=snapshot,reference_id=f'K{part["K"]}-complete-GPU',expected_sha256=part['sha256']))
            require(admitted['blob'] == dict(sha256=part['sha256'],bytes=part['bytes']),'Imported matched blob differs')
            state['imports'].append(dict(K=part['K'],**admitted))
            save(STATUS,state)
        sources = call('measurement_sources',dict(offset=0,limit=40))
        expected = {tuple(s) for s in previous['preexisting_sources']}|{(s['origin_catalog_id'],s['id']) for s in state['imports']}
        require(sources['total'] == 29 and {(s['origin_catalog_id'],s['id']) for s in sources['items']} == expected,
                'Native matched source population differs')
        for part,admitted in zip(prep['packets'],state['imports'],strict=True):
            metadata = call('measurement_metadata',dict(origin_catalog_id=admitted['origin_catalog_id'],packet_sha256=part['sha256']))['value']
            require((metadata['subject_count'],metadata['case_count'],metadata['observation_count']) ==
                    (len(part['subjects']),64,part['measured']),'Native matched subject/case/scalar counts differ')
            packet = read(PACKET/part['path'])
            for metric in ('pressure_relative_l2','velocity_relative_l2','per_output_median_ms'):
                query = call('measurement_compare',dict(sources=[dict(origin_catalog_id=admitted['origin_catalog_id'],packet_sha256=part['sha256'])],
                    metric_id=metric,subject_ids=part['subjects'],filters={},offset=0,limit=1))['value']['query']
                require(query['total'] == 64*len(part['subjects']),'Native matched query omitted cases')
                for summary in query['summary']:
                    values = [o['value'] for o in packet['observations'] if o['subject_id'] == summary['subject_id'] and o['metric_id'] == metric]
                    require(summary['states'] == dict(measured=64,failed=0,unsupported=0,unavailable=0)
                            and struct.pack('!d',summary['maximum']['value']) == struct.pack('!d',max(values)),
                            'Native matched scalar/maximum differs')
        state.update(status='four_readable_matched_GPU_imports_native_metadata_and_maxima_passed',finished_unix=time.time(),
            active_request=None,measured_scalars=24960,original_fits=35,assembly_executions=4,source_count=29,project_left_open=True)
    except BaseException as error:
        state.update(status='original_matched_request_requires_reconciliation',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        save(STATUS,state)


def after():
    require(not REPORT.exists(),'Preserve original matched import custody')
    prep,_ = qualified()
    previous,state = read(BEFORE),read(STATUS)
    require(state['status'] == 'four_readable_matched_GPU_imports_native_metadata_and_maxima_passed'
            and not Path(f'/proc/{state["pid"]}').exists() and len(state['imports']) == 4,'All original matched imports/queries must finish')
    raw,current = header()
    require(current['checkpoint'] == state['last_acknowledged_checkpoint'],'Acknowledged matched checkpoint differs')
    keys = [read(JOB/r['request'])['arguments']['request_key'] for r in state['requests']
        if r['operation'] in ('project_files.source_register','project_files.measurement_import')]
    require(len(keys) == len(set(keys)) == 5 and all(r.get('status') == 200 for r in state['requests']),'Five acknowledged mutations required')
    additions = {('project_operation',current['project_id']+'.'+digest(k.encode())) for k in keys}
    source_hash = prep['source_manifest']['sha256']
    additions.update({('project_snapshot',current['project_id']+'.'+source_hash),('project_artifact',source_hash)})
    for part,admitted in zip(prep['packets'],state['imports'],strict=True):
        additions.update({('project_artifact',part['sha256']),('measurement_source',admitted['origin_catalog_id']+'.'+part['sha256'])})
    indexed = {(e['family'],e['id']):e for e in current['inventory']}
    require(len(additions) == 15 and additions <= indexed.keys() and len(indexed) == previous['canonical_records']+15,
            'Unexpected matched-import record additions')
    old = dict(current,checkpoint=previous['checkpoint'])
    for name in ('inventory','roots'):
        old[name] = [e for e in current[name] if (e['family'],e['id']) not in additions]
    old['receipts'] = [r for r in current['receipts'] if r['request_key'] not in keys]
    require(digest(encode(old)) == previous['header_sha256'],'Original records/roots/receipts changed')
    total = 0
    for entry in current['inventory']:
        path = PROJECT/'.tuldok'/entry['path']
        body = path.read_bytes()
        require(path.is_file() and not path.is_symlink() and len(body) == entry['bytes'] and digest(body) == entry['sha256'],
                'Canonical matched custody body differs')
        total += len(body)
    require(payloads(exclude=previous['expected_new_blobs']) == previous['payloads']
            and payloads()['files'] == previous['payloads']['files']+5,'Prior scientific payload identities changed')
    for path in [PACKET/'source-project.json',*[PACKET/p['path'] for p in prep['packets']]]:
        body = path.read_bytes()
        sha = digest(body)
        require((PROJECT/'artifacts/sha256'/sha).read_bytes() == body
                and read(PROJECT/'.tuldok'/indexed[('project_artifact',sha)]['path']) == dict(format='project_artifact_v1',sha256=sha,bytes=len(body)),
                'Stored matched metadata/declaration differs')
    manifest = read(PACKET/'source-project.json')
    snapshot = read(PROJECT/'.tuldok'/indexed[('project_snapshot',current['project_id']+'.'+source_hash)]['path'])
    require(snapshot['origin_catalog_id'] == current['project_id'] and snapshot['value']['id'] == manifest['id']
            and snapshot['value']['manifest'] == manifest,'Native source snapshot differs')
    for part,admitted in zip(prep['packets'],state['imports'],strict=True):
        ref = next(r for r in manifest['data'] if r['id'] == f'K{part["K"]}-complete-GPU')
        doc = read(PROJECT/'.tuldok'/indexed[('measurement_source',admitted['origin_catalog_id']+'.'+part['sha256'])]['path'])
        expected = dict(origin_catalog_id=current['project_id'],project_id=manifest['id'],manifest_sha256=source_hash,
            reference_id=ref['id'],path=ref['path'],measured_sha256=part['sha256'])
        require(doc['source'] == expected and doc['blob'] == dict(sha256=part['sha256'],bytes=part['bytes']),
                'Native matched source lost exact original binding')
    require((PROJECT/'.tuldok/project.json').read_bytes() == raw,'Checkpoint changed during custody')
    REPORT.write_bytes(encode(dict(status='four_readable_matched_GPU_tables_native_custody_exact',created_unix=time.time(),
        checkpoint=current['checkpoint'],canonical_header_sha256=digest(raw),canonical_records=len(indexed),canonical_record_bytes=total,
        original_header_reconstructed_exactly=True,all_canonical_bodies_freshly_hashed=True,
        prior_scientific_payload_identities_exact=True,old_payload_bodies_rehashed=False,new_metadata_blobs_verified=5,
        original_fits=35,assembly_executions=4,measured_scalars=24960,native_sources=29,project_left_open=True,native_UI_pending=True,
        producer=binding(Path(__file__)),original_owner=binding(STATUS),before=binding(BEFORE),payloads=payloads(),full_portfolio_complete=False)))
    print('Matched native custody exact: 35 fits/4 assemblies/24,960 scalars; UI pending.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('before','import','after'))
    args = parser.parse_args()
    {'before':before,'import':adopt,'after':after}[args.phase]()
