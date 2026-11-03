"""One native owner for four qualified full-state GPU measurement tables.

Before/after custody stores compact fingerprints and reconstructs the prior
header in memory. Every native request is saved once before dispatch. A failure
requires explicit reconciliation of its original request, never an automatic
retry. No fit, Trial, reference simulation or model inference is created.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PACKET = RUN/'common-generator-GPU-measurements-01'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
JOB = RUN/'native-common-generator-GPU-import-01'
CONTROLLER = RUN/'native-portfolio-ui-accepted-01'
BEFORE = JOB/'before-custody.json'
STATUS = JOB/'status.json'
REPORT = DOC/'reports/native-common-generator-GPU-import-custody-01.json'


def read(path):
    return json.loads(path.read_bytes())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()


def binding(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=digest(path.read_bytes()))


def require(condition,message):
    if not condition:
        raise ValueError(message)


def save(path,value):
    temporary = path.with_suffix('.pending')
    temporary.write_bytes(encode(value))
    temporary.replace(path)


def header():
    raw = (PROJECT/'.tuldok/project.json').read_bytes()
    value = json.loads(raw)
    require(raw == encode(value) and value['checkpoint'] == digest(encode({k:v for k,v in value.items() if k != 'checkpoint'})),
            'Canonical header encoding or checkpoint differs')
    return raw,value


def payloads(exclude=()):
    rows = []
    for path in sorted((PROJECT/'artifacts/sha256').iterdir()):
        if path.name in exclude:
            continue
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode),'Payload is not an ordinary file')
        rows.append(dict(name=path.name,device=info.st_dev,inode=info.st_ino,bytes=info.st_size,mode=stat.S_IMODE(info.st_mode)))
    return dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),identity_sha256=digest(encode(rows)))


def qualified():
    prep = read(PACKET/'preparation.json')
    proof = read(DOC/'reports/common-generator-GPU-measurements-qualification-01.json')
    require(proof['measured_scalars'] == prep['measured_scalars'] == 6400
            and proof['exact_models'] == 10 and len(proof['queries']) == 400
            and proof['incompatible_K_pair_rejections'] == 6 and len(prep['packets']) == 4,
            'Complete qualified ten-model GPU tables required')
    for spec in [proof['producer'],proof['preparation'],proof['native_owner'],*proof['packet_bindings'],
            prep['producer'],prep['source_manifest']]:
        require(binding(ROOT/spec['path']) == spec,'Qualified packet, producer or native owner changed')
    for part in prep['packets']:
        raw = (PACKET/part['path']).read_bytes()
        require(digest(raw) == part['sha256'] and len(raw) == part['bytes'],'Prepared native packet changed')
    return prep,proof


def before():
    require(not JOB.exists() and not REPORT.exists(),'Preserve original native import owner; reconcile existing work')
    prep,_ = qualified()
    audit = read(DOC/'reports/native-205-method-review-adoption-01.json')
    browser = read(RUN/'native-205-review-browser-01/report.json')
    owner = read(RUN/'method-review-native-adoption-16/status.json')
    raw,current = header()
    require(owner['status'] == 'native_twentysix_note_adoption_acknowledged'
            and not Path(f'/proc/{owner["pid"]}').exists()
            and audit['reviewed_fit_cards'] == 205 and audit['all_non_note_scientific_records_exact']
            and current['checkpoint'] == audit['checkpoint'] == owner['checkpoint']
            and digest(raw) == audit['canonical_header_sha256']
            and browser['status'] == 'passed_native_project_UI_actions' and browser['browser_stopped']
            and not browser['errors'] and any(s.get('text') == current['checkpoint'] for s in browser['steps']),
            'Final205review acknowledgement, custody and actual UI must finish first')
    hashes = [prep['source_manifest']['sha256'],*[p['sha256'] for p in prep['packets']]]
    require(len(set(hashes)) == 5 and all(not (PROJECT/'artifacts/sha256'/h).exists() for h in hashes),
            'Five genuinely new metadata artifacts expected')
    prior_sources = [e['id'].split('.',1) for e in current['inventory'] if e['family'] == 'measurement_source']
    require(len(prior_sources) == 21,'Preserve all21previous measurement sources')
    keep = payloads()
    require(keep == audit['payload_preservation'],'Prior scientific payload identities changed')
    JOB.mkdir()
    save(BEFORE,dict(status='compact_before_custody_captured',created_unix=time.time(),checkpoint=current['checkpoint'],
        header_sha256=digest(raw),canonical_records=len(current['inventory']),project_id=current['project_id'],
        payloads=keep,preexisting_sources=prior_sources,preparation=binding(PACKET/'preparation.json'),
        expected_new_blobs=hashes,no_full_header_or_payload_backup=True))
    print('Captured compact before-custody; no native request dispatched.',flush=True)


def adopt():
    require(BEFORE.exists() and not STATUS.exists() and not list(JOB.glob('*.request.json')),
            'Capture before custody and preserve original request ownership')
    prep,proof = qualified()
    previous = read(BEFORE)
    raw,current_header = header()
    require(current_header['checkpoint'] == previous['checkpoint'] and digest(raw) == previous['header_sha256'],
            'Current checkpoint changed before import')
    controller = read(CONTROLLER/'controller.json')
    source = Path(controller['source'])
    require(controller['status'] == 'ready' and controller['source_commit'] == 'f8ba4cedd22580f273f8c4fcbc06bca53797a7a7'
            and all(Path(f'/proc/{controller[k]}').exists() for k in ('launcher_pid','app_pid'))
            and subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() == controller['source_commit']
            and not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
            and binding(source/'retained_measurements.py') == proof['native_owner'],'Exact clean live UI source required')
    state = dict(status='starting',pid=os.getpid(),created_unix=time.time(),project_id=current_header['project_id'],
        starting_checkpoint=previous['checkpoint'],requests=[],imports=[],source_commit=controller['source_commit'],
        new_model_forwards=0,new_reference_generation=False,dataset_or_weight_copies=0,
        native_UI_accepted=False,full_portfolio_complete=False)
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
            state.update(status='original_native_request_in_flight',active_request=stem,updated_unix=time.time())
            save(STATUS,state)
            tick = time.monotonic()
            result = client.call(operation,arguments)
            save(JOB/(stem+'.response.json'),result)
            receipt.update(status=result['status'],response=stem+'.response.json',wall_seconds=time.monotonic()-tick)
            save(STATUS,state)
            require(result['status'] == 200,'Native rejection retained; reconcile original request without retry')
            envelope = result['result']
            require(envelope['binding'] == current['binding'] and envelope['project_id'] == current_header['project_id'],
                    'Native binding or project changed')
            current['checkpoint'] = envelope['checkpoint']
            state['last_acknowledged_checkpoint'] = current['checkpoint']
            save(STATUS,state)
            print(action,'HTTP200',stem,flush=True)
            return envelope['value']

        require((PROJECT/'.tuldok/project.json').read_bytes() == raw,'Checkpoint changed before first native mutation')
        registered = call('source_register',dict(root='lab',manifest_path=str((PACKET/'source-project.json').relative_to(ROOT))))
        require(registered['project_id'] == read(PACKET/'source-project.json')['id']
                and registered['manifest_sha256'] == prep['source_manifest']['sha256'],'Registered source differs')
        snapshot = {k:registered[k] for k in ('origin_catalog_id','project_id','manifest_sha256')}
        state['source_snapshot'] = snapshot
        save(STATUS,state)
        for part in prep['packets']:
            admitted = call('measurement_import',dict(snapshot=snapshot,reference_id=f'K{part["K"]}-complete-GPU',
                expected_sha256=part['sha256']))
            require(admitted['blob'] == dict(sha256=part['sha256'],bytes=part['bytes']),'Native measurement blob differs')
            state['imports'].append(dict(K=part['K'],**admitted))
            save(STATUS,state)
        sources = call('measurement_sources',dict(offset=0,limit=40))
        expected = {tuple(s) for s in previous['preexisting_sources']}|{(s['origin_catalog_id'],s['id']) for s in state['imports']}
        require(sources['total'] == 25 and {(s['origin_catalog_id'],s['id']) for s in sources['items']} == expected,
                'Existing or new measurement-source population changed')
        for part,admitted in zip(prep['packets'],state['imports'],strict=True):
            metadata = call('measurement_metadata',dict(origin_catalog_id=admitted['origin_catalog_id'],packet_sha256=part['sha256']))['value']
            require((metadata['subject_count'],metadata['case_count'],metadata['observation_count']) ==
                    (len(part['subjects']),64,part['measured']),'Native subject/case/scalar counts differ')
            packet = read(PACKET/part['path'])
            for metric in ('pressure_relative_l2','velocity_relative_l2','per_output_median_ms'):
                query = call('measurement_compare',dict(sources=[dict(origin_catalog_id=admitted['origin_catalog_id'],packet_sha256=part['sha256'])],
                    metric_id=metric,subject_ids=part['subjects'],filters={},offset=0,limit=1))['value']['query']
                require(query['total'] == 64*len(part['subjects']),'Native query omitted physical cases')
                for summary in query['summary']:
                    values = [o['value'] for o in packet['observations'] if o['subject_id'] == summary['subject_id'] and o['metric_id'] == metric]
                    require(summary['states'] == dict(measured=64,failed=0,unsupported=0,unavailable=0)
                            and struct.pack('!d',summary['maximum']['value']) == struct.pack('!d',max(values)),
                            'Native query changed a retained scalar or maximum')
        state.update(status='four_common_generator_GPU_imports_native_metadata_and_maxima_passed',finished_unix=time.time(),
            active_request=None,measured_scalars=6400,models=10,source_count=25,project_left_open=True)
    except BaseException as error:
        state.update(status='original_request_requires_explicit_reconciliation',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        save(STATUS,state)


def after():
    require(not REPORT.exists(),'Preserve original native import custody audit')
    prep,_ = qualified()
    previous,state = read(BEFORE),read(STATUS)
    require(state['status'] == 'four_common_generator_GPU_imports_native_metadata_and_maxima_passed'
            and not Path(f'/proc/{state["pid"]}').exists() and len(state['imports']) == 4,
            'All original imports and native queries must finish')
    raw,current = header()
    require(current['checkpoint'] == state['last_acknowledged_checkpoint'],'Native checkpoint differs from acknowledgement')
    keys = [read(JOB/r['request'])['arguments']['request_key'] for r in state['requests']
        if r['operation'] in ('project_files.source_register','project_files.measurement_import')]
    require(len(keys) == len(set(keys)) == 5 and all(r.get('status') == 200 for r in state['requests']),
            'Exactly five acknowledged mutations required')
    additions = {('project_operation',current['project_id']+'.'+digest(k.encode())) for k in keys}
    source_hash = prep['source_manifest']['sha256']
    additions.update({('project_snapshot',current['project_id']+'.'+source_hash),('project_artifact',source_hash)})
    for part,admitted in zip(prep['packets'],state['imports'],strict=True):
        additions.update({('project_artifact',part['sha256']),('measurement_source',admitted['origin_catalog_id']+'.'+part['sha256'])})
    indexed = {(e['family'],e['id']):e for e in current['inventory']}
    require(len(additions) == 15 and additions <= indexed.keys()
            and len(indexed) == len(current['inventory']) == previous['canonical_records']+15,'Unexpected native record additions')
    old = dict(current,checkpoint=previous['checkpoint'])
    for name in ('inventory','roots'):
        old[name] = [e for e in current[name] if (e['family'],e['id']) not in additions]
    old['receipts'] = [r for r in current['receipts'] if r['request_key'] not in keys]
    require(digest(encode(old)) == previous['header_sha256'],'Original canonical inventory, roots or receipts changed')
    total = 0
    for entry in current['inventory']:
        path = PROJECT/'.tuldok'/entry['path']
        body = path.read_bytes()
        require(path.is_file() and not path.is_symlink() and len(body) == entry['bytes'] and digest(body) == entry['sha256'],
                'Canonical body differs from acknowledged inventory')
        total += len(body)
    hashes = previous['expected_new_blobs']
    require(payloads(exclude=hashes) == previous['payloads'] and payloads()['files'] == previous['payloads']['files']+5,
            'Prior payload membership, identities, sizes or modes changed')
    for path in [PACKET/'source-project.json',*[PACKET/p['path'] for p in prep['packets']]]:
        body = path.read_bytes()
        blob_hash = digest(body)
        require((PROJECT/'artifacts/sha256'/blob_hash).read_bytes() == body,'Native metadata artifact differs from qualified source')
        record = read(PROJECT/'.tuldok'/indexed[('project_artifact',blob_hash)]['path'])
        require(record == dict(format='project_artifact_v1',sha256=blob_hash,bytes=len(body)),'Native metadata artifact declaration differs')
    manifest = read(PACKET/'source-project.json')
    source_record = read(PROJECT/'.tuldok'/indexed[('project_snapshot',current['project_id']+'.'+source_hash)]['path'])
    require(source_record['origin_catalog_id'] == current['project_id']
            and source_record['value']['id'] == manifest['id'] and source_record['value']['manifest'] == manifest,
            'Native registered source differs from qualified manifest')
    for part,admitted in zip(prep['packets'],state['imports'],strict=True):
        reference = next(r for r in manifest['data'] if r['id'] == f'K{part["K"]}-complete-GPU')
        record = read(PROJECT/'.tuldok'/indexed[('measurement_source',admitted['origin_catalog_id']+'.'+part['sha256'])]['path'])
        expected = dict(origin_catalog_id=current['project_id'],project_id=manifest['id'],manifest_sha256=source_hash,
            reference_id=reference['id'],path=reference['path'],measured_sha256=part['sha256'])
        require(record['source'] == expected and record['blob'] == dict(sha256=part['sha256'],bytes=part['bytes']),
                'Native measurement source lost its exact original packet binding')
    require((PROJECT/'.tuldok/project.json').read_bytes() == raw,'Project changed during final custody audit')
    result = dict(status='four_common_generator_GPU_tables_native_custody_exact',created_unix=time.time(),
        checkpoint=current['checkpoint'],canonical_header_sha256=digest(raw),canonical_records=len(indexed),canonical_record_bytes=total,
        all_canonical_bodies_freshly_hashed=True,original_header_reconstructed_exactly=True,
        prior_scientific_payload_identities_exact=True,old_payload_bodies_rehashed=False,new_metadata_blobs_verified=5,
        models=10,measured_scalars=6400,native_sources=25,project_left_open=True,native_UI_pending=True,
        producer=binding(Path(__file__)),original_owner=binding(STATUS),before=binding(BEFORE),
        payloads=payloads(),full_portfolio_complete=False)
    REPORT.write_bytes(encode(result))
    print('Native custody exact: ten generators /6400scalars /four tables; UI verification pending.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('before','import','after'))
    phase = parser.parse_args().phase
    {'before':before,'import':adopt,'after':after}[phase]()
