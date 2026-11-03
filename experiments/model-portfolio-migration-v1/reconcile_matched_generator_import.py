"""Finish a proven pre-admission401 without repeating accepted table imports.

The original owner and request/response bytes remain immutable. A separate
owner reuses the rejected request's exact key/body once, then imports K4 and
checks all four tables. The unchanged original custody verifier consumes this
explicitly reconciled successful-request view.
"""
import argparse
from pathlib import Path
import os
import struct
import subprocess
import sys
import time
import uuid

import import_matched_generator_measurements as original
from import_common_generator_measurements import ROOT,RUN,DOC,PROJECT,CONTROLLER,read,save,header,payloads,encode,digest,binding,require

PRIOR = original.JOB
JOB = RUN/'native-matched-generator-GPU-import-reconciliation-01'
STATUS = JOB/'status.json'
RECONCILIATION = JOB/'before-reconciliation.json'


def rejected_request(state):
    require(state['status'] == 'original_matched_request_requires_reconciliation'
            and not Path(f'/proc/{state["pid"]}').exists() and [i['K'] for i in state['imports']] == [1,2]
            and len(state['requests']) == 4 and [r.get('status') for r in state['requests']] == [200,200,200,401],
            'Terminal original owner with exactly two accepted tables and explicit401 required')
    rejected = state['requests'][-1]
    response = read(PRIOR/rejected['response'])
    require(response == dict(status=401,result=dict(code='unauthenticated',error='A valid credential is required.')),
            'Only the explicit unauthenticated pre-admission response is reconcilable here')
    request = read(PRIOR/rejected['request'])
    require(request['operation'] == 'project_files.measurement_import'
            and request['arguments']['reference_id'] == 'K3-complete-GPU'
            and request['arguments']['expected_checkpoint'] == state['last_acknowledged_checkpoint'],
            'Rejected request must be the original K3 request at the accepted checkpoint')
    return request


def before():
    require(not JOB.exists() and not original.REPORT.exists(),'Preserve the original reconciliation owner')
    prep,_ = original.qualified()
    prior = read(PRIOR/'status.json')
    request = rejected_request(prior)
    previous = read(original.BEFORE)
    raw,current = header()
    require(current['checkpoint'] == prior['last_acknowledged_checkpoint'],'Accepted checkpoint changed after401')
    accepted = prior['requests'][:3]
    keys = [read(PRIOR/r['request'])['arguments']['request_key'] for r in accepted]
    require(request['arguments']['request_key'] not in {r['request_key'] for r in current['receipts']},
            'Rejected request must have no native receipt')
    additions = {('project_operation',current['project_id']+'.'+digest(k.encode())) for k in keys}
    source_hash = prep['source_manifest']['sha256']
    additions.update({('project_snapshot',current['project_id']+'.'+source_hash),('project_artifact',source_hash)})
    for part,admitted in zip(prep['packets'][:2],prior['imports'],strict=True):
        additions.update({('project_artifact',part['sha256']),('measurement_source',admitted['origin_catalog_id']+'.'+part['sha256'])})
    indexed = {(e['family'],e['id']):e for e in current['inventory']}
    require(len(additions) == 9 and additions <= indexed.keys() and len(indexed) == previous['canonical_records']+9,
            'Only source registration and K1/K2 may have been admitted')
    old = dict(current,checkpoint=previous['checkpoint'])
    for name in ('inventory','roots'):
        old[name] = [e for e in current[name] if (e['family'],e['id']) not in additions]
    old['receipts'] = [r for r in current['receipts'] if r['request_key'] not in keys]
    require(digest(encode(old)) == previous['header_sha256'],'Partial import changed prior canonical records')
    require(payloads(exclude=previous['expected_new_blobs']) == previous['payloads']
            and payloads()['files'] == previous['payloads']['files']+3,
            'Exactly the source manifest and K1/K2 metadata blobs must be added')
    for part in prep['packets'][2:]:
        require(not (PROJECT/'artifacts/sha256'/part['sha256']).exists()
                and ('project_artifact',part['sha256']) not in indexed,'An unacknowledged later table already exists')
    require((PROJECT/'.tuldok/project.json').read_bytes() == raw,'Checkpoint changed during reconciliation')
    JOB.mkdir()
    save(RECONCILIATION,dict(status='original401_proven_unadmitted_and_prior_custody_exact',created_unix=time.time(),
        checkpoint=current['checkpoint'],header_sha256=digest(raw),producer=binding(Path(__file__)),
        original_status=binding(PRIOR/'status.json'),original_before=binding(original.BEFORE),
        rejected_request=binding(PRIOR/prior['requests'][-1]['request']),
        rejected_response=binding(PRIOR/prior['requests'][-1]['response']),accepted_tables=[1,2],
        exact_rejected_key_and_body_will_be_reused_once=True,payload_copies=0))
    print('Original401 has no receipt or table; K1/K2 and prior canonical custody exact.',flush=True)


def run():
    require(RECONCILIATION.exists() and not STATUS.exists() and not list(JOB.glob('*.request.json')),
            'Fresh qualified reconciliation owner required; never replay it')
    prep,proof = original.qualified()
    before_proof = read(RECONCILIATION)
    require(binding(PRIOR/'status.json') == before_proof['original_status'],'Original stopped owner changed')
    prior = read(PRIOR/'status.json')
    rejected = rejected_request(prior)
    raw,current_header = header()
    require(current_header['checkpoint'] == before_proof['checkpoint'] and digest(raw) == before_proof['header_sha256'],
            'Reconciled checkpoint changed before dispatch')
    controller = read(CONTROLLER/'controller.json')
    source = Path(controller['source'])
    require(controller['status'] == 'ready' and all(Path(f'/proc/{controller[k]}').exists() for k in ('launcher_pid','app_pid'))
            and subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() == prior['source_commit']
            and not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
            and binding(source/'retained_measurements.py') == proof['native_owner'],'Exact clean live source required')
    state = dict(prior,status='starting_explicit401_reconciliation',pid=os.getpid(),created_unix=time.time(),
        original_stopped_owner=before_proof['original_status'],reconciliation=binding(RECONCILIATION),
        reconciliation_producer=binding(Path(__file__)),requests=[],imports=list(prior['imports']),active_request=None)
    for receipt in prior['requests'][:3]:
        state['requests'].append(dict(receipt,request=str(PRIOR/receipt['request']),response=str(PRIOR/receipt['response'])))
    for name in ('error','error_type','updated_unix'):
        state.pop(name,None)
    save(STATUS,state)
    sys.path.insert(0,str(source))
    import agent_api
    client = agent_api.ProjectFilesOwnerClient(controller['origin'],(CONTROLLER/'private/owner-session.txt').read_text().strip(),
        controller['catalog_id'],timeout=7200)
    try:
        client.bootstrap()
        response = client.call('project_files.status',{})
        save(JOB/'before-status.response.json',response)
        require(response['status'] == 200,'Renewed native status rejected')
        selected = response['result']
        require(selected['state'] == 'open' and selected['project']['project_id'] == prior['project_id']
                and selected['project']['checkpoint'] == current_header['checkpoint']
                and selected['binding'] == rejected['arguments']['binding'],'Actual native selection changed')
        current = dict(binding=selected['binding'],checkpoint=current_header['checkpoint'])

        def call(action,body,exact=None):
            operation = 'project_files.'+action
            if exact is not None:
                arguments = exact['arguments']
                require(exact['operation'] == operation and exact['origin'] == controller['origin']
                        and exact['controller_catalog_id'] == controller['catalog_id']
                        and arguments['expected_checkpoint'] == current['checkpoint'],'Rejected authority/body changed')
            else:
                arguments = dict(body,binding=current['binding'])
                if agent_api.operation(operation)['side_effects']:
                    arguments.update(expected_checkpoint=current['checkpoint'],request_key=uuid.uuid4().hex)
                else:
                    arguments['checkpoint'] = current['checkpoint']
            stem = str(len(state['requests'])+1).zfill(3)+'-'+action
            save(JOB/(stem+'.request.json'),dict(operation=operation,arguments=arguments,
                origin=controller['origin'],controller_catalog_id=controller['catalog_id']))
            receipt = dict(operation=operation,request=stem+'.request.json',outcome='explicit401_reconciliation_once' if exact else 'dispatched_once')
            state['requests'].append(receipt)
            state.update(status='reconciled_native_matched_request_in_flight',active_request=stem,updated_unix=time.time())
            save(STATUS,state)
            tick = time.monotonic()
            response = client.call(operation,arguments)
            save(JOB/(stem+'.response.json'),response)
            receipt.update(status=response['status'],response=stem+'.response.json',wall_seconds=time.monotonic()-tick)
            save(STATUS,state)
            require(response['status'] == 200,'Reconciliation outcome requires inspection; no retry')
            envelope = response['result']
            require(envelope['binding'] == current['binding'] and envelope['project_id'] == prior['project_id'],
                    'Native binding/project changed')
            current['checkpoint'] = envelope['checkpoint']
            state['last_acknowledged_checkpoint'] = current['checkpoint']
            save(STATUS,state)
            print(action,'HTTP200',stem,flush=True)
            return envelope['value']

        for part in prep['packets'][2:]:
            admitted = call('measurement_import',dict(snapshot=state['source_snapshot'],reference_id=f'K{part["K"]}-complete-GPU',
                expected_sha256=part['sha256']),exact=rejected if part['K'] == 3 else None)
            require(admitted['blob'] == dict(sha256=part['sha256'],bytes=part['bytes']),'Imported matched blob differs')
            state['imports'].append(dict(K=part['K'],**admitted))
            save(STATUS,state)
        sources = call('measurement_sources',dict(offset=0,limit=40))
        previous = read(original.BEFORE)
        expected = {tuple(s) for s in previous['preexisting_sources']}|{(s['origin_catalog_id'],s['id']) for s in state['imports']}
        require(sources['total'] == 29 and {(s['origin_catalog_id'],s['id']) for s in sources['items']} == expected,
                'Native matched source population differs')
        for part,admitted in zip(prep['packets'],state['imports'],strict=True):
            metadata = call('measurement_metadata',dict(origin_catalog_id=admitted['origin_catalog_id'],packet_sha256=part['sha256']))['value']
            require((metadata['subject_count'],metadata['case_count'],metadata['observation_count']) ==
                    (len(part['subjects']),64,part['measured']),'Native matched metadata population differs')
            packet = read(original.PACKET/part['path'])
            for metric in ('pressure_relative_l2','velocity_relative_l2','per_output_median_ms'):
                query = call('measurement_compare',dict(sources=[dict(origin_catalog_id=admitted['origin_catalog_id'],packet_sha256=part['sha256'])],
                    metric_id=metric,subject_ids=part['subjects'],filters={},offset=0,limit=1))['value']['query']
                require(query['total'] == 64*len(part['subjects']),'Native query omitted cases')
                for summary in query['summary']:
                    values = [o['value'] for o in packet['observations'] if o['subject_id'] == summary['subject_id'] and o['metric_id'] == metric]
                    require(summary['states'] == dict(measured=64,failed=0,unsupported=0,unavailable=0)
                            and struct.pack('!d',summary['maximum']['value']) == struct.pack('!d',max(values)),
                            'Native matched maximum differs')
        state.update(status='four_readable_matched_GPU_imports_native_metadata_and_maxima_passed',finished_unix=time.time(),
            active_request=None,measured_scalars=24960,original_fits=35,assembly_executions=4,source_count=29,project_left_open=True)
    except BaseException as error:
        state.update(status='reconciled_matched_request_requires_inspection',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        save(STATUS,state)


def after():
    require(binding(PRIOR/'status.json') == read(RECONCILIATION)['original_status'],'Original stopped owner evidence changed')
    original.JOB,original.STATUS = JOB,STATUS
    original.after()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('before','run','after'))
    args = parser.parse_args()
    {'before':before,'run':run,'after':after}[args.phase]()
