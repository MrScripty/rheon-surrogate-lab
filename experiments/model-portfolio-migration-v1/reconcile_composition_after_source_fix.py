"""Explicit same-key publication reconciliation after a proved constructor fix.

Requires the original terminal-before-copy evidence, qualified fixed source,
acknowledged reopened primary and sufficient space including a private snapshot.
No automatic mutation retry. The native server owns all actual publication.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/model-portfolio-migration-v1'
AUTHORING=BASE/'native-composition-authoring-01'


def run():
    controller=BASE/'native-composition-controller-01';state=json.loads((controller/'controller.json').read_bytes())
    expected='84960fb702e4c0f9eae1fc2fdce5e0791fec358a';source=Path(state['source'])
    if (state['status']!='ready' or state['source_commit']!=expected
            or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=expected
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()):
        raise ValueError('Exact qualified clean running source required')
    if str(source/'app.py').encode() not in Path('/proc',str(state['app_pid']),'cmdline').read_bytes().split(b'\0'):
        raise ValueError('Expected controller process is not live')
    terminal=json.loads((ROOT/'docs/plans/model-portfolio-migration-v1/reports/native-composition-attempt-01-terminal.json').read_bytes())
    fixed=json.loads((ROOT/'docs/plans/model-portfolio-migration-v1/reports/native-composition-candidate-reproduction-02.json').read_bytes())
    if terminal['status']!='terminal_before_payload_copy_or_destination_commit' or fixed['status']!='candidate_constructed':
        raise ValueError('Terminal old attempt and real fixed constructor proof required')
    for name,sha in fixed['source_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=sha:raise ValueError('Qualified source changed')
    opened=json.loads((AUTHORING/'open-after-source-fix-01.response.json').read_bytes())
    original=json.loads((AUTHORING/'compose.request.json').read_bytes())
    if (opened['status']!=200 or opened['result']['state']!='open'
            or opened['result']['project']['checkpoint']!=original['arguments']['expected_checkpoint']):
        raise ValueError('Native reopened primary acknowledgement required')
    space=json.loads((AUTHORING/'space-preflight-02.json').read_bytes())
    if space['status']!='sufficient' or shutil.disk_usage(ROOT).free<space['required_free_bytes']:
        raise ValueError('Adequate snapshot/copy/projection headroom required before dispatch')
    out=AUTHORING/'compose-after-source-fix-01'
    if any(out.with_suffix(ext).exists() for ext in ('.request.json','.response.json','.error.json')):
        raise ValueError('Existing reconciliation must be inspected, never automatically replayed')
    sys.path.insert(0,str(source));import agent_api
    client=agent_api.ProjectFilesOwnerClient(state['origin'],(controller/'private/owner-session.txt').read_text().strip(),state['catalog_id'],timeout=300)
    identity=client._request('GET','/api/access/session')
    if identity['status']!=200 or identity['result']['kind']!='owner' or identity['result']['catalog_id']!=state['catalog_id']:
        raise ValueError('Fresh exact same-catalog owner identity required')
    op='project_files.compose';arguments=dict(original['arguments'],binding=opened['result']['binding'])
    if {k:v for k,v in arguments.items() if k!='binding'}!={k:v for k,v in original['arguments'].items() if k!='binding'}:
        raise ValueError('Original logical request must remain exact')
    method,path,body=agent_api.project_request(op,arguments,1)
    request=dict(operation=op,operation_version=1,arguments=arguments,contract_sha256=agent_api.contract(op,1)['sha256'],
        original_request='runs/model-portfolio-migration-v1/native-composition-authoring-01/compose.request.json',
        source_commit=expected,origin=state['origin'],controller_catalog_id=state['catalog_id'],
        authority='Explicit same-logical-key reconciliation after verified terminal failure before copying, native source fix/102 tests, full real candidate success, safe restart and acknowledged fresh primary binding. Only process binding changes; no timeout-triggered automatic retry.')
    with out.with_suffix('.request.json').open('x') as stream:json.dump(request,stream,indent=2)
    try:
        response=client._request(method,path,body,selection=(op,1));agent_api.validate_output(op,response,version=1)
        if response['status']<400:
            value=response['result']
            if value['fork']['request_key']!=arguments['request_key'] or value['fork']['directory']!=arguments['directory']:
                raise ValueError('Native result differs from exact publication request')
    except BaseException as error:
        with out.with_suffix('.error.json').open('x') as stream:
            json.dump(dict(error_type=type(error).__name__,code=getattr(error,'code',None),status=getattr(error,'status',None),
                           request_outcome='Inspect the same retained request and live owner; do not automatically retry.'),stream,indent=2)
        raise SystemExit(1)
    with out.with_suffix('.response.json').open('x') as stream:json.dump(response,stream,indent=2)
    print(json.dumps(dict(status=response['status'],response=str(out.with_suffix('.response.json').relative_to(ROOT)))))


if __name__=='__main__':run()
