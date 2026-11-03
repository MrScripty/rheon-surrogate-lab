"""Import/read the real retained scalar packet through one owned native HTTP host.

No original service credential, database, numerical runtime or producer is used.
Each request is retained before one dispatch. Unknown outcomes are not retried.
Import and independent cold read use distinct owned controllers and directories.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import uuid


ROOT=Path(__file__).resolve().parents[2]
PACKET=ROOT/'runs/model-portfolio-migration-v1/native-measurements-packet-05/measurements.json'
MANIFEST='runs/model-portfolio-migration-v1/native-measurements-packet-05/source-project.json'


def save(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def run(controller,output,report,phase,root,directory):
    if output.exists() or report.exists():
        raise ValueError('Fresh retained request/report paths required; reconcile existing outcomes explicitly')
    state=json.loads((controller/'controller.json').read_text())
    source=Path(state['source']).resolve()
    if (state['status']!='ready' or not source.is_relative_to(ROOT/'runs/model-portfolio-migration-v1/sources')
            or not (source/'project_measurements.py').is_file()):
        raise ValueError('One ready task-owned native measurement host required')
    sys.path.insert(0,str(source))
    import agent_api
    from retained_measurements import MeasurementTable
    raw=PACKET.read_bytes();expected=MeasurementTable(raw)
    packet=json.loads(raw)
    output.mkdir(parents=True)
    result=dict(schema='rheon.native-retained-measurements-actual.v1',status='running',phase=phase,
        created_unix=time.time(),controller=str(controller.relative_to(ROOT)),source=state['source'],
        source_commit=state['source_commit'],runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        packet_sha256=expected.sha256,packet_bytes=len(raw),new_model_forwards=0,
        reference_generation=False,original_service_accessed=False,UI_qualified=False,
        full_portfolio_qualified=False,requests=[])
    save(report,result)
    client=agent_api.ProjectFilesOwnerClient(state['origin'],
        (controller/'private/owner-session.txt').read_text().strip(),state['catalog_id'],timeout=60)
    client.bootstrap()
    def call(action,body=None):
        name='project_files.'+action
        status=client.call('project_files.status',{})
        if status['status']!=200:raise ValueError('Current native selection is unavailable; no action dispatched')
        current=status['result'];arguments=dict(body or {})
        contract=agent_api.operation(name)
        arguments['binding']=current['binding']
        if contract['side_effects']:
            arguments['expected_checkpoint']=current['project']['checkpoint'] if current['project'] else None
            arguments['request_key']=uuid.uuid4().hex
        else:
            if current['project'] is None:raise ValueError('Read requires a currently open project')
            arguments['checkpoint']=current['project']['checkpoint']
        number=len(result['requests'])+1
        stem=f'{number:03d}-{action}'
        request=dict(operation=name,arguments=arguments,controller_catalog_id=state['catalog_id'],origin=state['origin'])
        save(output/(stem+'.request.json'),request)
        entry=dict(operation=name,request=stem+'.request.json',outcome='dispatched_once')
        result['requests'].append(entry);save(report,result)
        response=client.call(name,arguments)
        save(output/(stem+'.response.json'),response)
        entry.update(status=response['status'],response=stem+'.response.json');save(report,result)
        if response['status']!=200:raise RuntimeError('Retained native request returned an error; inspect before any retry')
        return response['result']
    try:
        initial=client.call('project_files.status',{})
        if initial['status']!=200 or initial['result']['project'] is not None:
            raise ValueError('Start with the fresh owned controller and no selected project')
        if phase=='import':
            call('new',dict(root=root,directory=directory,name='Retained C106 C107 scalar rehearsal'))
            registered=call('source_register',dict(root=root,manifest_path=MANIFEST))['value']
            snapshot={key:registered[key] for key in ('origin_catalog_id','project_id','manifest_sha256')}
            admitted=call('measurement_import',dict(snapshot=snapshot,reference_id='pressure-measurements',
                                                   expected_sha256=expected.sha256))
            origin=admitted['project_id']
        else:
            opened=call('open',dict(root=root,directory=directory))
            origin=opened['project']['project_id']
        sources=call('measurement_sources')['value']['items']
        matches=[doc for doc in sources if doc['id']==expected.sha256]
        if len(matches)!=1:raise ValueError('The real packet source must be retained exactly once')
        selected=matches[0]
        # A cold fork could retain a different origin; use its actual native
        # immutable source identity rather than inventing a new capture fact.
        selection=dict(origin_catalog_id=selected['origin_catalog_id'],packet_sha256=expected.sha256)
        metadata=call('measurement_metadata',selection)
        if metadata['value']['value']!=expected.metadata():raise ValueError('Native metadata differs from exact packet')
        for offset in range(0,expected.metadata()['subject_count'],40):
            page=call('measurement_subjects',dict(selection,offset=offset,limit=40))['value']['value']
            if page!=expected.subjects(offset=offset,limit=40):raise ValueError('Native subject page differs')
        measured=unmeasured=0;totals={};inspections=[]
        for metric in packet['metrics']:
            metric_id=metric['id'];wanted=expected.query(metric_id,limit=100)
            totals[metric_id]=wanted['summary']
            for offset in range(0,wanted['total'],100):
                query=call('measurement_query',dict(selection,metric_id=metric_id,offset=offset,limit=100))['value']
                if query['source']!=metadata['value']['source']:raise ValueError('Native query source selection differs')
                query=query['value'];reference=expected.query(metric_id,offset=offset,limit=100)
                if query!=reference:raise ValueError('Native complete scalar page differs')
                for actual,saved in zip(query['rows'],reference['rows'],strict=True):
                    if actual['state']=='measured':
                        if struct.pack('!d',actual['value'])!=struct.pack('!d',saved['value']):
                            raise ValueError('Binary64 scalar changed')
                        measured+=1
                    else:unmeasured+=1
            for summary in wanted['summary']:
                maximum=summary['maximum']
                identity=maximum['id'] if maximum is not None else next(row['id'] for row in packet['observations']
                    if row['subject_id']==summary['subject_id'] and row['metric_id']==metric_id)
                row=call('measurement_inspect',dict(selection,observation_id=identity))['value']['value']
                if row!=expected.inspect(identity):raise ValueError('Native original case/scalar inspection differs')
                inspections.append(identity)
        call('save');closed=call('close')
        if closed['state']!='no_project_open':raise ValueError('Native Close did not release the selected project')
        result.update(status='passed_actual_native_scalar_import' if phase=='import' else 'passed_actual_native_scalar_cold_read',
            project_id=origin,source_origin_catalog_id=selection['origin_catalog_id'],
            native_source=metadata['value']['source'],measured_scalars_exact=measured,
            explicit_unmeasured_observations=unmeasured,summaries=totals,inspected_observations=inspections,
            Save_Close_qualified=True,finished_unix=time.time())
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error),
                      next_action='Inspect retained request/response and owned host; no automatic retry')
        raise
    finally:save(report,result)
    print(json.dumps({key:result[key] for key in ('status','project_id','measured_scalars_exact',
        'explicit_unmeasured_observations','Save_Close_qualified')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--phase',choices=('import','read'),required=True)
    parser.add_argument('--root',required=True)
    parser.add_argument('--directory',required=True)
    args=parser.parse_args()
    run(args.controller.resolve(),args.output.resolve(),args.report.resolve(),args.phase,args.root,args.directory)
