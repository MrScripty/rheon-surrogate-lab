"""Serial software-only rollout; one retained native request per Close/Open.

No model, dataset or project copy. The current source changes only after an
acknowledged Close and clean shutdown of its verified owned application/PG.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import uuid

from import_common_generator_measurements import ROOT, DOC, RUN, PROJECT, CONTROLLER, read, save, binding, require
from component_material import observed_file


JOB = RUN/'native-measurement-source-read-rollout-01'
COMMIT = DOC/'reports/measurement-source-reads-source-commit-01.json'


def git(source, *args):
    return subprocess.check_output(['git','-C',str(source),*args], text=True).strip()


def qualified():
    accepted = read(COMMIT)
    observed_file(ROOT, accepted['qualification']['path'], accepted['qualification'])
    proof = read(ROOT/accepted['qualification']['path'])
    require(accepted['status'] == 'qualified_measurement_source_read_commit_retained_not_installed'
            and accepted['distinct_passed_checks'] == proof['distinct_passed_checks'] == 122
            and proof['status'] == 'measurement_source_read_proposal_source_qualified_not_installed'
            and accepted['base'] == 'f8ba4cedd22580f273f8c4fcbc06bca53797a7a7',
            'Exact qualified current-base measurement read proposal required')
    proposal = ROOT/accepted['source']
    require(git(proposal,'rev-parse','HEAD') == accepted['commit']
            and not git(proposal,'status','--porcelain'), 'Clean qualified proposal must remain available')
    for spec in [*accepted['production'],accepted['fixture']]:
        observed_file(ROOT,spec['path'],spec)
    baseline = read(RUN/'measurement-source-reads-native-baseline-01/status.json')
    require(baseline['status'] == 'two_native_measurement_reads_acknowledged'
            and not Path('/proc',str(baseline['pid'])).exists(), 'Original native baseline owner must be terminal')
    return accepted


def custody(phase):
    accepted = qualified()
    output = JOB/(phase+'-custody.json')
    require(not output.exists(), 'Preserve original custody observation')
    started = time.monotonic()
    raw = (PROJECT/'.tuldok/project.json').read_bytes()
    proof = read(DOC/'reports/native-pressure-publication-UI-01.json')
    require(hashlib.sha256(raw).hexdigest() == proof['canonical_header_sha256'], 'Accepted canonical header required')
    header = json.loads(raw)
    require(header['checkpoint'] == proof['checkpoint'], 'Exact accepted checkpoint required')
    total = 0
    for row in header['inventory']:
        relative = Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Confined record paths required')
        path = PROJECT/'.tuldok'/relative
        for part in (path,*path.parents):
            require(not part.is_symlink(), 'No canonical symlink permitted')
            if part == PROJECT:
                break
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size == row['bytes'],
                'Original ordinary single-link canonical file required')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream,'sha256').hexdigest()
        after = path.stat()
        require(digest == row['sha256'] and
                (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns) ==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
                'Canonical record changed or failed its exact digest')
        total += info.st_size
    payloads = []
    for path in sorted((PROJECT/'artifacts/sha256').iterdir()):
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode), 'Ordinary retained payload required')
        payloads.append(dict(name=path.name,device=info.st_dev,inode=info.st_ino,
            bytes=info.st_size,mode=stat.S_IMODE(info.st_mode)))
    info = PROJECT.stat()
    require((PROJECT/'.tuldok/project.json').read_bytes() == raw, 'Project changed during custody read')
    facts = dict(project_id=header['project_id'],checkpoint=header['checkpoint'],header_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_records=len(header['inventory']),canonical_record_bytes=total,
        project_root_identity=[info.st_dev,info.st_ino],payload_files=len(payloads),
        payload_bytes=sum(r['bytes'] for r in payloads),payload_identity_sha256=hashlib.sha256(
            json.dumps(payloads,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    if phase == 'after':
        require(facts == read(JOB/'before-custody.json')['facts'], 'Project or payload identity changed across rollout')
        opened = read(JOB/'open.response.json')
        require(opened['status'] == 200 and opened['result']['state'] == 'open'
                and opened['result']['project']['checkpoint'] == facts['checkpoint'], 'Original timed Open must acknowledge')
    save(output,dict(status='exact_project_custody_'+phase,created_unix=time.time(),facts=facts,
        canonical_bodies_freshly_hashed=True,payload_bodies_rehashed=False,project_or_payload_copies=0,
        new_neural_forwards=0,observation_seconds=time.monotonic()-started,producer=binding(Path(__file__)),
        accepted_source_commit=accepted['commit']))
    print('Exact project custody',phase,facts['canonical_records'],'records;',facts['payload_files'],'payload identities',flush=True)


def owner(action):
    accepted = qualified()
    status_path,request_path = JOB/(action+'-status.json'),JOB/(action+'.request.json')
    require(not status_path.exists() and not request_path.exists(), 'Follow the original lifecycle owner; never replay on timeout')
    require(read(JOB/'before-custody.json')['status'] == 'exact_project_custody_before', 'Pinned before custody required')
    state = read(CONTROLLER/'controller.json')
    source = Path(state['source'])
    expected = accepted['base'] if action == 'close' else accepted['commit']
    require(state['status'] == 'ready' and state['source_commit'] == git(source,'rev-parse','HEAD') == expected
            and not git(source,'status','--porcelain'), 'Exact clean ready installed source required')
    require(binding(PROJECT/'.tuldok/project.json')['sha256'] == read(JOB/'before-custody.json')['facts']['header_sha256'],
            'Project changed before native lifecycle request')
    checkpoint = read(JOB/'before-custody.json')['facts']['checkpoint']
    result = dict(status='reading_native_selection',pid=os.getpid(),action=action,created_unix=time.time(),
        source_commit=expected,origin=state['origin'],catalog_id=state['catalog_id'],producer=binding(Path(__file__)),
        project_or_payload_copies=0,new_neural_forwards=0)
    save(status_path,result)
    sys.path.insert(0,str(source))
    import agent_api
    started = time.monotonic()
    try:
        client = agent_api.ProjectFilesOwnerClient(state['origin'],
            (CONTROLLER/'private/owner-session.txt').read_text().strip(),state['catalog_id'],timeout=7200)
        client.bootstrap()
        response = client.call('project_files.status',{})
        save(JOB/(action+'-before-status.response.json'),response)
        require(response['status'] == 200, 'Native selection must acknowledge before lifecycle dispatch')
        current = response['result']
        if action == 'close':
            require(current['state'] == 'open' and current['project']['checkpoint'] == checkpoint
                    and current['project']['project_id'] == read(JOB/'before-custody.json')['facts']['project_id'],
                    'Close must retain exact original project selection')
        else:
            require(current['state'] == 'no_project_open' and current['project'] is None,
                    'Actual empty workspace required before timed Open')
        arguments = dict(binding=current['binding'],expected_checkpoint=checkpoint if action == 'close' else None,
                         request_key=uuid.uuid4().hex)
        if action == 'open':
            arguments.update(root='lab',directory='projects/rheon-model-portfolio-unified-04')
        save(request_path,dict(operation='project_files.'+action,arguments=arguments,
            origin=state['origin'],catalog_id=state['catalog_id'],source_commit=expected,timeout_seconds=7200))
        result['status'] = 'original_native_request_in_flight'
        save(status_path,result)
        native_started = time.monotonic()
        response = client.call('project_files.'+action,arguments)
        seconds = time.monotonic()-native_started
        save(JOB/(action+'.response.json'),response)
        require(response['status'] == 200, 'Retained lifecycle rejection requires explicit reconciliation')
        if action == 'close':
            require(response['result']['state'] == 'no_project_open' and response['result']['project'] is None,
                    'Close did not acknowledge empty selection')
        else:
            require(response['result']['state'] == 'open' and response['result']['project']['checkpoint'] == checkpoint,
                    'Open did not acknowledge exact retained checkpoint')
        result.update(status='native_'+action+'_acknowledged',finished_unix=time.time(),native_call_wall_seconds=seconds,
            owner_wrapper_wall_seconds=time.monotonic()-started,checkpoint=checkpoint,project_left_open=action == 'open',
            load_after=list(os.getloadavg()))
    except BaseException as error:
        result.update(status='stopped_for_explicit_reconciliation',error_type=type(error).__name__)
        raise
    finally:
        save(status_path,result)
    print('Native',action,'acknowledged:',seconds,'seconds',flush=True)


def install():
    accepted = qualified()
    report = JOB/'source-integration.json'
    require(not report.exists() and not (JOB/'shutdown-request.json').exists(), 'Follow original shutdown/integration; do not replay')
    closed = read(JOB/'close-status.json')
    require(closed['status'] == 'native_close_acknowledged' and not Path('/proc',str(closed['pid'])).exists(),
            'Original Close must acknowledge and its owner must be terminal')
    state = read(CONTROLLER/'controller.json')
    source = Path(state['source'])
    require(state['status'] == 'ready' and state['source_commit'] == git(source,'rev-parse','HEAD') == accepted['base']
            and not git(source,'status','--porcelain'), 'Original clean installed source required')
    subprocess.run(['git','-C',str(source),'merge-base','--is-ancestor',accepted['base'],accepted['commit']],check=True)
    launcher,app = state['launcher_pid'],state['app_pid']
    descriptor = os.pidfd_open(launcher)
    try:
        argv = [value.decode() for value in Path('/proc',str(launcher),'cmdline').read_bytes().split(b'\0') if value]
        app_argv = Path('/proc',str(app),'cmdline').read_bytes().split(b'\0')
        require(any(v.endswith('experiments/model-portfolio-migration-v1/native_project_server.py') for v in argv)
                and '--source' in argv and '--output' in argv
                and Path(argv[argv.index('--source')+1]).resolve() == source
                and Path(argv[argv.index('--output')+1]).resolve() == CONTROLLER
                and str(source/'app.py').encode() in app_argv, 'Actual owned launcher/application identity required')
        save(JOB/'shutdown-request.json',dict(created_unix=time.time(),launcher_pid=launcher,app_pid=app,
            source_commit=accepted['base'],catalog_id=state['catalog_id'],signal_target='verified_launcher_pidfd',
            close_status=binding(JOB/'close-status.json'),direct_app_or_PostgreSQL_signal_sent=False))
        signal.pidfd_send_signal(descriptor,signal.SIGTERM)
    finally:
        os.close(descriptor)
    deadline = time.monotonic()+60
    while time.monotonic() < deadline:
        stopped = read(CONTROLLER/'controller.json')
        if (stopped['status'] == 'stopped' and stopped['app_stopped'] and stopped['postgres_stopped']
                and not any(Path('/proc',str(pid)).exists() for pid in (launcher,app))):
            break
        time.sleep(.2)
    else:
        raise TimeoutError('Original launcher cleanup not complete; source remains unchanged')
    subprocess.run(['git','-C',str(source),'merge','--ff-only',accepted['commit']],check=True)
    require(git(source,'rev-parse','HEAD') == accepted['commit'] and not git(source,'status','--porcelain'),
            'Installed source must match the qualified clean commit')
    result = dict(status='qualified_source_fast_forwarded_after_acknowledged_Close_and_clean_owned_stop',
        created_unix=time.time(),source=str(source.relative_to(ROOT)),old_commit=accepted['base'],accepted_commit=accepted['commit'],
        old_launcher_pid=launcher,old_app_pid=app,old_app_and_PostgreSQL_stopped=True,
        catalog_id=state['catalog_id'],origin=state['origin'],project_or_payload_copies=0,producer=binding(Path(__file__)),
        actual_Open_and_UI_pending=True)
    save(report,result)
    logpath = JOB/'server-launch.log'
    with logpath.open('xb') as log:
        child = subprocess.Popen([str(ROOT/'runs/dependency-migration-v1/app-env/bin/python'),'-B',
            str(ROOT/'experiments/model-portfolio-migration-v1/native_project_server.py'),
            '--output',str(CONTROLLER),'--source',str(source),'--resume','--keep-open'],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    result['resumed_launcher_pid'] = child.pid
    save(report,result)
    deadline = time.monotonic()+60
    while time.monotonic() < deadline:
        current = read(CONTROLLER/'controller.json')
        if current['status'] == 'ready' and current['launcher_pid'] == child.pid:
            require(current['source_commit'] == accepted['commit'] and current['catalog_id'] == state['catalog_id']
                    and current['origin'] == state['origin'] and current['keep_open'], 'Same native installation identity required')
            result.update(status='qualified_source_installed_same_catalog_UI_resumed',ready_unix=time.time(),
                          resumed_app_pid=current['app_pid'],project_Open_pending=True)
            save(report,result)
            print('Qualified source installed; same catalog/port serving; timed project Open next.',flush=True)
            return
        require(child.poll() is None, 'Original resumed launcher stopped; inspect its log without replay')
        time.sleep(.2)
    raise TimeoutError('Observe the original resumed launcher; never start a duplicate')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('custody-before','close','install','open','custody-after'))
    parser.add_argument('--job',type=Path,default=JOB)
    arguments = parser.parse_args()
    JOB = arguments.job.resolve()
    require(JOB.is_relative_to(RUN) and JOB != RUN, 'Task-owned rollout directory required')
    JOB.mkdir(exist_ok=True)
    if arguments.stage.startswith('custody-'):
        custody(arguments.stage.split('-')[1])
    elif arguments.stage == 'install':
        install()
    else:
        owner(arguments.stage)
