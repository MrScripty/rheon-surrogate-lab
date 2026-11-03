"""Bounded metadata-only profile of native validation on fixed canonical input.

This does not open a ProjectStore/Session, SQL, model runtime or HTTP request.
It isolates codec CPU work from live freshness checks and lifecycle operations.
Artifact access is unavailable; a codec requiring it ends this diagnostic.
"""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from qualify_native_workspace import ROOT
from preview_portfolio_history import save

BASE = ROOT / 'runs/model-portfolio-migration-v1'
REPORT = ROOT / 'docs/plans/model-portfolio-migration-v1/reports/current-validation-profile-01.json'


def run(report=REPORT, history_only=False, *, source=None, project=None,
        phase_only=False, budget_seconds=90):
    source = source or BASE / 'sources/Tuldok-history-material-view'
    project = project or BASE / 'full-current-history-cold-01/root/project'
    if not source.is_relative_to(BASE/'sources') or not project.is_relative_to(ROOT):
        raise ValueError('Explicit task source and workspace portable project required')
    if type(phase_only) is not bool or type(budget_seconds) is not int or not 1 <= budget_seconds <= 600:
        raise ValueError('Explicit profiling mode and bounded diagnostic budget required')
    path = project / '.tuldok/project.json'
    if report.exists() or not report.is_relative_to(ROOT / 'docs/plans/model-portfolio-migration-v1/reports'):
        raise ValueError('Fresh diagnostic report required')
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))
    original = path.read_bytes()
    manifest = json.loads(original)
    result = dict(schema='rheon.native-current-validation-profile.v1', status='reading_fixed_records',
                  checkpoint=manifest['checkpoint'], source_commit=subprocess.check_output(
                      ['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                  source=str(source.relative_to(ROOT)),project=str(project.relative_to(ROOT)),
                  phases=[], models_executed=0, SQL_opened=False, project_written=False,
                  scope=__doc__.strip(), memory_limit_bytes=16*1024**3)
    save(report, result)
    tracked = subprocess.check_output(['git','-C',str(source),'ls-files','-z']).decode().split('\0')
    tracked = sorted(name for name in tracked if name)
    def complete_source_digest():
        entries = [(name,hashlib.sha256((source/name).read_bytes()).hexdigest()) for name in tracked]
        return hashlib.sha256(json.dumps(entries,separators=(',',':')).encode()).hexdigest()
    result['complete_tracked_source_files'] = len(tracked)
    result['complete_tracked_source_sha256'] = complete_source_digest()
    result['canonical_header_sha256'] = hashlib.sha256(original).hexdigest()
    result['producer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['Store_or_Session_constructed'] = False
    result['artifact_bodies_available'] = False
    records = []
    for row in manifest['inventory']:
        raw = (project / '.tuldok' / row['path']).read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Diagnostic source records changed')
        records.append(dict(family=row['family'], id=row['id'], document=json.loads(raw)))
    sys.path.insert(0,str(source))
    import project_projection as native
    source_files=('project_projection.py','project_history.py','project_material.py','project_composition.py',
                  'project_fork.py','project_store.py','agent_api.py')
    result['source_sha256']={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in source_files}
    materials = native.MaterialSnapshot(records) if history_only else None
    def wrapped(name, function):
        def invoke(*args, **kwargs):
            entry=dict(phase=name, started_cpu=time.process_time(), completed=False)
            result['phases'].append(entry);save(report,result)
            value=function(*args, **kwargs)
            entry.update(completed=True, finished_cpu=time.process_time());save(report,result)
            return value
        return invoke
    for name in ('decode_data_records','decode_history_records','decode_file_records',
                 'decode_native_records','decode_producer_records','decode_measurement_records',
                 'MaterialSnapshot','operation_records'):
        setattr(native,name,wrapped(name,getattr(native,name)))
    import project_composition,project_fork,project_transfer
    for module,name in ((project_composition,'decode_records'),(project_fork,'decode_fork'),
                        (project_transfer,'decode_records')):
        setattr(module,name,wrapped(module.__name__+'.'+name,getattr(module,name)))
    operation_totals = {}
    for name in ('validate_data_result','validate_material_result','validate_history_result',
                 'validate_source_result','validate_measurement_result','validate_execution_result',
                 'validate_producer_result','validate_file_result','validate_study_execution_result',
                 'validate_runtime_result','validate_preparation_result','validate_artifact_resolution'):
        original_function = getattr(native,name)
        totals = dict(calls=0,cpu_seconds=0.0,wall_seconds=0.0)
        operation_totals[name] = totals
        def receipt_probe(*args,_function=original_function,_totals=totals,**kwargs):
            cpu,wall=time.process_time(),time.monotonic()
            try:
                return _function(*args,**kwargs)
            finally:
                _totals['calls']+=1
                _totals['cpu_seconds']+=time.process_time()-cpu
                _totals['wall_seconds']+=time.monotonic()-wall
        setattr(native,name,receipt_probe)
    receipt_owners = {}
    for name in ('file_receipt_validator','source_receipt_validator','measurement_receipt_validator'):
        if not hasattr(native,name):
            continue
        make_validator=getattr(native,name)
        totals=dict(candidate_decodes=0,candidate_decode_cpu_seconds=0.0,
                    receipt_calls=0,receipt_cpu_seconds=0.0)
        receipt_owners[name]=totals
        def measure_factory(*args,_make=make_validator,_totals=totals,**kwargs):
            cpu=time.process_time()
            validate=_make(*args,**kwargs)
            _totals['candidate_decodes']+=1
            _totals['candidate_decode_cpu_seconds']+=time.process_time()-cpu
            def measure_receipt(*receipt_args,**receipt_kwargs):
                began=time.process_time()
                try:
                    return validate(*receipt_args,**receipt_kwargs)
                finally:
                    _totals['receipt_calls']+=1
                    _totals['receipt_cpu_seconds']+=time.process_time()-began
            return measure_receipt
        setattr(native,name,measure_factory)
    profiler = cProfile.Profile()
    def expired(_signal, _frame):
        raise TimeoutError('Diagnostic wall budget expired')
    signal.signal(signal.SIGALRM, expired)
    result.update(status='native_codec_profile_running', records=len(records), wall_budget_seconds=budget_seconds,
                  cProfile_enabled=not phase_only,
                  selected_mode='history_only' if history_only else 'complete_projection_validation')
    save(report,result)
    codec_cpu,codec_wall=time.process_time(),time.monotonic()
    signal.alarm(budget_seconds)
    if not phase_only:profiler.enable()
    try:
        if history_only:
            native.decode_history_records(records, materials=materials, artifacts=None)
        else:
            native.ProjectProjection.validate_records(
                SimpleNamespace(project_id=manifest['project_id'], artifacts=None), records)
        result['status']='fixed_record_validation_completed'
    except BaseException as error:
        result.update(status='bounded_profile_incomplete', error_type=type(error).__name__,
                      native_error_code=getattr(error,'code',None))
    finally:
        result['codec_cpu_seconds']=time.process_time()-codec_cpu
        result['codec_wall_seconds']=time.monotonic()-codec_wall
        result['operation_validator_totals']=operation_totals
        result['owned_receipt_validator_totals']=receipt_owners
        if not phase_only:profiler.disable()
        signal.alarm(0)
        stats=[]
        for entry in profiler.getstats():
            code=entry.code
            stats.append(dict(function=code if isinstance(code,str) else code.co_name,
                              file=None if isinstance(code,str) else Path(code.co_filename).name,
                              line=None if isinstance(code,str) else code.co_firstlineno,
                              calls=entry.callcount, recursive_calls=entry.reccallcount,
                              cumulative_seconds=entry.totaltime, self_seconds=entry.inlinetime))
        result['top_cumulative']=sorted(stats,key=lambda r:r['cumulative_seconds'],reverse=True)[:35]
        result['top_self']=sorted(stats,key=lambda r:r['self_seconds'],reverse=True)[:25]
        result['manifest_unchanged']=path.read_bytes()==original
        result['neural_runtime_imported']='torch' in sys.modules
        result['source_unchanged']=result['source_sha256']=={
            name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in source_files}
        result['complete_tracked_source_unchanged']=complete_source_digest()==result['complete_tracked_source_sha256']
        result['peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save(report,result)
    print(json.dumps(dict(status=result['status'], phases=result['phases'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=REPORT)
    parser.add_argument('--history-only',action='store_true')
    parser.add_argument('--source',type=Path)
    parser.add_argument('--project',type=Path)
    parser.add_argument('--phase-only',action='store_true')
    parser.add_argument('--budget-seconds',type=int,default=90)
    args=parser.parse_args()
    run(args.report.resolve(),args.history_only,
        source=args.source.resolve() if args.source is not None else None,
        project=args.project.resolve() if args.project is not None else None,
        phase_only=args.phase_only,budget_seconds=args.budget_seconds)
