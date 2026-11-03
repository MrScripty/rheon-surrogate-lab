"""Measure read-only canonical-store work on an explicitly selected checkpoint.

Compacted inputs are metadata-only diagnostic subjects. This never constructs a
domain projection, opens SQL, resolves artifacts or qualifies a native UI Open.
No metadata or payload fixture copy is made; public snapshot values are hashed
with an independent compact JSON representation outside the measured calls.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from preview_portfolio_history import save

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def fingerprint(snapshot):
    result=hashlib.sha256()
    def add(value):
        result.update(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,
                                 allow_nan=False).encode('utf-8'));result.update(b'\n')
    add({k:v for k,v in snapshot.items() if k!='records'})
    for row in snapshot.get('records',[]):add(row)
    return result.hexdigest()


def run(source,project,manifest_sha,report,*,read_method='snapshot'):
    if (report.exists() or not source.is_relative_to(ROOT/'runs/model-portfolio-migration-v1/sources')
            or not project.is_relative_to(ROOT)):
        raise ValueError('Explicit task source, project and fresh report required')
    if read_method not in ('snapshot','snapshot-summary','summary'):
        raise ValueError('Select a complete snapshot or an explicit scalar-summary mechanism')
    header=project/'.tuldok/project.json';before=header.read_bytes()
    if hashlib.sha256(before).hexdigest()!=manifest_sha:raise ValueError('Pinned canonical metadata changed')
    descriptor=json.loads(before)
    resource.setrlimit(resource.RLIMIT_AS,(20*1024**3,20*1024**3))
    sys.path.insert(0,str(source));import project_store as native
    source_sha=digest(source/'project_store.py')
    counts=dict(record_reads=0,record_bytes=0,decode_calls=0,decode_bytes=0,encode_calls=0,encode_bytes=0)
    read,decode,encode=native.ProjectStore._record_file,native._decode,native._encode
    def counted_read(*args,**kwargs):
        value=read(*args,**kwargs);counts['record_reads']+=1;counts['record_bytes']+=len(value);return value
    def counted_decode(raw):
        counts['decode_calls']+=1;counts['decode_bytes']+=len(raw);return decode(raw)
    def counted_encode(value):
        raw=encode(value);counts['encode_calls']+=1;counts['encode_bytes']+=len(raw);return raw
    native.ProjectStore._record_file=counted_read;native._decode=counted_decode;native._encode=counted_encode
    state=dict(schema='rheon.portable-store-read-work.v1',status='running',created_unix=time.time(),pid=__import__('os').getpid(),
        source=str(source.relative_to(ROOT)),source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        source_worktree_dirty=bool(subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()),
        source_sha256=source_sha,project=str(project.relative_to(ROOT)),checkpoint=descriptor['checkpoint'],
        manifest_sha256=manifest_sha,record_count=len(descriptor['inventory']),
        canonical_record_bytes=sum(r['bytes'] for r in descriptor['inventory']),python=platform.python_version(),
        phases=[],model_forwards=0,SQL_opened=False,payload_copies=0,project_written=False,
        full_native_Open_qualified=False,scope=__doc__.strip(),memory_limit_bytes=20*1024**3,
        read_method=read_method)
    def measured(name,function):
        phase=dict(name=name,started_unix=time.time(),completed=False);state['phases'].append(phase);save(report,state)
        old=dict(counts);cpu=time.process_time();wall=time.monotonic();value=function()
        phase.update(completed=True,cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,
                     work={k:counts[k]-old[k] for k in counts});save(report,state);return value
    store=None
    try:
        store=measured('generic_store_open',lambda:native.ProjectStore.open(project))
        def selected_read():
            if read_method=='summary':return store.summary()
            snapshot=store.snapshot()
            if read_method=='snapshot-summary':
                return {key:snapshot[key] for key in ('project_id','name','checkpoint')}
            return snapshot
        def checked_fingerprint(value):
            expected={key:descriptor[key] for key in ('project_id','name','checkpoint')}
            if {key:value[key] for key in expected}!=expected:
                raise ValueError('Returned metadata differs from the independent pinned manifest')
            if read_method=='snapshot':
                if len(value['records'])!=len(descriptor['inventory']):
                    raise ValueError('Complete snapshot membership differs')
            elif set(value)!=set(expected):
                raise ValueError('Scalar summary has unexpected fields')
            return fingerprint(value)
        first=measured(read_method,selected_read);first_hash=checked_fingerprint(first);del first;gc.collect()
        def rooted():
            with store.root_directory(verify_after=True) as descriptor:__import__('os').fstat(descriptor)
        measured('verified_root_borrow',rooted)
        second=measured('repeat_'+read_method,selected_read);second_hash=checked_fingerprint(second);del second;gc.collect()
        if first_hash!=second_hash:raise ValueError('Returned values changed')
        if header.read_bytes()!=before or digest(source/'project_store.py')!=source_sha:
            raise ValueError('Source or canonical checkpoint changed during measurement')
        state.update(status='completed_read_only_canonical_store_measurement',returned_values_sha256=first_hash,
                     both_returned_values_exact=True,canonical_header_unchanged=True,source_unchanged=True,
                     peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if read_method=='snapshot':
            state.update(snapshot_values_sha256=first_hash,both_snapshot_values_exact=True)
        else:state['both_summary_values_exact']=True
    except BaseException as error:
        state.update(status='failed_or_incomplete_read_measurement',error_type=type(error).__name__)
        raise
    finally:
        if store is not None:store.close()
        state['finished_unix']=time.time();save(report,state)
    print(json.dumps({k:state[k] for k in ['status','record_count','canonical_record_bytes','returned_values_sha256','phases']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--read-method',choices=('snapshot','snapshot-summary','summary'),default='snapshot')
    args=parser.parse_args();run(args.source.resolve(),args.project.resolve(),args.manifest_sha256,args.report.resolve(),read_method=args.read_method)
