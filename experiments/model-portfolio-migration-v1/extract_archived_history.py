"""Native owner Stop, backup and extraction of the isolated original archive.

Only the prepared temporary working catalog and its restored private SQL are
opened for ownership. The keeper and original services remain read-only.
"""
import argparse
from collections import Counter
import faulthandler
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

from qualify_native_workspace import ROOT, PG
from preview_portfolio_history import ROOTS, TABLES, sha, save

BASE=ROOT/'runs/model-portfolio-migration-v1'
DOCS=ROOT/'docs/plans/model-portfolio-migration-v1/reports'


def bind(source, original, roots):
    proof=json.loads((DOCS/'archive-retained-file-bindings-01.json').read_bytes())
    index=ROOT/proof['binding_index']
    if sha(index)!=proof['binding_index_sha256']:raise ValueError('Archive file binding index changed')
    by_sha={row['sha256']:row for row in json.loads(index.read_bytes())['bindings']}
    import re
    referenced=set(re.findall(r'[a-f0-9]{64}',json.dumps(original['tables'],separators=(',',':'))))
    bound=set()
    for digest in sorted(referenced & by_sha.keys()):
        row=by_sha[digest]
        source.bind_payload(digest,row['bytes'],'archive-catalog',row['relative']);bound.add(digest)
    recovery=json.loads((DOCS/'archive-manifest-recovery-01.json').read_bytes())
    index=ROOT/recovery['binding_index']
    if sha(index)!=recovery['binding_index_sha256']:raise ValueError('Archive manifest binding index changed')
    for row in json.loads(index.read_bytes())['bindings']:
        if row['root']!='current-lab' or row['sha256'] not in referenced:
            raise ValueError('Recovered manifest is not an original archive reference')
        source.bind_payload(row['sha256'],row['bytes'],row['root'],row['relative']);bound.add(row['sha256'])
    recovered=json.loads((BASE/'full-current-host-recovery-01/recovery.json').read_bytes())
    recovered_by_sha={r['sha256']:ROOT/r['recovered_path'] for r in recovered['rows']
                      if r['state']=='verified_exact_original_host_bytes'}
    hosts=0
    for row in source.db.execute('SELECT document FROM workspace_runs ORDER BY id'):
        run=json.loads(row[0])
        for name,digest in run.get('material',{}).get('host_files',{}).items():
            if digest not in bound:
                candidates=[('current-lab',recovered_by_sha[digest])] if digest in recovered_by_sha else []
                candidates += [(alias,roots[alias]/name) for alias in (
                    'archive-original-host','migration-host','migration-original-host')]
                for alias,path in candidates:
                    if path.is_file() and not path.is_symlink() and sha(path)==digest:
                        source.bind_payload(digest,path.stat().st_size,alias,str(path.relative_to(roots[alias])))
                        bound.add(digest);break
            if digest not in bound:raise ValueError('Original host source is unavailable')
            hosts+=1
    return dict(bound_payloads=len(bound),host_file_observations=hosts)


def run(source, qualification, report):
    if (report.exists() or not report.is_relative_to(DOCS) or not qualification.is_relative_to(DOCS)
            or not source.is_relative_to(BASE/'sources')):
        raise ValueError('Fresh owned report and explicit native source required')
    q=json.loads(qualification.read_bytes())
    if (not q['source_stable_through_tests'] or q['test_returncode']!=0 or not q['postgres_stopped']
            or q['source_path']!=str(source.relative_to(ROOT))
            or any(sha(source/name)!=expected for name,expected in q['source_sha256'].items())):
        raise ValueError('Native source does not match its completed qualification')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True):
        raise ValueError('Qualified source must be committed before native ownership')
    prepared=json.loads((DOCS/'archived-owner-files-01.json').read_bytes())
    restored=json.loads((DOCS/'archived-history-metadata-restore-01.json').read_bytes())
    if prepared['status']!='exact_original_archive_owner_files_prepared' or not prepared['exact_native_tree_manifest']:
        raise ValueError('Exact prepared archive working catalog required')
    workspace=Path(prepared['temporary_workspace']);catalog=Path(prepared['catalog'])
    exchange_root=Path(prepared['exchange']);destination=Path(prepared['destination_root'])
    exchange=exchange_root/report.stem
    cluster=Path(restored['physical_cluster'])
    if (workspace.parent!=Path('/tmp') or not workspace.name.startswith('rheon-archive-owner-')
            or catalog!=workspace/'catalog' or exchange_root!=workspace/'exchange' or exchange.exists()
            or destination!=workspace/'projects' or (cluster/'postmaster.pid').exists()
            or (exchange/'stopped-copy').exists() or (destination/'rheon-archived-history-01').exists()):
        raise ValueError('Fresh prepared owned archive exchange and stopped SQL required')
    original_path=ROOT/'runs/dependency-migration-v1/original-schema8-backup-02/metadata.json'
    if sha(original_path)!=restored['source_snapshot_sha256']:raise ValueError('Original archived metadata changed')
    original=json.loads(original_path.read_bytes())
    receipt=json.loads((original_path.parent/'backup-receipt.json').read_bytes())
    if sha(original_path.parent/'backup-receipt.json')!=prepared['original_receipt_sha256']:
        raise ValueError('Original archive receipt changed')
    if shutil.disk_usage(workspace).free<prepared['copied_bytes']+12*1024**3:
        raise ValueError('Archive native backup/publication reserve unavailable')
    os.umask(0o077);exchange.mkdir(mode=0o700)
    socket=Path(tempfile.mkdtemp(prefix='rheon-archive-owner-pg-',dir='/tmp'))
    dsn=f'dbname=postgres host={socket} port=56979'
    env=dict(os.environ,TULDOK_DATABASE_URL=dsn,PYTHONDONTWRITEBYTECODE='1',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    roots={k:v for k,v in ROOTS.items() if v.is_dir()}
    roots.update({'archive-catalog':Path(prepared['keeper']),
                  'archive-original-host':ROOT/'runs/live-integration/Tuldok'})
    selected=json.loads((BASE/'archived-history-preview-05/selection.json').read_bytes())
    result=dict(schema='rheon.native-archived-history-extraction.v1',status='starting_owned_archive_SQL',
                created_unix=time.time(),source=str(source.relative_to(ROOT)),
                source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                qualification=str(qualification.relative_to(ROOT)),qualification_sha256=sha(qualification),
                archive_metadata_sha256=restored['source_snapshot_sha256'],runner_sha256=sha(Path(__file__)),
                temporary_workspace=str(workspace),catalog=str(catalog),destination=str(destination/'rheon-archived-history-01'),
                exchange=str(exchange),
                original_service_changed=False,original_database_opened=False,keeper_written=False,
                copy_owner_started=False,copy_stop_lease_issued=False,project_extracted=False,
                new_neural_forwards=0,postgres_stopped=False,phases=[])
    def phase(name):
        result['status']=name;result['phases'].append(name);save(report,result);print(name,flush=True)
    def retain(name,preview):
        with gzip.open(exchange/(name+'.json.gz'),'wt',encoding='utf-8',compresslevel=1) as f:
            json.dump(preview,f,ensure_ascii=False,allow_nan=False);f.write('\n')
        return dict(can_extract=preview['can_extract'],report_sha256=preview['report_sha256'],
                    included_families=dict(Counter(r['family'] for r in preview['included'])),
                    blockers=dict(Counter(r['code'] for r in preview['blockers'])),
                    artifacts=dict(Counter(r['state'] for r in preview['artifacts'])))
    dataset=lease=None;backup_fd=destination_fd=None
    stack=(exchange/'python-stacks.log').open('x')
    faulthandler.register(signal.SIGUSR1,file=stack,all_threads=True)
    def interrupted(_sig,_frame):raise InterruptedError('Owned archive extraction interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    save(report,result)
    try:
        with (exchange/'postgres-start.log').open('x') as log:
            subprocess.run([str(PG/'pg_ctl'),'-D',str(cluster),'-l',str(exchange/'postgres.log'),
                            '-o',f"-h '' -k {socket} -p 56979 -c max_connections=20 -c shared_buffers=32MB",
                            '-w','-t','45','start'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
        sys.path.insert(0,str(source))
        from app import Dataset
        from backup_metadata import tree_manifest,table_state
        from project_transfer import ProjectTransfer,preview_source
        from psycopg import sql
        current_tree=tree_manifest(catalog)
        if any(current_tree.get(path)!=value for path,value in receipt['files'].items()):
            raise ValueError('Prepared original file bytes changed before ownership')
        additions={p:v for p,v in current_tree.items() if p not in receipt['files']}
        journal='workspace-runs/.coordinator-v1.json'
        if journal in additions:
            prior=json.loads((DOCS/'archived-history-extraction-01.json').read_bytes())
            observed=json.loads((DOCS/'archived-owner-start-filesystem-diff-01.json').read_bytes())
            if (prior['catalog']!=str(catalog) or not prior['copy_owner_started']
                    or prior['copy_stop_lease_issued'] or not prior['postgres_stopped']
                    or observed['missing_original_paths'] or observed['changed_original_paths']
                    or additions[journal]!=observed['added_paths'].get(journal)):
                raise ValueError('Native prior-start journal has no exact stopped-owner provenance')
            result['prior_native_coordinator_journal']=additions.pop(journal)
        if any(value!={'kind':'directory'} for value in additions.values()):
            raise ValueError('Prepared owner has unaccounted additional file bytes')
        result['native_empty_working_directories_from_prior_start']=sorted(additions)
        phase('starting_real_archive_copy_owner')
        dataset=Dataset(catalog,dsn=dsn,project_roots=roots);result['copy_owner_started']=True
        if 'torch' in sys.modules:raise ValueError('Archive owner imported the neural runtime')
        with dataset.lock:
            for table in TABLES:
                cursor=dataset.db.execute(sql.SQL('SELECT * FROM {}').format(sql.Identifier(table)))
                actual=table_state([c.name for c in cursor.description],cursor.fetchall())
                if table not in original['tables']:
                    if actual['count']!=0:raise ValueError('Schema upgrade added nonempty scientific state')
                    result.setdefault('new_empty_scientific_tables',[]).append(table)
                elif actual!=original['tables'][table]:
                    result['mismatched_table']=table;raise ValueError('Archive copy startup changed original scientific rows')
            result['all_original_scientific_tables_exact_after_owner_start']=True
        token=dataset.access.exchange(dataset.access._local_owner_token())['token']
        backup_fd=os.open(exchange,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        destination_fd=os.open(destination,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        with dataset.access.bind(token):
            phase('previewing_real_archive_copy_owner')
            with preview_source(dataset) as view:
                result['live_bindings']=bind(view,original,roots)
                preview=ProjectTransfer(view).preview(selected)
            result['live_preview']=retain('live-preview',preview)
            if result['live_preview']['blockers']!={'cessation_proof_unavailable':1}:
                raise ValueError('Archive copy content is not ready for native Stop')
            del preview
            phase('native_Stop_and_backup_of_owned_archive_copy')
            lease=dataset.stop_for_project_extraction(backup_fd,'stopped-copy')
            result['copy_stop_lease_issued']=True
            result['native_backup_receipt_sha256']=sha(exchange/'stopped-copy/backup-receipt.json')
            result['stopped_bindings']=bind(lease,original,roots)
            phase('previewing_real_stopped_archive_copy')
            transfer=ProjectTransfer(lease);preview=transfer.preview(selected)
            result['stopped_preview']=retain('stopped-preview',preview)
            if not preview['can_extract']:raise ValueError('Native stopped archive preview is not extractable')
            expected=preview['report_sha256'];del preview
            phase('extracting_native_archived_history_project')
            extracted=transfer.extract(selected,expected_report_sha256=expected,
                destination_parent_fd=destination_fd,directory='rheon-archived-history-01',
                name='Rheon archived model history',request_key='rheon-archived-history-extraction-01')
            if extracted['committed'] is not True:
                raise ValueError('Native archive publication is not acknowledged committed')
            save(exchange/'extraction-receipt.json',extracted)
            result.update(project_extracted=True,extraction_receipt_sha256=sha(exchange/'extraction-receipt.json'),
                          receipt=extracted)
            phase('completed_native_archived_history_extraction')
    except BaseException as error:
        result.update(status='failed_or_incomplete_archived_history_extraction',error_type=type(error).__name__,
                      native_error_code=getattr(error,'code',None),
                      error_frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name)
                                    for f in traceback.extract_tb(error.__traceback__)])
        if type(error).__name__ in ('WorkbenchError','ProjectStoreError'):result['native_error_message']=str(error)
        if getattr(error,'staging_directory',None):result['unpublished_staging_directory']=error.staging_directory
        if getattr(error,'receipt',None) is not None:result['publication_reconciliation_receipt']=error.receipt
        raise
    finally:
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,signal.SIG_IGN)
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGUSR1})
        faulthandler.unregister(signal.SIGUSR1);signal.signal(signal.SIGUSR1,signal.SIG_IGN)
        signal.pthread_sigmask(signal.SIG_SETMASK,mask);stack.close()
        if lease is not None:lease.close()
        if dataset is not None:dataset.close()
        for fd in (backup_fd,destination_fd):
            if fd is not None:os.close(fd)
        stopped=subprocess.run([str(PG/'pg_ctl'),'-D',str(cluster),'-m','fast','-w','-t','30','stop'],capture_output=True,text=True,timeout=45)
        (exchange/'postgres-stop.log').write_text(stopped.stdout+stopped.stderr)
        result['postgres_stopped']=stopped.returncode==0 or 'no server running' in stopped.stdout+stopped.stderr
        if result['postgres_stopped']:shutil.rmtree(socket)
        save(report,result)
    print(json.dumps({k:v for k,v in result.items() if k!='receipt'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--qualification',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    try:run(args.source.resolve(),args.qualification.resolve(),args.report.resolve())
    except BaseException as error:
        print('Native archive extraction failed:',type(error).__name__,file=sys.stderr);raise SystemExit(1)
