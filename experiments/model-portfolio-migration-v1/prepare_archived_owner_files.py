"""Prepare a temporary writable archive catalog without touching its keeper.

The original receipt selects every copied path. Native anchored reads verify
each source file, and the native tree manifest proves the completed copy.
This is an isolated owner workspace, not another retained original backup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

from qualify_native_workspace import ROOT
from preview_portfolio_history import save, sha


def run(output, report):
    base=ROOT/'runs/model-portfolio-migration-v1'
    docs=ROOT/'docs/plans/model-portfolio-migration-v1/reports'
    if output.exists() or report.exists() or not output.is_relative_to(base) or not report.is_relative_to(docs):
        raise ValueError('Fresh task-owned workspace descriptors required')
    proof=json.loads((docs/'archive-retained-file-bindings-01.json').read_bytes())
    qualified=json.loads((docs/'archived-history-preview-05.json').read_bytes())
    receipt_path=ROOT/'runs/dependency-migration-v1/original-schema8-backup-02/backup-receipt.json'
    if (sha(receipt_path)!=proof['original_receipt_sha256']
            or qualified['status']!='native_archive_readonly_preview_completed'
            or qualified['blockers']!={'cessation_proof_unavailable':1}
            or not qualified['postgres_stopped']):
        raise ValueError('Exact archive receipt and successful native content preview required')
    receipt=json.loads(receipt_path.read_bytes())
    required=2*proof['verified_logical_bytes']+6*1024**3
    if shutil.disk_usage('/tmp').free<required:
        raise ValueError('Insufficient system-partition space for copied owner, native backup and reserve')
    os.umask(0o077);output.mkdir(mode=0o700)
    workspace=Path(tempfile.mkdtemp(prefix='rheon-archive-owner-',dir='/tmp'))
    catalog=workspace/'catalog';catalog.mkdir(mode=0o700)
    exchange=workspace/'exchange';exchange.mkdir(mode=0o700)
    destination=workspace/'projects';destination.mkdir(mode=0o700)
    source_path=ROOT/qualified['source_path']
    sys.path.insert(0,str(source_path))
    from project_transfer import _file, parts
    from backup_metadata import tree_manifest
    keeper=Path(proof['keeper'])
    result=dict(schema='rheon.archived-owner-files.v1',status='copying_original_catalog_files',
                created_unix=time.time(),temporary_workspace=str(workspace),catalog=str(catalog),
                exchange=str(exchange),destination_root=str(destination),
                keeper=str(keeper),original_receipt_sha256=proof['original_receipt_sha256'],
                copied_files=0,copied_bytes=0,source_modified=False,SQL_opened=False,
                Dataset_constructed=False,models_executed=0,exact_native_tree_manifest=False,
                required_initial_free_bytes=required,runner_sha256=sha(Path(__file__)),
                retirement_condition='Retire copied working catalog only after native extraction and independent preservation proof; retain original keeper/metadata/receipts.')
    save(report,result);save(output/'workspace.json',result)
    root_fd=None
    try:
        for relative,item in sorted(receipt['files'].items(),key=lambda pair:(pair[0].count('/'),pair[0])):
            parts(relative)
            if item['kind']=='directory':(catalog/relative).mkdir(mode=0o700)
            elif item['kind']!='file':raise ValueError('Unsupported original archive filesystem kind')
        root_fd=os.open(keeper,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        for relative,item in sorted(receipt['files'].items()):
            if item['kind']!='file':continue
            with _file(root_fd,relative,max(item['bytes'],1)) as (fd,info):
                if info.st_size!=item['bytes']:raise ValueError('Retained source size changed')
                digest=hashlib.sha256()
                with (catalog/relative).open('xb') as target:
                    while raw:=os.read(fd,1024*1024):target.write(raw);digest.update(raw)
                if digest.hexdigest()!=item['sha256']:raise ValueError('Retained source bytes changed')
            result['copied_files']+=1;result['copied_bytes']+=item['bytes']
            if result['copied_files']%1000==0:save(report,result)
        result['status']='verifying_native_original_tree_manifest';save(report,result)
        if tree_manifest(catalog)!=receipt['files']:raise ValueError('Copied original archive tree differs')
        result.update(status='exact_original_archive_owner_files_prepared',exact_native_tree_manifest=True,
                      free_system_bytes=shutil.disk_usage('/tmp').free)
    except BaseException as error:
        result.update(status='failed_or_incomplete_archive_owner_files',error_type=type(error).__name__)
        raise
    finally:
        if root_fd is not None:os.close(root_fd)
        save(report,result);save(output/'workspace.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    run(args.output.resolve(),args.report.resolve())
