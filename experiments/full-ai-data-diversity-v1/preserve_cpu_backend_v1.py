"""Preserve the completed CPU backend without importing Torch or running inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text())
def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):result.update(block)
    return result.hexdigest()


def run(args):
    args.output=args.output.resolve()
    if args.output.exists():raise ValueError('Fresh CPU reproduction snapshot required')
    source=read(args.protocol);files={}
    def add(relative,expected=None):
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or 'private' in path.parts:
            raise ValueError('Only public repo-relative reproduction artifacts permitted')
        actual=digest(ROOT/path)
        if expected is not None and actual!=expected:raise ValueError('CPU snapshot source changed: '+str(path))
        if str(path) in files and files[str(path)]['sha256']!=actual:raise ValueError('Conflicting snapshot identity')
        files[str(path)]={'sha256':actual,'bytes':(ROOT/path).stat().st_size}
    for relative,expected in source['source_sha256'].items():add(relative,expected)
    for relative in source['benchmark_protocols']:
        add(relative)
        reg=read(ROOT/relative)
        for path,expected in reg['source_sha256'].items():add(path,expected)
        for path in reg['required_terminal_reference_controllers']:add(path)
    for relative in source['extra_artifacts']:add(relative)
    reg=read(ROOT/source['benchmark_protocols'][0]);bank=read(ROOT/reg['bank'])
    references=0
    for row in bank['trajectories']:
        add(row['receipt'],row['receipt_sha256']);receipt=read(ROOT/row['receipt'])
        for state in receipt['states']:
            add(str(Path(row['receipt']).parent/state['path']),state['sha256']);references+=1
    byte_count=sum(row['bytes'] for row in files.values())
    if byte_count>source['maximum_snapshot_bytes']:raise ValueError('Finite CPU snapshot size cap reached')
    if shutil.disk_usage(ROOT).free<byte_count+source['minimum_free_bytes']:raise ValueError('Snapshot free-space floor required')
    began=time.monotonic();destination=args.output/'repo';destination.mkdir(parents=True)
    for relative,spec in files.items():
        if time.monotonic()-began>source['maximum_wall_seconds']:raise TimeoutError('Finite snapshot copy deadline reached')
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        # Reflink where available, otherwise independent bytes; never hard links.
        copied=subprocess.run(['cp','--reflink=auto','--preserve=mode,timestamps','--',str(ROOT/relative),str(target)],capture_output=True)
        if copied.returncode:raise RuntimeError('Snapshot copy failed: '+relative)
        if digest(target)!=spec['sha256'] or digest(ROOT/relative)!=spec['sha256']:
            raise ValueError('CPU snapshot source/destination identity mismatch: '+relative)
    for path,expected in source['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Snapshot protocol dependency changed: '+path)
    report=dict(status='completed_cpu_backend_reproduction_snapshot',created_unix=time.time(),
        original_repo=str(ROOT),snapshot_repo=str(destination.relative_to(ROOT)),
        files=files,total_bytes=byte_count,reference_state_archives=references,
        cpu_model_forward_executed=False,torch_imported=False,optimizer_updates=0,
        independent_copy_or_reflink=True,hard_links_used=False,
        runtime_bundled=False,runtime_requirement='Existing managed Python/Torch installation, pinned by copied runtime manifest; this snapshot does not bundle the5.7GB external runtime.',
        fftw_requirement='Pinned system FFTW library at the original registered path; the digest and rebuild/source provenance remain in the archived protocols.',
        protocol_sha256=digest(args.protocol),source_sha256=source['source_sha256'])
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Preserved CPU backend:',len(files),'files;',byte_count,'bytes;',references,'native state archives. No model inference executed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
