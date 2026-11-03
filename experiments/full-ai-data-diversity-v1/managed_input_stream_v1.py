"""Portable typed-input validation for large managed fits; stdlib only.

Matches the existing fitting input names and exact Project-owner rule without
materializing a shard in RAM. Checkpoint artifact custody remains the existing
separate typed restore check.
"""
import hashlib
import json
import os
from pathlib import Path
import stat


def streamed_identity(path):
    path=Path(path).absolute()
    if '..' in path.parts:raise ValueError('Input path traversal')
    handles=[]
    parents=[]
    directory=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    handles.append(directory)
    directory_id=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_uid,x.st_gid)
    file_id=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_uid,x.st_gid,x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    try:
        for part in path.parts[1:-1]:
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
            handles.append(child);parents.append((directory,part,directory_id(os.fstat(child))))
            directory=child
        descriptor=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
        handles.append(descriptor)
        before=os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):raise ValueError('Input must be a regular file')
        digest=hashlib.sha256();size=0
        while size<before.st_size:
            chunk=os.read(descriptor,min(1024**2,before.st_size-size))
            if not chunk:raise ValueError('Input truncated during validation')
            digest.update(chunk);size+=len(chunk)
        if (file_id(before)!=file_id(os.fstat(descriptor)) or
                file_id(before)!=file_id(os.stat(path.name,dir_fd=directory,follow_symlinks=False))):
            raise ValueError('Input changed during validation')
        for parent,name,expected in parents:
            if directory_id(os.stat(name,dir_fd=parent,follow_symlinks=False))!=expected:
                raise ValueError('Input parent path changed')
        return dict(bytes=size,sha256=digest.hexdigest())
    finally:
        for descriptor in reversed(handles):os.close(descriptor)


def validate_inputs(args):
    evidence=json.loads(Path(args.input_evidence).read_text())
    if args.mode not in ('train','restore'):raise ValueError('Explicit train or restore mode required')
    static=dict(manifest=args.manifest,normalization=args.normalization,runtime_manifest=args.runtime_manifest,
        **{f'part_{index:02d}':path for index,path in enumerate(args.shards)})
    if getattr(args,'optimization_normalization',None) is not None:
        static['optimization_normalization']=args.optimization_normalization
    artifact_slots={'trained_weights','trained_initial','trained_history','trained_probe','trained_receipt'} if args.mode=='restore' else set()
    if type(evidence) is not dict or set(evidence)!=set(static)|artifact_slots:
        raise ValueError('Unexpected or missing typed fitting/restore inputs')
    owners=set();identities={}
    for slot,path in static.items():
        row=evidence[slot]
        if type(row) is not dict or not {'reference_id','measured_sha256','project_id','manifest_sha256'}<=set(row):
            raise ValueError('Static inputs must carry exact Project custody')
        measured=streamed_identity(path)
        if row['reference_id']!=slot or row['measured_sha256']!=measured['sha256']:
            raise ValueError('Typed static input identity failed')
        owners.add((row['project_id'],row['manifest_sha256']));identities[slot]=measured
    if len(owners)!=1:raise ValueError('Mixed Project inputs rejected')
    return evidence,identities
