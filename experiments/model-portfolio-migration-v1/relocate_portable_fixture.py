"""Move one closed, already published project for a no-copy cold test.

This owns test-fixture transport only. Native project owners still perform
Open, Save and Close. Unknown native operations must be reconciled before
explicit restoration; no exception handler moves a potentially open project.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

from compact_retired_catalog import digest, inactive, safe, write
from verify_history_project import files

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'runs/model-portfolio-migration-v1/sources/Tuldok-composition-source-artifacts'


def state_path(evidence): return evidence.with_suffix('.state.json')


def save_state(evidence, value):
    path = state_path(evidence); temporary = path.with_suffix('.pending')
    write(temporary, value, not temporary.exists()); temporary.replace(path)
    sync(path.parent)


def sync(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)


def selected(path):
    path = Path(path)
    if not path.is_absolute() or not path.is_relative_to(ROOT):
        raise ValueError('Fixture transport is confined to this workspace')
    return safe(ROOT, str(path.relative_to(ROOT)))


def move(source, destination):
    source, destination = selected(source), selected(destination)
    if os.path.lexists(destination): raise ValueError('Absent no-overwrite destination required')
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise ValueError('Same-filesystem relocation required; copying is forbidden')
    sys.path.insert(0, str(SOURCE))
    from project_store import _rename_noreplace
    fd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: _rename_noreplace(fd, str(source.relative_to(ROOT)), str(destination.relative_to(ROOT)))
    finally: os.close(fd)
    sync(source.parent); sync(destination.parent)


def identity(path):
    info = selected(path).stat()
    return [info.st_dev, info.st_ino]


@contextmanager
def closed(project):
    selected(project); inactive(project)
    fd = os.open(project, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if os.fstat(fd).st_uid != os.getuid(): raise ValueError('Owned fixture required')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally: os.close(fd)


def header(project, checkpoint, expected_sha):
    path = selected(project)/'.tuldok/project.json'
    if digest(path) != expected_sha or json.loads(path.read_bytes())['checkpoint'] != checkpoint:
        raise ValueError('Published checkpoint changed')


def inode_map(project, inventory):
    return {name:identity(project/name) for name in inventory}


def prepare(project, destination, original_local, evidence, checkpoint, manifest_sha):
    for path in (project, destination, original_local, evidence): selected(path)
    if (evidence.exists() or state_path(evidence).exists() or os.path.lexists(destination)
            or os.path.lexists(original_local)):
        raise ValueError('Fresh fixture destinations and evidence required')
    if (destination.is_relative_to(project) or original_local.is_relative_to(project)
            or original_local.is_relative_to(destination.parent)):
        raise ValueError('Continuation archive must be outside the confined cold root')
    with closed(project):
        header(project, checkpoint, manifest_sha)
        before = files(project)
        local = project/'.tuldok-local'
        if local.is_symlink(): raise ValueError('Local continuation cannot be a symlink')
        local_files = files(local) if local.exists() else None
        value = dict(schema='rheon.no-copy-portable-fixture.v1', source=str(project.relative_to(ROOT)),
                     destination=str(destination.relative_to(ROOT)), original_local=str(original_local.relative_to(ROOT)),
                     checkpoint=checkpoint, manifest_sha256=manifest_sha, root_identity=identity(project),
                     portable_files=before, portable_inodes=inode_map(project,before), local_files=local_files,
                     original_local_identity=identity(local) if local.exists() else None,
                     helper_sha256=digest(Path(__file__)))
        write(evidence,value,True)
        status=dict(phase='preserving_original_local',evidence_sha256=digest(evidence),payload_copies=0)
        save_state(evidence,status)
        if local.exists(): move(local,original_local)
        status['phase']='relocating_closed_project';save_state(evidence,status)
        move(project,destination)
        header(destination,checkpoint,manifest_sha)
        if identity(destination)!=value['root_identity'] or inode_map(destination,before)!=value['portable_inodes']:
            raise ValueError('Relocation changed physical file identities')
        status['phase']='relocated_for_native_cold_open';save_state(evidence,status)
    return value


def restore(evidence, generated_local):
    selected(evidence);selected(generated_local)
    status=json.loads(state_path(evidence).read_bytes())
    if digest(evidence)!=status['evidence_sha256']:raise ValueError('Fixture evidence changed')
    value=json.loads(evidence.read_bytes())
    project=selected(ROOT/value['source']);destination=selected(ROOT/value['destination'])
    original_local=selected(ROOT/value['original_local'])
    candidates=[p for p in (project,destination) if p.exists() and identity(p)==value['root_identity']]
    if len(candidates)!=1:raise ValueError('Reconcile the original physical project before restoration')
    current=candidates[0]
    if current!=project and os.path.lexists(project):raise ValueError('Original path is occupied')
    with closed(current):
        header(current,value['checkpoint'],value['manifest_sha256'])
        if files(current)!=value['portable_files']:raise ValueError('Cold lifecycle changed portable bytes or modes')
        inodes=inode_map(current,value['portable_files'])
        if any(inodes[k]!=v for k,v in value['portable_inodes'].items() if k.startswith('artifacts/sha256/')):
            raise ValueError('Cold lifecycle copied or replaced an original payload')
        local=current/'.tuldok-local'
        # Reconcile an interruption immediately after either transport rename.
        original_here=local.exists() and identity(local)==value['original_local_identity']
        if value['local_files'] is not None:
            original=local if original_here else original_local
            if not original.exists() or identity(original)!=value['original_local_identity'] or files(original)!=value['local_files']:
                raise ValueError('Original local continuation differs')
        status['phase']='restoring_original_continuation';save_state(evidence,status)
        if local.exists() and not original_here:move(local,generated_local)
        if value['local_files'] is not None and not original_here:move(original_local,local)
        status['phase']='restoring_original_project_path';save_state(evidence,status)
        if current!=project:move(current,project)
        header(project,value['checkpoint'],value['manifest_sha256'])
        if identity(project)!=value['root_identity']:raise ValueError('Original root identity changed')
        status.update(phase='restored_original_path_and_continuation',portable_bytes_modes_exact=True,
                      payload_inodes_unchanged=True,payload_copies=0)
        save_state(evidence,status)
    return status
