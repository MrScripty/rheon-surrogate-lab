"""Stage exact current runtime bytes, retiring task-only converter and tests.

No catalog, credentials, service or Run is opened. Existing attempts are immutable.
"""
import argparse
import ast
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import time

MAX_FILE = 8 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
MAX_FILES = 2048
MANIFEST = 'deployment-manifest.json'
NEW_FILES = frozenset(('project_activity.py', 'torch_runtime.py', 'torch_runtimes.py',
    'static/workspace-project-activity.js', 'docs/contracts/managed-torch-runtime.md',
    'docs/contracts/project-workspace.md'))
TREES = ('static', 'extension_schemas', 'prompts', 'tuldok_extension_sdk', 'docs/contracts')
INFERENCE = frozenset('examples/extensions/field-surrogate/' + version + '/' + name
    for version in ('inference-v1', 'inference-v2-private-qa')
    for name in ('infer.py', 'contract.json', 'NOTICE.txt'))


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(source, *arguments):
    result = subprocess.run(['git', '-C', str(source), *arguments], capture_output=True, timeout=15, check=True)
    if len(result.stdout) > MAX_TOTAL:
        raise ValueError('Git metadata exceeds deployment bound')
    return result.stdout


def origin(source):
    return {'git_head': git(source, 'rev-parse', 'HEAD').decode().strip(),
            'git_tree': git(source, 'rev-parse', 'HEAD^{tree}').decode().strip(),
            'branch': git(source, 'rev-parse', '--abbrev-ref', 'HEAD').decode().strip(),
            'status_sha256': sha(git(source, 'status', '--porcelain=v1', '--untracked-files=normal'))}


def eligible(name):
    parts = Path(name).parts
    if (any(p.startswith('.') or p == '__pycache__' or p == 'tests' or p.startswith('test_') for p in parts)
            or name.endswith(('.pyc', '.pyo'))):
        return False
    if len(parts) == 1:
        return name.endswith('.py') or name == 'README.md' or name.startswith('LICENSE') or (
            name.startswith('requirements') and name.endswith(('.txt', '.lock')))
    return any(name.startswith(tree + '/') for tree in TREES) or name in INFERENCE


def selection(source):
    tracked = set(git(source, 'ls-files', '-z').decode().split('\0')) - {''}
    names = {name for name in tracked if eligible(name)}
    names.update(name for name in NEW_FILES if os.path.lexists(source / name))
    for tree in TREES:
        base = source / tree
        if not base.exists():
            continue
        if base.is_symlink() or not base.is_dir():
            raise ValueError('Resource tree must be an ordinary directory: ' + tree)
        for parent, directories, files in os.walk(base, followlinks=False):
            directories[:] = [d for d in directories if d != '__pycache__' and not d.startswith('.')]
            for name in directories + files:
                path = Path(parent) / name
                relative = path.relative_to(source).as_posix()
                if path.is_symlink():
                    raise ValueError('Symlink in deployment resources: ' + relative)
                if path.is_file() and eligible(relative) and relative not in names:
                    raise ValueError('Unreviewed untracked resource: ' + relative)
    if 'app.py' not in names or len(names) > MAX_FILES:
        raise ValueError('Deployment requires app.py and at most 2048 files')
    return sorted(names)


def identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def read_owned(source, name):
    """No-follow component walk with before/after file and directory-entry checks."""
    parts = Path(name).parts
    if not parts or Path(name).is_absolute() or any(p in ('', '.', '..') for p in parts):
        raise ValueError('Invalid deployment path')
    descriptors = [os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)]
    file_fd = None
    try:
        for part in parts[:-1]:
            descriptors.append(os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptors[-1]))
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptors[-1])
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > MAX_FILE:
            raise ValueError('Deployment input is not a bounded single-link regular file: ' + name)
        chunks, remaining = [], MAX_FILE + 1
        while remaining:
            block = os.read(file_fd, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b''.join(chunks)
        after = os.fstat(file_fd)
        entry = os.stat(parts[-1], dir_fd=descriptors[-1], follow_symlinks=False)
        if len(raw) != before.st_size or identity(before) != identity(after) or identity(after) != identity(entry):
            raise ValueError('Deployment source changed during read: ' + name)
        return raw, {'path': name, 'bytes': len(raw), 'sha256': sha(raw), 'mode': stat.S_IMODE(before.st_mode)}, identity(after)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


SMOKE = r'''
import ast,hashlib,importlib,json,os,re,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root))
import psycopg
attempts=[]
def denied(*args,**kwargs):
 attempts.append('database');raise RuntimeError('Deployment smoke forbids database connections')
psycopg.connect=denied
psycopg.Connection.connect=classmethod(denied)
def audit(event,args):
 if event.startswith(('socket.','subprocess.')) or event in ('os.system','os.exec','os.posix_spawn','os.fork'):
  attempts.append(event);raise RuntimeError('Deployment smoke forbids process/network effects')
 if event in ('os.mkdir','os.remove','os.rename','os.link','os.symlink','os.rmdir','os.chmod'):
  attempts.append(event);raise RuntimeError('Deployment smoke forbids file mutations')
 if event=='open' and args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
  attempts.append(event);raise RuntimeError('Deployment smoke forbids file writes')
sys.addaudithook(audit)
modules=[importlib.import_module(p.stem) for p in sorted(root.glob('*.py'))]
import app,agent_api,pipeline,recipe_contract,release_validation,candidate_packages,extension_contract
from tuldok_extension_sdk.manifest import verify_schema_copies
assert app.ROOT==root
registry=agent_api.validate_registry();capabilities=agent_api.capabilities('0'*32)
assert capabilities['operations'] and pipeline.BUILTIN_TRAINERS and pipeline.ADAPTERS
schema_hashes=verify_schema_copies()
host={}
for attr in ('IMPORTED_HOST_FILES','IMPORTED_TORCH_HOST_FILES','IMPORTED_EVIDENCE_HOST_FILES','IMPORTED_BOUND_EVIDENCE_HOST_FILES','IMPORTED_CANONICAL_EVIDENCE_HOST_FILES'):
 values=getattr(recipe_contract,attr);assert values,attr
 for name,digest in values.items():
  assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
  host[name]=digest
for name,digest in release_validation.IMPORTED_DEFINITION_FILES.items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
for row in candidate_packages.SCALAR_A_ADAPTERS.values():
 for name in ('infer.py','contract.json','NOTICE.txt'):
  assert (root/'examples/extensions/field-surrogate'/row['directory']/name).is_file()
static=[];asset_routes={}
for node in ast.walk(ast.parse((root/'app.py').read_bytes())):
 if isinstance(node,ast.Tuple) and len(node.elts)==2 and all(isinstance(x,ast.Constant) and isinstance(x.value,str) for x in node.elts):
  name,kind=[x.value for x in node.elts]
  if kind in ('text/html','text/javascript','text/css','application/json'):
   assert (root/'static'/name).is_file(),name
   static.append(name)
for node in ast.walk(ast.parse((root/'app.py').read_bytes())):
 if isinstance(node,ast.Dict):
  for key,value in zip(node.keys,node.values):
   if isinstance(key,ast.Constant) and isinstance(key.value,str) and isinstance(value,ast.Tuple) and len(value.elts)==2 and all(isinstance(x,ast.Constant) and isinstance(x.value,str) for x in value.elts):
    name,kind=[x.value for x in value.elts]
    if kind in ('text/html','text/javascript','text/css','application/json'):
     asset_routes[key.value]=name
from html.parser import HTMLParser
class Resources(HTMLParser):
 def handle_starttag(self,tag,attrs):
  values=dict(attrs);name=values.get('src') if tag=='script' else values.get('href') if tag=='link' and values.get('rel')=='stylesheet' else None
  if name and not name.startswith(('https:','http:','data:')):
   route=name.split('?',1)[0]
   assert (root/'static'/asset_routes.get(route,route.lstrip('/'))).is_file(),name
for path in (root/'static').glob('*.html'):
 Resources().feed(path.read_text())
assert (root/'prompts/corners.md').read_text().strip()
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
assert not attempts
local={m.__name__:str(Path(m.__file__).resolve().relative_to(root)) for m in modules}
print(json.dumps(dict(status='passed',registry_sha256=registry,operation_count=len(capabilities['operations']),
 builtin_trainers=list(pipeline.BUILTIN_TRAINERS),adapter_count=len(pipeline.ADAPTERS),sdk_schema_sha256=schema_hashes,
 imported_modules=local,host_file_sha256=host,static_route_assets=sorted(set(static)),
 catalog_opened=False,service_started=False,network_or_process_effects=False,files_written=False,torch_imported=False)))
'''


def smoke(directory, python):
    # Isolated Python and a separate empty cwd prevent source/candidate fallback.
    with tempfile.TemporaryDirectory(prefix='tuldok-deployment-smoke-') as scratch:
        result = subprocess.run([str(python), '-I', '-B', '-c', SMOKE, str(directory)], cwd=scratch,
            env={'PATH': '/usr/bin:/bin', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, timeout=45)
    if result.returncode or len(result.stdout) > 1024 * 1024 or len(result.stderr) > 1024 * 1024:
        raise ValueError('Inert deployment smoke failed: ' + result.stderr.decode(errors='replace')[-3000:])
    return json.loads(result.stdout)


def publish_directory(staging, destination):
    # Linux atomic no-replace publication; os.rename can overwrite an empty dir.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise ValueError('Atomic no-replace directory publication is unavailable')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError('Deployment attempt already exists')
        raise OSError(code, os.strerror(code))


def verify(directory):
    raw, _, _ = read_owned(directory, MANIFEST)
    manifest = json.loads(raw)
    expected = {row['path']: row for row in manifest['files']}
    actual = set()
    for parent, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError('Symlink in deployment artifact')
        actual.update((Path(parent) / name).relative_to(directory).as_posix() for name in files)
    if actual != set(expected) | {MANIFEST}:
        raise ValueError('Deployment artifact contains missing or extra files')
    for name, row in expected.items():
        _, measured, _ = read_owned(directory, name)
        if measured != row:
            raise ValueError('Deployment artifact differs from its manifest: ' + name)
    return manifest, sha(raw)


def stage(source, destination, python, *, smoke_runner=smoke):
    started = time.monotonic()
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if source.is_symlink() or os.path.lexists(destination):
        raise ValueError('Choose an absent destination outside the ordinary source root')
    source, destination = source.resolve(strict=True), destination.resolve(strict=False)
    if destination.is_relative_to(source):
        raise ValueError('Choose an absent destination outside the ordinary source root')
    destination.parent.mkdir(parents=True, exist_ok=True)
    before_origin = origin(source)
    names = selection(source)
    staging = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.stage-', dir=destination.parent))
    identities, files, total = {}, [], 0
    try:
        for name in names:
            raw, row, source_identity = read_owned(source, name)
            total += len(raw)
            if total > MAX_TOTAL:
                raise ValueError('Deployment payload exceeds 64 MiB')
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            target.chmod(row['mode'])
            identities[name] = source_identity
            files.append(row)
        smoke_result = smoke_runner(staging, python)
        if selection(source) != names or origin(source) != before_origin:
            raise ValueError('Source selection or Git origin changed while staging')
        for row in files:
            _, measured, source_identity = read_owned(source, row['path'])
            if measured != row or source_identity != identities[row['path']]:
                raise ValueError('Source bytes/identity changed while staging: ' + row['path'])
        manifest = {'schema_version': 1, 'format': 'tuldok_current_deployment_v1',
            'origin': before_origin, 'source': str(source), 'files': files,
            'source_content_sha256': sha(encoded(files)), 'payload_bytes': total,
            'exclusions': ['tools', 'tests', '.github', '.git', 'venvs', 'caches', 'catalogs',
                           'scripts', 'adapters', 'non-runtime examples', 'task/private payloads'],
            'smoke': smoke_result, 'retirement': 'One-time converter and all tests/CI are absent from this runtime artifact. Source Git and migration receipts remain separate.',
            'manifest_custody': 'Payload paths are bound here; the external staging receipt binds these exact manifest bytes.'}
        with (staging / MANIFEST).open('xb') as stream:
            stream.write(encoded(manifest));stream.flush();os.fsync(stream.fileno())
        verified, manifest_sha = verify(staging)
        publish_directory(staging, destination)
        return {'status': 'published', 'destination': str(destination), 'manifest_sha256': manifest_sha,
                'app_sha256': next(row['sha256'] for row in files if row['path'] == 'app.py'),
                'origin': before_origin, 'source_content_sha256': verified['source_content_sha256'],
                'file_count': len(files), 'payload_bytes': total, 'elapsed_s': time.monotonic() - started,
                'source_pre_post_equal': True, 'smoke': smoke_result,
                'catalog_opened': False, 'service_started': False, 'owner_credentials_accessed': False}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if os.path.lexists(args.receipt):
        parser.error('Receipt exists; choose a fresh attempt')
    result = stage(args.source, args.destination, args.python)
    # Never remove a published artifact on receipt failure/unknown acknowledgement.
    # Reconcile its manifest; do not blindly restage into this destination.
    with args.receipt.open('xb') as stream:
        stream.write(encoded(result));stream.flush();os.fsync(stream.fileno())
    print(json.dumps({k: result[k] for k in ('status', 'destination', 'manifest_sha256', 'app_sha256', 'source_content_sha256', 'file_count', 'payload_bytes', 'elapsed_s')}))


if __name__ == '__main__':
    main()
