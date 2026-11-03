"""Linux per-attempt child custody. Runs owns admission, scheduling and publication.

Trusted reviewed recipes share the local UID. Subreaping is lifecycle ownership,
not filesystem/network confinement. A missing drain receipt never proves cessation.
"""
import base64
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import selectors
import signal
import socket
import stat
import subprocess
import time

READY_SECONDS = 10
GRACE_SECONDS = 3
CLEANUP_SECONDS = 10
STREAM_LIMIT = 1024 * 1024
OUTPUT_LIMIT = 64 * 1024 * 1024
CAPTURE_FILE_LIMIT = 64 * 1024 * 1024
LARGE_OUTPUT_PROFILE = 'captured_cpu_large_outputs_v1'
LARGE_OUTPUT_LIMIT = 4 * 1024 * 1024 * 1024
LARGE_OUTPUT_FILE_LIMIT = 512 * 1024 * 1024
SPEC_LIMIT = 4 * 1024 * 1024
MODEL_CONTEXT_PATH = 'runtime-bootstrap/model-evidence-context.json'
MODEL_CONTEXT_LIMIT = 64 * 1024
SOURCE = Path(__file__).read_bytes()
HASH = re.compile(r'^[a-f0-9]{64}$')
ID = re.compile(r'^[a-f0-9]{32}$')
# Linux UAPI values: some supported Python builds omit the symbolic constants.
F_ADD_SEALS = getattr(fcntl, 'F_ADD_SEALS', 1033)
IMMUTABLE_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008


class ProcessError(ValueError):
    def __init__(self, message, code='invalid'):
        super().__init__(message)
        self.code = code


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate key')
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ProcessError('Invalid custody JSON.') from error


def text(value, maximum=4096):
    return type(value) is str and 0 < len(value) <= maximum and not any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value)


def absolute(value):
    return text(value) and Path(value).is_absolute() and '..' not in Path(value).parts


CPU_PROFILE = 'captured_cpu_python_v1'
CPU_MANIFEST_LIMIT = 16 * 1024 * 1024
CPU_FILE_LIMIT = 512 * 1024 * 1024
CPU_TOTAL_LIMIT = 2 * 1024 * 1024 * 1024
CPU_FILE_COUNT = 40000
CPU_READY_SECONDS = 60
TORCH_READY_SECONDS = 60
TORCH_VERIFY_SECONDS = 60


def preparation_seconds(spec):
    """Custody readiness budget; independent of the recipe's child wall time."""
    if 'torch_runtime' in spec:
        return TORCH_READY_SECONDS
    return CPU_READY_SECONDS if 'cpu_snapshot' in spec else READY_SECONDS


def verification_seconds(spec):
    if 'torch_runtime' in spec:
        return TORCH_VERIFY_SECONDS
    return CPU_READY_SECONDS if 'cpu_snapshot' in spec else 0


def _torch_owner():
    # launch seals this stdlib-only module into the isolated helper. Ordinary
    # application callers use the same current owner, never a second decoder.
    import torch_runtime
    return torch_runtime


def _cpu_relative(value):
    return (text(value, 1024) and not value.startswith('/') and '\\' not in value
            and len(value.split('/')) <= 32
            and all(part not in ('', '.', '..') for part in value.split('/')))


def _cpu_version_packages(value):
    if (type(value) is not dict or not {'numpy', 'torch'} <= set(value)
            or len(value) > 256 or any(not text(k, 128) or not text(v, 128) for k, v in value.items())):
        raise ProcessError('Invalid captured CPU package inventory.')


def _cpu_launch(value):
    keys = {'python', 'loader', 'home', 'stdlib', 'dynload', 'packages', 'library_dirs'}
    if type(value) is not dict or set(value) != keys:
        raise ProcessError('Invalid captured CPU launch layout.')
    if any(not _cpu_relative(value[key]) for key in keys - {'library_dirs'}):
        raise ProcessError('CPU launch paths must be contained relative paths.')
    dirs = value['library_dirs']
    if (type(dirs) is not list or not 1 <= len(dirs) <= 32
            or not all(_cpu_relative(item) and ':' not in item and '$' not in item for item in dirs)
            or len(set(dirs)) != len(dirs)):
        raise ProcessError('Invalid captured loader search paths.')


def validate_cpu_snapshot_descriptor(value):
    """Pure bounded descriptor validation; cold receipts need no live runtime."""
    keys = {'schema_version', 'profile', 'catalog_id', 'root', 'manifest_sha256',
            'manifest_bytes', 'file_count', 'total_bytes', 'python_version', 'packages', 'launch'}
    if (type(value) is not dict or set(value) != keys or type(value['schema_version']) is not int
            or value['schema_version'] != 1 or value['profile'] != CPU_PROFILE):
        raise ProcessError('Unsupported captured CPU descriptor.')
    if type(value['catalog_id']) is not str or not ID.fullmatch(value['catalog_id']):
        raise ProcessError('Invalid captured CPU catalog identity.')
    if type(value['manifest_sha256']) is not str or not HASH.fullmatch(value['manifest_sha256']):
        raise ProcessError('Invalid captured CPU manifest identity.')
    root = value['root']
    if (not absolute(root) or ':' in root or '$' in root or str(Path(root)) != root or Path(root).name != value['manifest_sha256']
            or Path(root).parent.name != 'runtime-snapshots'):
        raise ProcessError('CPU snapshot root must bind its manifest address.')
    for key, limit in [('manifest_bytes', CPU_MANIFEST_LIMIT), ('file_count', CPU_FILE_COUNT), ('total_bytes', CPU_TOTAL_LIMIT)]:
        if type(value[key]) is not int or not 1 <= value[key] <= limit:
            raise ProcessError('Captured CPU descriptor exceeds its bound.')
    if value['python_version'] != '3.12':
        raise ProcessError('Unsupported captured Python version.')
    _cpu_version_packages(value['packages']); _cpu_launch(value['launch'])
    return decode(encoded(value))


def validate_cpu_manifest(value):
    """One canonical manifest authority shared by capture and launch."""
    keys = {'schema_version', 'profile', 'python_version', 'packages', 'launch', 'files', 'total_bytes'}
    if (type(value) is not dict or set(value) != keys or type(value['schema_version']) is not int
            or value['schema_version'] != 1 or value['profile'] != CPU_PROFILE or value['python_version'] != '3.12'):
        raise ProcessError('Unsupported captured CPU manifest.')
    _cpu_version_packages(value['packages']); _cpu_launch(value['launch'])
    files = value['files']
    if type(files) is not list or not 1 <= len(files) <= CPU_FILE_COUNT:
        raise ProcessError('Captured CPU file count exceeds its bound.')
    seen = {}; total = 0
    for item in files:
        if type(item) is not dict or set(item) != {'path', 'bytes', 'sha256', 'mode', 'kind'}:
            raise ProcessError('Invalid captured CPU file record.')
        path = item['path']
        if not _cpu_relative(path) or path == 'manifest.json' or path in seen:
            raise ProcessError('Duplicate or escaping captured CPU file.')
        if (type(item['bytes']) is not int or not 0 <= item['bytes'] <= CPU_FILE_LIMIT
                or type(item['sha256']) is not str or not HASH.fullmatch(item['sha256'])
                or type(item['mode']) is not int or item['mode'] not in (0o400, 0o500)
                or item['kind'] not in ('elf', 'data')):
            raise ProcessError('Invalid captured CPU file bounds, hash, mode or kind.')
        total += item['bytes']; seen[path] = item
    if type(value['total_bytes']) is not int or total != value['total_bytes'] or not 1 <= total <= CPU_TOTAL_LIMIT:
        raise ProcessError('Captured CPU total bytes differ or exceed bound.')
    if list(seen) != sorted(seen):
        raise ProcessError('Captured CPU inventory must be canonically ordered.')
    directories = set()
    for path in seen:
        parents = {str(parent) for parent in Path(path).parents if str(parent) != '.'}
        if any(parent in seen for parent in parents):
            raise ProcessError('Captured CPU file conflicts with directory path.')
        directories.update(parents)
        if len(directories) > CPU_FILE_COUNT:
            raise ProcessError('Captured CPU directory count exceeds its bound.')
    for key in ('python', 'loader'):
        item = seen.get(value['launch'][key])
        if not item or item['kind'] != 'elf' or item['mode'] != 0o500:
            raise ProcessError('Captured Python and loader must be executable ELF files.')
    for relative in [value['launch'][key] for key in ('home', 'stdlib', 'dynload', 'packages')] + value['launch']['library_dirs']:
        if relative in seen or not any(path.startswith(relative + '/') for path in seen):
            raise ProcessError('Captured CPU launch directory is absent.')
    if len(encoded(value)) > CPU_MANIFEST_LIMIT:
        raise ProcessError('Captured CPU manifest exceeds its byte bound.')
    return decode(encoded(value))


def _cpu_open_root(root):
    """Walk every absolute component without following symlinks."""
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in Path(root).parts[1:]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = next_fd
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise ProcessError('Captured CPU root is not privately owned.', 'unavailable')
        return fd
    except BaseException:
        os.close(fd); raise


def _cpu_open_file(root_fd, relative):
    fd = os.dup(root_fd)
    try:
        parts = relative.split('/')
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = next_fd
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    finally:
        os.close(fd)


def _cpu_check_deadline(deadline, checkpoint=None):
    if checkpoint is not None: checkpoint()
    if time.monotonic() > deadline:
        raise ProcessError('CPU snapshot preparation deadline exceeded.', 'unavailable')


def _cpu_stream_file(root_fd, item, deadline, checkpoint=None):
    fd = _cpu_open_file(root_fd, item['path'])
    with os.fdopen(fd, 'rb') as handle:
        before = os.fstat(handle.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != item['mode'] or before.st_size != item['bytes']):
            raise ProcessError('Captured CPU file size, ownership or mode changed.', 'stale')
        hashed = hashlib.sha256(); count = 0; magic = None
        while True:
            _cpu_check_deadline(deadline, checkpoint)
            block = handle.read(1024 * 1024)
            if not block: break
            if magic is None: magic = block[:4]
            count += len(block)
            if count > item['bytes']: raise ProcessError('CPU file grew during verification.', 'stale')
            hashed.update(block)
        after = os.fstat(handle.fileno())
        identity_keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode', 'st_uid', 'st_nlink')
        if (any(getattr(before, key) != getattr(after, key) for key in identity_keys)
                or count != item['bytes'] or hashed.hexdigest() != item['sha256']
                or (magic == b'\x7fELF') != (item['kind'] == 'elf')):
            raise ProcessError('Captured CPU file identity changed.', 'stale')


def verify_cpu_snapshot(descriptor, *, checkpoint=None):
    """Stream exact immutable snapshot bytes; never import its packages here."""
    deadline = time.monotonic() + CPU_READY_SECONDS
    descriptor = validate_cpu_snapshot_descriptor(descriptor)
    try:
        root_fd = _cpu_open_root(descriptor['root'])
    except OSError as error:
        raise ProcessError('Captured CPU root is missing or unsafe.', 'unavailable') from error
    try:
        fd = _cpu_open_file(root_fd, 'manifest.json')
        with os.fdopen(fd, 'rb') as handle:
            info = os.fstat(handle.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_size != descriptor['manifest_bytes']
                    or info.st_uid != os.getuid() or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o400):
                raise ProcessError('Captured CPU manifest file changed.', 'stale')
            raw = handle.read(CPU_MANIFEST_LIMIT + 1)
        if len(raw) != descriptor['manifest_bytes'] or digest(raw) != descriptor['manifest_sha256']:
            raise ProcessError('Captured CPU manifest hash changed.', 'stale')
        manifest = validate_cpu_manifest(decode(raw))
        if raw != encoded(manifest): raise ProcessError('CPU manifest is not canonical.', 'invalid')
        for key in ('profile', 'python_version', 'packages', 'launch', 'total_bytes'):
            if manifest[key] != descriptor[key]: raise ProcessError('CPU descriptor differs from manifest.', 'stale')
        if len(manifest['files']) != descriptor['file_count']:
            raise ProcessError('CPU descriptor file count differs.', 'stale')
        expected = {item['path'] for item in manifest['files']}
        expected_dirs = {str(parent) for path in expected for parent in Path(path).parents if str(parent) != '.'}
        observed = set(); directory_count = 0
        pending = ['']
        while pending:
            _cpu_check_deadline(deadline, checkpoint)
            directory = pending.pop(); walk_fd = os.dup(root_fd)
            try:
                for part in directory.split('/') if directory else []:
                    next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=walk_fd)
                    os.close(walk_fd); walk_fd = next_fd
                with os.scandir(walk_fd) as entries:
                    for entry in entries:
                        _cpu_check_deadline(deadline, checkpoint)
                        relative = str(Path(directory) / entry.name)
                        info = entry.stat(follow_symlinks=False)
                        if stat.S_ISDIR(info.st_mode):
                            directory_count += 1
                            if (directory_count > CPU_FILE_COUNT or relative not in expected_dirs
                                    or info.st_uid != os.getuid() or info.st_mode & 0o022):
                                raise ProcessError('CPU snapshot contains unsafe or unlisted directories.', 'stale')
                            pending.append(relative)
                        elif stat.S_ISREG(info.st_mode):
                            if relative == 'manifest.json': continue
                            if relative not in expected:
                                raise ProcessError('CPU snapshot contains unlisted files.', 'stale')
                            observed.add(relative)
                        else:
                            raise ProcessError('CPU snapshot contains symlink or special file.', 'stale')
            finally:
                os.close(walk_fd)
        if observed != expected: raise ProcessError('CPU snapshot file inventory changed.', 'stale')
        for item in manifest['files']: _cpu_stream_file(root_fd, item, deadline, checkpoint)
        _cpu_check_deadline(deadline, checkpoint)
        return manifest
    except OSError as error:
        raise ProcessError('Captured CPU snapshot is missing, inaccessible or changed.', 'unavailable') from error
    finally:
        os.close(root_fd)


def validate_spec(value):
    keys = {'schema_version','attempt_id','request_sha256','argv','env','cwd','stdin_base64',
            'wall_seconds','executable_sha256','max_output_bytes','output_dir','captured_files','runtime_files','trusted_install_owner_uid'}
    if type(value) is not dict or type(value.get('schema_version')) is not int or value['schema_version'] not in (1,2):
        raise ProcessError('Unsupported custody spec.')
    if value['schema_version']==2:keys.add('model_evidence_context')
    if set(value)-{'native_loader_policy','cpu_snapshot','torch_runtime','resource_profile'}!=keys:
        raise ProcessError('Unsupported custody spec.')
    if not isinstance(value['attempt_id'], str) or not ID.fullmatch(value['attempt_id']):
        raise ProcessError('Invalid attempt identity.')
    for key in ('request_sha256','executable_sha256'):
        if type(value[key]) is not str or not HASH.fullmatch(value[key]):
            raise ProcessError('Invalid custody hash.')
    if 'native_loader_policy' in value and value['native_loader_policy'] != 'glibc_x86_64_no_preload_v1':
        raise ProcessError('Unsupported native loader policy.')
    argv, env = value['argv'], value['env']
    if type(argv) is not list or not 1 <= len(argv) <= 64 or not all(text(arg) for arg in argv) or not absolute(argv[0]):
        raise ProcessError('Invalid captured argument vector.')
    if type(env) is not dict or len(env) > 32 or any(not re.fullmatch('[A-Z_][A-Z0-9_]{0,63}', key) or type(val) is not str or len(val) > 4096 or '\0' in val for key,val in env.items()):
        raise ProcessError('Invalid explicit environment.')
    if not absolute(value['cwd']) or not absolute(value['output_dir']):
        raise ProcessError('Captured working/output directories must be absolute.')
    try:
        if type(value['stdin_base64']) is not str or len(value['stdin_base64']) > 87384:
            raise ValueError()
        if len(base64.b64decode(value['stdin_base64'], validate=True)) > 65536:
            raise ValueError()
    except ValueError as error:
        raise ProcessError('Stdin exceeds its 64 KiB bound or is invalid base64.') from error
    if type(value['trusted_install_owner_uid']) is not int or value['trusted_install_owner_uid'] == os.getuid() or value['trusted_install_owner_uid'] < 0:
        raise ProcessError('Runtime owner must be an explicitly distinct trusted installation identity.')
    runtime = value['runtime_files']
    if type(runtime) is not list or len(runtime)>10000:
        raise ProcessError('Invalid runtime inventory.')
    files = value['captured_files']
    if type(files) is not list or len(files) > 128:
        raise ProcessError('Invalid captured file inventory.')
    seen = set()
    for item in files+runtime:
        if type(item) is not dict or set(item) != {'path','sha256','bytes'} or not absolute(item['path']):
            raise ProcessError('Invalid captured file reference.')
        if item['path'] in seen or type(item['sha256']) is not str or not HASH.fullmatch(item['sha256']):
            raise ProcessError('Duplicate capture or invalid hash.')
        seen.add(item['path'])
        if type(item['bytes']) is not int or not 0 <= item['bytes'] <= CAPTURE_FILE_LIMIT:
            raise ProcessError('Invalid captured file size.')
    if 'cpu_snapshot' in value:
        descriptor = validate_cpu_snapshot_descriptor(value['cpu_snapshot'])
        if 'native_loader_policy' not in value:
            raise ProcessError('CPU snapshot requires the qualified loader policy.')
        if (len(argv) < 8 or argv[1:4] != ['-I', '-S', '-B'] or argv[5] != '--cpu-launch'
                or not absolute(argv[4]) or not absolute(argv[6]) or not absolute(argv[7])):
            raise ProcessError('Captured CPU launch vector differs from reserved bootstrap.')
        inventory = {item['path']: item for item in files}
        desc_file = inventory.get(argv[6]); bootstrap_file = inventory.get(argv[4])
        raw = encoded(descriptor)
        if (not desc_file or desc_file['bytes'] != len(raw) or desc_file['sha256'] != digest(raw)
                or not bootstrap_file or argv[7] not in inventory
                or Path(argv[4]).name != 'recipe_process.py' or Path(argv[6]).name != 'snapshot.json'):
            raise ProcessError('CPU bootstrap, descriptor and entry must be exact captured files.')
    if 'torch_runtime' in value:
        if 'cpu_snapshot' in value or 'resource_profile' in value:
            raise ProcessError('Managed Torch requires its own single runtime profile.')
        try:
            descriptor = _torch_owner().validate_descriptor(value['torch_runtime'])
        except ValueError as error:
            raise ProcessError('Invalid managed Torch descriptor.') from error
        if (len(argv) < 9 or argv[1:4] != ['-I', '-S', '-B'] or argv[5] != '--launch'
                or argv[7] not in ('cpu', 'cuda') or not all(absolute(argv[i]) for i in (4, 6, 8))
                or env != {}):
            raise ProcessError('Managed Torch requires its exact bootstrap and explicit device.')
        inventory = {item['path']: item for item in files}
        manifest_file = inventory.get(argv[6]); bootstrap_file = inventory.get(argv[4])
        root = Path(value['cwd']) / 'runtime-bootstrap'
        if (not manifest_file or manifest_file['bytes'] != descriptor['manifest_bytes']
                or manifest_file['sha256'] != descriptor['manifest_sha256'] or not bootstrap_file
                or Path(argv[4]) != root / 'torch_runtime.py' or Path(argv[6]) != root / 'torch-manifest.json'
                or argv[8] not in inventory or not Path(argv[8]).is_relative_to(Path(value['cwd']) / 'source')):
            raise ProcessError('Managed Torch manifest, bootstrap and entry must be exact captured files.')
    if value['schema_version']==2:
        context=value['model_evidence_context']
        if (type(context) is not dict or set(context)!={'path','sha256','bytes'} or
                context not in files or context.get('path')!=str(Path(value['cwd'])/MODEL_CONTEXT_PATH) or
                type(context.get('bytes')) is not int or not 1<=context['bytes']<=MODEL_CONTEXT_LIMIT or
                argv.count(context['path'])!=1 or not {'cpu_snapshot','torch_runtime'} & set(value)):
            raise ProcessError('Model context must be one exact bounded captured runtime argument.')
    capture_limit = 64 * 1024 * 1024
    if 'cpu_snapshot' in value or 'torch_runtime' in value:
        # The complete descriptor/bootstrap contract above qualifies this profile.
        capture_limit = 320 * 1024 * 1024
        input_root = Path(value['cwd']) / 'input'
        input_bytes = 0
        for item in files:
            relative = Path(item['path']).relative_to(Path(value['cwd'])) if Path(item['path']).is_relative_to(Path(value['cwd'])) else None
            if relative is None or len(relative.parts) < 2 or relative.parts[0] not in ('source', 'input', 'native-policy', 'runtime-bootstrap'):
                raise ProcessError('CPU capture must use its contained source/input/runtime layout.')
            if relative.parts[0] == 'input':
                if Path(item['path']).parent != input_root or item['bytes'] > 40 * 1024 * 1024:
                    raise ProcessError('CPU input must be a bounded named input file.')
                input_bytes += item['bytes']
        if input_bytes > 256 * 1024 * 1024:
            raise ProcessError('CPU input aggregate exceeds its byte bound.')
    if sum(item['bytes'] for item in files) > capture_limit or sum(item['bytes'] for item in runtime) > 512 * 1024 * 1024:
        raise ProcessError('Runtime/source closure exceeds its byte bound.')
    wall_limit, output_limit = 600, OUTPUT_LIMIT
    if 'resource_profile' in value:
        if value['resource_profile'] != LARGE_OUTPUT_PROFILE or 'cpu_snapshot' not in value:
            raise ProcessError('Large outputs require the explicit qualified CPU resource profile.')
        # CPU descriptor, launch vector and capture layout have all passed above.
        wall_limit, output_limit = 2400, LARGE_OUTPUT_LIMIT
    if type(value['wall_seconds']) is not int or not 1 <= value['wall_seconds'] <= wall_limit:
        raise ProcessError('Wall budget exceeds its admitted profile.')
    if type(value['max_output_bytes']) is not int or not 1 <= value['max_output_bytes'] <= output_limit:
        raise ProcessError('Invalid output byte budget.')
    if len(encoded(value)) > SPEC_LIMIT:
        raise ProcessError('Custody specification exceeds its bound.')
    return decode(encoded(value))


def read_regular(path, maximum):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ProcessError('Expected a bounded regular captured file.', 'unavailable')
        raw = handle.read(maximum + 1)
        after = os.fstat(handle.fileno())
        if len(raw) > maximum or (before.st_size,before.st_mtime_ns,before.st_ctime_ns) != (after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ProcessError('Captured file changed during observation.', 'stale')
        return raw


def verify_native_loader_policy(owner_uid):
    """The one admitted native policy permits no global loader preload entry."""
    descriptor = os.open('/etc', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if info.st_uid != owner_uid or info.st_mode & 0o022:
            raise ProcessError('Native loader configuration directory is not trusted.', 'unavailable')
        try:
            os.stat('ld.so.preload', dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise ProcessError('Native loader requires absent /etc/ld.so.preload.', 'unavailable')
    finally:
        os.close(descriptor)


def verify_model_context(spec):
    """The new generated input is owned/read-only at READY, GO and drainage."""
    if spec.get('schema_version')!=2:return
    root=directory=descriptor=None
    try:
        root=os.open(spec['cwd'],os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        root_before=os.fstat(root)
        directory=os.open('runtime-bootstrap',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=root)
        directory_before=os.fstat(directory)
        descriptor=os.open('model-evidence-context.json',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
        before=os.fstat(descriptor);context=spec['model_evidence_context']
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_uid!=os.getuid() or
                before.st_mode&0o222 or before.st_size!=context['bytes']):
            raise ProcessError('Generated model context is not the exact read-only owned file.','stale')
        chunks=[];remaining=MODEL_CONTEXT_LIMIT+1
        while remaining:
            chunk=os.read(descriptor,min(remaining,65536))
            if not chunk:break
            chunks.append(chunk);remaining-=len(chunk)
        raw=b''.join(chunks);after=os.fstat(descriptor)
        identity=lambda value:(value.st_dev,value.st_ino,value.st_uid,value.st_mode,value.st_nlink,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
        if (identity(before)!=identity(after) or identity(after)!=identity(os.stat('model-evidence-context.json',dir_fd=directory,follow_symlinks=False)) or
                identity(root_before)!=identity(os.fstat(root)) or identity(root_before)!=identity(os.stat(spec['cwd'],follow_symlinks=False)) or
                identity(directory_before)!=identity(os.fstat(directory)) or
                identity(directory_before)!=identity(os.stat('runtime-bootstrap',dir_fd=root,follow_symlinks=False)) or
                len(raw)!=context['bytes'] or digest(raw)!=context['sha256']):
            raise ProcessError('Generated model context path or file identity changed.','stale')
    except OSError:
        raise ProcessError('Generated model context path is unavailable.','stale') from None
    finally:
        for handle in (descriptor,directory,root):
            if handle is not None:os.close(handle)


def verify_captures(spec, *, checkpoint=None, include_cpu=True, include_torch=True,
                    on_torch_inventory=None):
    verify_model_context(spec)
    if 'cpu_snapshot' in spec and include_cpu:
        verify_cpu_snapshot(spec['cpu_snapshot'], checkpoint=checkpoint)
    if 'native_loader_policy' in spec:
        verify_native_loader_policy(spec['trusted_install_owner_uid'])
    for item in spec['captured_files']+spec['runtime_files']:
        if checkpoint is not None: checkpoint()
        raw = read_regular(item['path'], item['bytes'])
        if len(raw) != item['bytes'] or digest(raw) != item['sha256']:
            raise ProcessError('Captured input/source identity changed.', 'stale')
    if 'torch_runtime' in spec:
        owner = _torch_owner()
        try:
            manifest = owner.verify_descriptor(spec['torch_runtime'], owner.load(spec['argv'][6]))
            if include_torch:
                inventory_started = False
                def inventory_checkpoint():
                    nonlocal inventory_started
                    if not inventory_started:
                        inventory_started = True
                        if on_torch_inventory is not None: on_torch_inventory()
                    if checkpoint is not None: checkpoint()
                owner.verify(manifest, checkpoint=inventory_checkpoint)
        except ValueError as error:
            raise ProcessError('Managed Torch inventory verification failed.', 'stale') from error


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid, 'start_ticks':fields[19], 'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def valid_identity(value):
    return (type(value) is dict and set(value)=={'pid','start_ticks','boot_id'} and
            type(value['pid']) is int and value['pid']>0 and type(value['start_ticks']) is str and value['start_ticks'].isdigit() and
            type(value['boot_id']) is str and re.fullmatch('[a-f0-9-]{36}',value['boot_id']) is not None)


def sealed(name, raw):
    fd = os.memfd_create(name, os.MFD_ALLOW_SEALING)
    try:
        with os.fdopen(os.dup(fd), 'wb') as handle:
            handle.write(raw); handle.flush()
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, F_ADD_SEALS, IMMUTABLE_SEALS)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _trusted_python(owner_uid):
    path = Path('/usr/bin/python3').resolve(strict=True)
    if '/usr' not in str(path) or not str(path).startswith('/usr/'):
        raise ProcessError('Trusted system Python is unavailable.', 'unavailable')
    for item in (path,*path.parents):
        info = item.stat()
        if info.st_uid != owner_uid or info.st_mode & 0o022 or os.access(item,os.W_OK):
            raise ProcessError('System Python installation is writable.', 'unavailable')
    return str(path)


def _packet(channel, value):
    raw = encoded(value)
    if len(raw) > 4096:
        raise ProcessError('Custodian message exceeds its bound.')
    channel.send(raw)


def _receive(channel):
    raw, _ancillary, flags, _address = channel.recvmsg(4096)
    if not raw:
        return None
    if flags & socket.MSG_TRUNC:
        raise ProcessError('Truncated custodian message.')
    return decode(raw)


class Handle:
    def __init__(self, process, channel, spec, directory):
        self.process, self.channel, self.spec, self.directory = process, channel, spec, directory
        self.ready = None
        self.verifying = False
        self.sent_go = self.terminal = self.lost = self.sent_cancel = False
        self.started = time.monotonic()
        self.spec_sha256 = digest(encoded(spec))

    @property
    def expected_identity(self):
        return {'attempt_id':self.spec['attempt_id'],'request_sha256':self.spec['request_sha256'],
                'spec_sha256':self.spec_sha256,'helper':self.ready}

    def poll(self):
        events = []
        if self.channel is None:
            if not self.terminal and not self.lost:
                self.lost = True
                events.append({'type':'lost','reason':'custodian_channel_closed'})
            return events
        while True:
            try:
                event = _receive(self.channel)
            except BlockingIOError:
                break
            except (OSError, ProcessError):
                event = None
            if event is None:
                if not self.terminal and not self.lost:
                    self.lost = True
                    events.append({'type':'lost','reason':'custodian_channel_closed'})
                self.close_control(wait=False)
                break
            common = {'schema_version':1,'attempt_id':self.spec['attempt_id'],'request_sha256':self.spec['request_sha256'],'spec_sha256':self.spec_sha256}
            kind = event.get('type') if type(event) is dict else None
            fields = {'ready':{'helper'},'verifying':set(),'started':{'leader'},'terminal':{'receipt_sha256','drained','outcome'}}
            if kind not in fields or set(event) != set(common)|{'type'}|fields[kind] or any(event[key] != val for key,val in common.items()):
                self.cancel('protocol_error'); raise ProcessError('Invalid custodian event.')
            if kind == 'ready':
                if self.ready is not None or not valid_identity(event['helper']) or event['helper'] != identity(self.process.pid):
                    self.cancel('protocol_error'); raise ProcessError('Invalid custodian READY identity.')
                self.ready = event['helper']
            elif kind == 'verifying':
                if not self.sent_go or self.verifying or 'torch_runtime' not in self.spec:
                    self.cancel('protocol_error'); raise ProcessError('Invalid runtime verification observation.')
                self.verifying = True
            elif kind == 'started':
                if not self.sent_go or not valid_identity(event['leader']):
                    self.cancel('protocol_error'); raise ProcessError('Invalid recipe STARTED identity.')
            else:
                receipt = load_receipt(self.directory/'receipt.json', self.expected_identity)
                if digest(encoded(receipt)) != event['receipt_sha256'] or receipt['drained'] != event['drained'] or receipt['outcome'] != event['outcome']:
                    raise ProcessError('Terminal receipt differs from custodian event.')
                self.terminal = True
            events.append(event)
        if self.ready is None and not self.lost and time.monotonic()-self.started > preparation_seconds(self.spec):
            self.close_control(wait=False)
            raise ProcessError('Custodian readiness deadline exceeded.', 'unavailable')
        self.process.poll()
        return events

    def go(self, expected_ready_identity):
        if self.ready is None or expected_ready_identity != self.ready or self.sent_go or self.terminal or self.channel is None:
            raise ProcessError('GO requires the exact READY identity once.')
        _packet(self.channel, {'type':'go','attempt_id':self.spec['attempt_id'],'request_sha256':self.spec['request_sha256'],'helper':self.ready})
        self.sent_go = True

    def cancel(self, reason='cancelled'):
        if reason not in ('cancelled','shutdown','protocol_error'):
            raise ProcessError('Unsupported cancellation reason.')
        if self.channel is not None and not self.sent_cancel:
            self.sent_cancel = True
            try: _packet(self.channel, {'type':'cancel','reason':reason})
            except (OSError, ProcessError): self.close_control(wait=False)

    def close_control(self, *, wait=True):
        if self.channel is not None:
            self.channel.close(); self.channel = None
        if not wait:
            return None
        try:
            self.process.wait(timeout=CLEANUP_SECONDS + 2)
        except subprocess.TimeoutExpired as error:
            raise ProcessError('Custodian did not finish drainage within its cleanup budget.', 'unknown') from error
        if self.ready is None:
            return {'drained':False,'outcome':'unknown','reason':'no_ready_identity'}
        try:
            return load_receipt(self.directory/'receipt.json',self.expected_identity)
        except (OSError,ProcessError):
            return {'drained':False,'outcome':'unknown','reason':'missing_drain_receipt'}


def launch(spec, custody_dir):
    spec = validate_spec(spec)
    # Full managed inventory hashing belongs to the cancellable custodian after
    # admission/GO. Synchronous dispatch only checks the bounded captured files.
    verify_captures(spec, include_torch=False)
    executable = read_regular(spec['argv'][0], 128*1024*1024)
    if digest(executable) != spec['executable_sha256']:
        raise ProcessError('Executable identity changed.', 'stale')
    directory = Path(custody_dir)
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    parent = child = None
    descriptors = []
    try:
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        for name, raw in (('tuldok-custodian',SOURCE),('tuldok-spec',encoded(spec)),('tuldok-executable',executable)):
            descriptors.append(sealed(name,raw))
        directory_fd = os.open(directory, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        descriptors.append(directory_fd)
        if 'torch_runtime' in spec:
            # The helper runs from sealed source with -I; it cannot import an
            # application file through cwd or an ambient PYTHONPATH.
            raw = read_regular(spec['argv'][4], CAPTURE_FILE_LIMIT)
            capture = next(item for item in spec['captured_files'] if item['path'] == spec['argv'][4])
            if len(raw) != capture['bytes'] or digest(raw) != capture['sha256']:
                raise ProcessError('Managed Torch bootstrap changed before sealing.', 'stale')
            descriptors.append(sealed('tuldok-torch-owner',raw))
        command = [_trusted_python(spec['trusted_install_owner_uid']),'-I','-B',f'/proc/self/fd/{descriptors[0]}','--helper',str(child.fileno()),*(str(fd) for fd in descriptors[1:])]
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            pass_fds=(child.fileno(),*descriptors),close_fds=True,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'})
        parent.setblocking(False)
        return Handle(process,parent,spec,directory)
    except BaseException:
        if parent is not None: parent.close()
        raise
    finally:
        if child is not None: child.close()
        for fd in descriptors: os.close(fd)


def load_receipt(path, expected_identity):
    return validate_receipt(read_regular(path, 16384), expected_identity)


def validate_receipt(raw, expected_identity):
    """Validate retained exact receipt bytes without requiring a live filesystem."""
    if type(raw) is not bytes or not 1 <= len(raw) <= 16384:
        raise ProcessError('Custody receipt must contain 1–16384 bytes.')
    if type(expected_identity) is not dict or set(expected_identity) != {'attempt_id','request_sha256','spec_sha256','helper'} or not valid_identity(expected_identity['helper']):
        raise ProcessError('An exact READY identity is required to read a receipt.')
    value = decode(raw)
    fields = {'schema_version','attempt_id','request_sha256','spec_sha256','helper','go_received','spawn_closed',
              'drained','echild','outcome','reason','leader_returncode','streams'}
    if type(value) is not dict or set(value)!=fields or type(value['schema_version']) is not int or value['schema_version']!=1 or any(value.get(k)!=v for k,v in expected_identity.items()):
        raise ProcessError('Invalid retained custody receipt.')
    if not valid_identity(value['helper']) or any(type(value[k]) is not bool for k in ('go_received','spawn_closed','drained','echild')):
        raise ProcessError('Invalid receipt custody evidence.')
    if not value['spawn_closed'] or value['drained'] != value['echild'] or value['outcome'] not in ('completed','failed','cancelled','unknown'):
        raise ProcessError('Receipt cannot establish its claimed drainage.')
    if not value['drained'] and value['outcome']!='unknown':
        raise ProcessError('Unconfirmed drainage must remain unknown.')
    if value['leader_returncode'] is not None and type(value['leader_returncode']) is not int:
        raise ProcessError('Invalid leader exit status.')
    if value['outcome']=='completed' and (not value['go_received'] or value['leader_returncode']!=0):
        raise ProcessError('Completion requires successful execution.')
    if not text(value['reason'],100) or type(value['streams']) is not dict or set(value['streams'])!={'stdout','stderr'}:
        raise ProcessError('Invalid retained diagnostics.')
    for row in value['streams'].values():
        if type(row) is not dict or set(row)!={'bytes','sha256'} or type(row['bytes']) is not int or not 0<=row['bytes']<=STREAM_LIMIT or type(row['sha256']) is not str or not HASH.fullmatch(row['sha256']):
            raise ProcessError('Invalid stream receipt.')
    return value


def _builtin_artifact_source(source, expected_sha256, expected_bytes):
    """Versioned built-in publication proof; deliberately no external custody claim."""
    fields = {'schema_version', 'kind', 'catalog_id', 'run_id', 'request_sha256', 'material_sha256',
              'graph_sha256', 'release_sha256', 'adapter_sha256', 'artifact_name', 'artifact_role',
              'artifact_bytes', 'measured_sha256', 'artifact_manifest_sha256', 'artifact_manifest_base64', 'population'}
    if type(source) is not dict or set(source) != fields or source['schema_version'] != 2 or type(source['schema_version']) is not int:
        raise ProcessError('Built-in Run source fields differ from schema 2.')
    if source['kind'] != 'builtin_classifier_publication_v1' or source['artifact_role'] != 'model_evidence':
        raise ProcessError('Unsupported built-in publication source.')
    for key in ('catalog_id', 'run_id'):
        if type(source[key]) is not str or not ID.fullmatch(source[key]):
            raise ProcessError('Invalid built-in publication identity.')
    for key in ('request_sha256', 'material_sha256', 'graph_sha256', 'release_sha256', 'adapter_sha256',
                'measured_sha256', 'artifact_manifest_sha256'):
        if type(source[key]) is not str or not HASH.fullmatch(source[key]):
            raise ProcessError('Invalid built-in publication hash.')
    if (type(source['artifact_bytes']) is not int or source['artifact_bytes'] != expected_bytes
            or source['measured_sha256'] != expected_sha256 or source['population'] not in ('train', 'validation', 'test')):
        raise ProcessError('Built-in capture bytes or population differ.')
    name = source['artifact_name']
    if (type(name) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', name)
            or name in ('artifact-manifest.json', 'custody-receipt.json', 'release.zip')):
        raise ProcessError('Invalid built-in artifact name.')
    value = source['artifact_manifest_base64']
    if type(value) is not str or not 1 <= len(value) <= 87384:
        raise ProcessError('Built-in manifest encoding exceeds bound.')
    try:
        raw = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ProcessError('Invalid built-in manifest encoding.') from error
    if (len(raw) > 65536 or base64.b64encode(raw).decode('ascii') != value
            or digest(raw) != source['artifact_manifest_sha256']):
        raise ProcessError('Built-in publication manifest differs.')
    manifest = decode(raw)
    if (type(manifest) is not dict or set(manifest) != {'schema_version', 'adapter', 'source_identity', 'artifacts'}
            or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1
            or manifest['adapter'] != {'name': 'tinycnn_cpu', 'version': 2}):
        raise ProcessError('Unsupported built-in classifier publication.')
    identity = manifest['source_identity']
    if (type(identity) is not dict or set(identity) != {'dataset_id', 'run_id', 'graph_sha256', 'release_id', 'release_sha256'}
            or identity != {'dataset_id': source['catalog_id'], 'run_id': source['run_id'],
                'graph_sha256': source['graph_sha256'], 'release_id': source['release_sha256'], 'release_sha256': source['release_sha256']}):
        raise ProcessError('Built-in publication identity differs.')
    artifacts = manifest['artifacts']
    if type(artifacts) is not list or not 1 <= len(artifacts) <= 32:
        raise ProcessError('Built-in artifact inventory exceeds bound.')
    matches = [item for item in artifacts if type(item) is dict and item.get('name') == name]
    if len(matches) != 1:
        raise ProcessError('Built-in artifact not uniquely published.')
    item = matches[0]
    if (set(item) != {'name', 'sha256', 'bytes', 'media_type', 'split'} or item['sha256'] != expected_sha256
            or type(item['bytes']) is not int or item['bytes'] != expected_bytes
            or item['split'] not in (None, 'validation', 'test')
            or source['population'] != ('test' if item['split'] == 'test' else 'validation' if item['split'] == 'validation' else 'train')):
        raise ProcessError('Built-in artifact binding differs.')
    return dict(source)


def validate_artifact_source(source, expected_sha256, expected_bytes):
    """Check retained Run artifact custody declarations, not historical authority."""
    if (type(expected_sha256) is not str or not HASH.fullmatch(expected_sha256)
            or type(expected_bytes) is not int or expected_bytes < 0):
        raise ProcessError('Invalid expected artifact identity.')
    if type(source) is dict and source.get('schema_version') == 2:
        return _builtin_artifact_source(source, expected_sha256, expected_bytes)
    def exact(value, keys, label):
        if type(value) is not dict or set(value) != set(keys):
            raise ProcessError(label + ' fields differ from schema 1.')
    exact(source, ('schema_version', 'catalog_id', 'run_id', 'attempt_id',
                   'request_sha256', 'material_sha256', 'artifact_name',
                   'artifact_role', 'artifact_bytes', 'measured_sha256',
                   'custody_receipt_sha256', 'custody_identity',
                   'custody_receipt_base64'), 'Run source')
    if type(source['schema_version']) is not int or source['schema_version'] != 1:
        raise ProcessError('Unsupported Run source version.')
    for key in ('catalog_id', 'run_id', 'attempt_id'):
        if type(source[key]) is not str or not re.fullmatch('[a-f0-9]{32}', source[key]):
            raise ProcessError('Invalid Run source identity.')
    for key in ('request_sha256', 'material_sha256', 'measured_sha256', 'custody_receipt_sha256'):
        if type(source[key]) is not str or not HASH.fullmatch(source[key]):
            raise ProcessError('Invalid Run source hash.')
    if (type(source['artifact_name']) is not str
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', source['artifact_name'])
            or source['artifact_name'] in ('artifact-manifest.json', 'custody-receipt.json')
            or type(source['artifact_role']) is not str or not re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$').fullmatch(source['artifact_role'])):
        raise ProcessError('Invalid Run artifact declaration.')
    if type(source['artifact_bytes']) is not int or source['artifact_bytes'] != expected_bytes:
        raise ProcessError('Run artifact size differs from captured bytes.')
    identity = source['custody_identity']
    exact(identity, ('attempt_id', 'request_sha256', 'spec_sha256', 'helper'), 'Custody identity')
    if (source['run_id'] != source['attempt_id'] or source['attempt_id'] != identity['attempt_id']
            or source['request_sha256'] != identity['request_sha256']
            or type(identity['spec_sha256']) is not str or not HASH.fullmatch(identity['spec_sha256'])):
        raise ProcessError('Run source and custody identities differ.')
    encoded = source['custody_receipt_base64']
    if type(encoded) is not str or not 1 <= len(encoded) <= 21848:
        raise ProcessError('Invalid retained custody receipt encoding.')
    try:
        receipt_raw = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ProcessError('Invalid retained custody receipt encoding.') from error
    if (not 0 < len(receipt_raw) <= 16384
            or base64.b64encode(receipt_raw).decode('ascii') != encoded
            or hashlib.sha256(receipt_raw).hexdigest() != source['custody_receipt_sha256']):
        raise ProcessError('Retained custody receipt bytes differ.')
    try:
        receipt = validate_receipt(receipt_raw, identity)
    except ProcessError as error:
        raise ProcessError('Invalid retained custody evidence.') from error
    if receipt['outcome'] != 'completed' or not receipt['drained']:
        raise ProcessError('Run source requires completed and drained custody.')
    if source['measured_sha256'] != expected_sha256:
        raise ProcessError('Captured bytes differ from their source receipt.', 'conflict')
    return dict(source)


def _children():
    # /proc/.../children is not exposed on every supported host. PPID is the
    # admitted canonical observation; ECHILD, never this scan, proves drainage.
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[1]) == os.getpid():
                result.append(int(path.name))
        except FileNotFoundError:
            continue
        if len(result)>4096:
            raise ProcessError('Child inventory exceeds custody bound.', 'unknown')
    return result


def _signal_children(number):
    for pid in _children():
        try:
            fd = os.pidfd_open(pid)
            try: signal.pidfd_send_signal(fd,number)
            finally: os.close(fd)
        except ProcessLookupError:
            pass


def _outputs(spec):
    root=Path(spec['output_dir']); total=count=0
    large = spec.get('resource_profile') == LARGE_OUTPUT_PROFILE
    count_limit, file_limit = (96, LARGE_OUTPUT_FILE_LIMIT) if large else (128, OUTPUT_LIMIT)
    if root.is_symlink() or not root.is_dir():
        raise ProcessError('Output directory changed.')
    for path in root.iterdir():
        info=path.lstat();count+=1;total+=info.st_size
        if count>count_limit or not stat.S_ISREG(info.st_mode) or info.st_size>file_limit or total>spec['max_output_bytes']:
            raise ProcessError('Output inventory exceeded its admitted bound.')


def _helper(control_fd,spec_fd,exe_fd,directory_fd):
    channel=socket.socket(fileno=control_fd);channel.setblocking(False)
    spec=validate_spec(decode(os.pread(spec_fd,SPEC_LIMIT+1,0)))
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0): raise OSError(ctypes.get_errno(),'subreaper unavailable')
    who=identity(os.getpid());common={'schema_version':1,'attempt_id':spec['attempt_id'],'request_sha256':spec['request_sha256'],'spec_sha256':digest(encoded(spec))}
    logs={};hashes={name:hashlib.sha256() for name in ('stdout','stderr')};sizes={name:0 for name in hashes}
    for name in hashes: logs[name]=os.open(name+'.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600,dir_fd=directory_fd)
    selector=selectors.DefaultSelector();selector.register(channel,selectors.EVENT_READ,'control')
    leader=None;code=None;go=False;go_received=False;stopping=None;reason='not_started';drained=False
    started=time.monotonic();deadline=started+preparation_seconds(spec)
    verification_deadline=None
    def stop(why):
        nonlocal stopping,reason
        if stopping is None: stopping=time.monotonic();reason=why
    class PreparationStopped(Exception):
        pass
    def preparation_checkpoint():
        if verification_deadline is not None and time.monotonic() >= verification_deadline:
            stop('verification_timeout')
            raise PreparationStopped()
        try:
            message = _receive(channel)
        except BlockingIOError:
            return
        except (OSError, ProcessError):
            message = None
        if message is None:
            selector.unregister(channel)
            stop('controller_lost')
        elif (type(message) is dict and set(message) == {'type', 'reason'}
              and message['type'] == 'cancel' and message['reason'] in ('cancelled', 'shutdown', 'protocol_error')):
            stop(message['reason'])
        else:
            stop('protocol_error')
        raise PreparationStopped()
    try:
        # Parent launch already verified the complete CPU snapshot. Keep READY
        # responsive; recheck it cancellably at GO, before any recipe fork.
        verify_captures(spec, include_cpu=False, include_torch=False)
        _packet(channel,dict(common,type='ready',helper=who))
        while True:
            now=time.monotonic()
            if stopping is None and now>=deadline:stop('wall_timeout' if go else 'ready_timeout')
            if go and stopping is None:
                try:_outputs(spec)
                except (OSError,ProcessError):stop('output_limit')
            if stopping is not None:
                try:_signal_children(signal.SIGKILL if now-stopping>=GRACE_SECONDS else signal.SIGTERM)
                except (OSError,ProcessError):reason='child_observation_failed';break
            for _ in range(256):
                try:
                    pid,status=os.waitpid(-1,os.WNOHANG)
                except ChildProcessError:
                    if stopping is not None or go:drained=True
                    break
                if pid==0:break
                if pid==leader:code=os.waitstatus_to_exitcode(status)
            if go and code is not None and stopping is None:stop('exited' if code==0 else 'nonzero_exit')
            streams_open=any(key.data!='control' for key in selector.get_map().values())
            if drained and not streams_open:break
            if stopping is not None and now-stopping>=CLEANUP_SECONDS:
                if streams_open:drained=False;reason='stream_cessation_unknown'
                break
            for key,_mask in selector.select(.05):
                if key.data=='control':
                    try:message=_receive(channel)
                    except (OSError,ProcessError):message=None
                    if message is None:
                        selector.unregister(channel);stop('controller_lost');continue
                    if type(message) is dict and set(message)=={'type','reason'} and message['type']=='cancel' and message['reason'] in ('cancelled','shutdown','protocol_error'):
                        stop(message['reason']);continue
                    expected={'type':'go','attempt_id':spec['attempt_id'],'request_sha256':spec['request_sha256'],'helper':who}
                    if message!=expected or go_received or stopping is not None:stop('protocol_error');continue
                    # GO authorizes verification, then recipe creation. Receipt
                    # truth must retain that command even when hashing stops
                    # before a leader exists; `go` tracks successful preparation.
                    go_received=True
                    try:
                        verification_deadline=time.monotonic()+verification_seconds(spec)
                        verify_captures(spec, checkpoint=preparation_checkpoint if verification_seconds(spec) else None,
                            on_torch_inventory=lambda: _packet(channel,dict(common,type='verifying')))
                    except PreparationStopped:
                        continue
                    go=True;deadline=time.monotonic()+spec['wall_seconds']
                    input_fd=sealed('tuldok-stdin',base64.b64decode(spec['stdin_base64']))
                    pipes={name:os.pipe() for name in hashes}
                    leader=os.fork()
                    if leader==0:
                        try:
                            os.setsid();os.chdir(spec['cwd'])
                            os.dup2(input_fd,0);os.dup2(pipes['stdout'][1],1);os.dup2(pipes['stderr'][1],2)
                            for entry in os.listdir('/proc/self/fd'):
                                fd=int(entry)
                                if fd>2 and fd!=exe_fd:
                                    try:os.close(fd)
                                    except OSError:pass
                            resource.setrlimit(resource.RLIMIT_FSIZE,(spec['max_output_bytes'],spec['max_output_bytes']))
                            os.execve(exe_fd,spec['argv'],spec['env'])
                        except BaseException:os._exit(126)
                    os.close(input_fd)
                    for name,(read_fd,write_fd) in pipes.items():
                        os.close(write_fd);os.set_blocking(read_fd,False);selector.register(read_fd,selectors.EVENT_READ,name)
                    _packet(channel,dict(common,type='started',leader=identity(leader)))
                else:
                    data=os.read(key.fd,65536)
                    if not data:selector.unregister(key.fd);os.close(key.fd);continue
                    name=key.data;kept=data[:max(0,STREAM_LIMIT-sizes[name])]
                    pending=memoryview(kept)
                    while pending:
                        written=os.write(logs[name],pending)
                        if not written:raise OSError('Diagnostic write made no progress')
                        pending=pending[written:]
                    hashes[name].update(kept);sizes[name]+=len(kept)
                    if len(kept)!=len(data):stop('stream_limit')
    except BaseException:
        stop('custodian_error')
        deadline=time.monotonic()+CLEANUP_SECONDS
        while time.monotonic()<deadline:
            try:
                _signal_children(signal.SIGKILL)
                for _ in range(256):
                    if not os.waitpid(-1,os.WNOHANG)[0]:break
            except ChildProcessError:drained=True;break
            except (OSError,ProcessError):break
            time.sleep(.02)
    finally:
        for key in list(selector.get_map().values()):
            if key.data!='control':
                try:os.close(key.fd)
                except OSError:pass
        selector.close()
    if drained and go and spec['schema_version']==2:
        try:verify_model_context(spec)
        except (OSError,ProcessError):reason='model_context_changed'
    outcome='unknown' if not drained else 'completed' if go and code==0 and reason=='exited' else 'cancelled' if reason in ('cancelled','shutdown','controller_lost') else 'failed'
    receipt=dict(common,helper=who,go_received=go_received,spawn_closed=True,drained=drained,echild=drained,outcome=outcome,reason=reason,
        leader_returncode=code,streams={name:{'bytes':sizes[name],'sha256':hashes[name].hexdigest()} for name in hashes})
    for fd in logs.values():os.fsync(fd);os.close(fd)
    fd=os.open('receipt.tmp',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600,dir_fd=directory_fd)
    with os.fdopen(fd,'wb') as handle:handle.write(encoded(receipt));handle.flush();os.fsync(handle.fileno())
    os.rename('receipt.tmp','receipt.json',src_dir_fd=directory_fd,dst_dir_fd=directory_fd);os.fsync(directory_fd)
    try:_packet(channel,dict(common,type='terminal',receipt_sha256=digest(encoded(receipt)),drained=drained,outcome=outcome))
    except OSError:pass
    channel.close()


def _cpu_descriptor_file(path):
    if not absolute(path): raise ProcessError('CPU descriptor path must be absolute.')
    raw = read_regular(path, CPU_MANIFEST_LIMIT)
    value = validate_cpu_snapshot_descriptor(decode(raw))
    if raw != encoded(value): raise ProcessError('CPU descriptor must be canonical.')
    return value


def _cpu_origins(manifest, descriptor, bootstrap):
    import sys
    root = Path(descriptor['root'])
    permitted = {str(root / item['path']) for item in manifest['files']}
    virtual_modules = {}
    virtual = {'torch.ops': ('torch._ops', '_Ops', '_ops.py'),
               'torch.classes': ('torch._classes', '_Classes', '_classes.py')}
    for name, module in list(sys.modules.items()):
        location = getattr(module, '__file__', None)
        if name in virtual:
            defining_name, class_name, pseudo_file = virtual[name]
            defining = sys.modules.get(defining_name)
            if (defining is None or type(module) is not vars(defining).get(class_name)
                    or location != pseudo_file or type(module).__module__ != defining_name):
                raise ProcessError('Captured virtual module identity differs: ' + name, 'unavailable')
            location = vars(defining).get('__file__')
            if location not in permitted:
                raise ProcessError('Captured virtual module definition differs: ' + name, 'unavailable')
            virtual_modules[name] = {'pseudo_file': pseudo_file, 'defining_module': defining_name,
                                     'defining_path': location, 'class': class_name}
        if location is None: continue
        path = str(Path(location).absolute())
        if path != bootstrap and path not in permitted:
            raise ProcessError('Python module originates outside captured runtime: ' + name, 'unavailable')
    expected = [str(root / manifest['launch'][key]) for key in ('stdlib', 'dynload', 'packages')]
    if sys.path != expected:
        raise ProcessError('Captured Python import path changed.', 'unavailable')
    return virtual_modules


def _cpu_mappings(manifest, descriptor):
    root = Path(descriptor['root'])
    permitted = {str(root / item['path']) for item in manifest['files'] if item['kind'] == 'elf'}
    observed = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) < 6 or columns[5].startswith('['): continue
        path = columns[5]
        if path not in permitted:
            raise ProcessError('Native mapping originates outside captured runtime: ' + path[:512], 'unavailable')
        observed.add(path)
    for key in ('python', 'loader'):
        if str(root / manifest['launch'][key]) not in observed:
            raise ProcessError('Captured Python/loader mapping is absent.', 'unavailable')
    return sorted(observed)


def _cpu_launch_runtime(descriptor_path, entry, arguments):
    descriptor = _cpu_descriptor_file(descriptor_path)
    manifest = verify_cpu_snapshot(descriptor)
    verify_native_loader_policy(os.stat('/usr').st_uid)
    root = Path(descriptor['root']); launch = manifest['launch']
    bootstrap = str(Path(__file__).absolute())
    if not absolute(entry) or not absolute(bootstrap):
        raise ProcessError('Captured CPU entry and bootstrap paths must be absolute.')
    # Fresh profile-owned environment: C locale avoids host locale/gconv mmaps;
    # Python UTF-8 mode retains deterministic Unicode filesystem/text behavior.
    # No inherited loader, Python, site-hook or package-path overrides.
    env = {'LANG': 'C', 'LC_ALL': 'C', 'PYTHONCOERCECLOCALE': '0', 'PYTHONUTF8': '1'}
    env.update(PYTHONHOME=str(root / launch['home']), PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    command = [str(root / launch['loader']), '--inhibit-cache', '--library-path',
               ':'.join(str(root / item) for item in launch['library_dirs']),
               '--glibc-hwcaps-mask', '', str(root / launch['python']), '-P', '-S', '-B',
               bootstrap, '--cpu-child', descriptor_path, entry, *arguments]
    os.execve(command[0], command, env)


def _cpu_child(descriptor_path, entry, arguments):
    import sys
    import runpy
    descriptor = _cpu_descriptor_file(descriptor_path)
    manifest = verify_cpu_snapshot(descriptor)
    root = Path(descriptor['root']); launch = manifest['launch']
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix) != root / launch['home']:
        raise ProcessError('Captured Python bootstrap identity differs.', 'unavailable')
    # Check already imported bootstrap modules before permitting package imports.
    # -P -S excludes cwd, script directory, site hooks and user-site processing.
    expected = [str(root / launch[key]) for key in ('stdlib', 'dynload', 'packages')]
    initial = [str(root / launch['home'] / 'lib' / 'python312.zip'), *expected[:2]]
    if sys.path != initial:
        raise ProcessError('Captured Python initial path differs.', 'unavailable')
    sys.path[:] = expected
    bootstrap = str(Path(__file__).absolute())
    _cpu_origins(manifest, descriptor, bootstrap)
    _cpu_mappings(manifest, descriptor)
    import numpy
    import torch
    if str(numpy.__version__) != manifest['packages']['numpy'] or str(torch.__version__) != manifest['packages']['torch']:
        raise ProcessError('Captured CPU package versions differ.', 'unavailable')
    if torch.version.cuda is not None:
        raise ProcessError('Captured profile must be CPU-only.', 'unsupported')
    virtual_modules = _cpu_origins(manifest, descriptor, bootstrap)
    mappings = _cpu_mappings(manifest, descriptor)
    evidence = {'tuldok_cpu_runtime': {'snapshot_sha256': descriptor['manifest_sha256'],
        'python': sys.version, 'sys_path': list(sys.path), 'virtual_modules': virtual_modules,
        'environment': {key: os.environ.get(key) for key in ('LANG', 'LC_ALL', 'PYTHONCOERCECLOCALE',
            'PYTHONUTF8', 'PYTHONHOME', 'PYTHONNOUSERSITE', 'PYTHONDONTWRITEBYTECODE', 'CUDA_VISIBLE_DEVICES')},
        'packages': {'numpy': str(numpy.__version__), 'torch': str(torch.__version__)},
        'module_paths': {'numpy': numpy.__file__, 'torch': torch.__file__,
                         'os': os.__file__, 'ctypes': ctypes.__file__}, 'mappings': mappings}}
    line = encoded(evidence)
    if len(line) > STREAM_LIMIT // 2:
        raise ProcessError('Captured CPU initialization evidence exceeds its bound.', 'unsupported')
    print(line.decode(), flush=True)
    if not absolute(entry): raise ProcessError('Captured recipe entry must be absolute.')
    sys.argv[:] = [entry, *arguments]
    # Recipe source lookup starts only after the captured runtime qualification.
    sys.path.insert(0, str(Path(entry).parent))
    runpy.run_path(entry, run_name='__main__')


if __name__=='__main__':
    import sys
    if len(sys.argv) in (6,7) and sys.argv[1]=='--helper':
        if len(sys.argv)==7:
            import types
            module=types.ModuleType('torch_runtime')
            module.__file__=f'/proc/self/fd/{sys.argv[6]}'
            exec(compile(os.pread(int(sys.argv[6]),CAPTURE_FILE_LIMIT+1,0),module.__file__,'exec'),module.__dict__)
            sys.modules['torch_runtime']=module
            os.close(int(sys.argv[6]))
        _helper(*(int(value) for value in sys.argv[2:6]))
    elif len(sys.argv)>=4 and sys.argv[1] in ('--cpu-launch', '--cpu-child'):
        (_cpu_launch_runtime if sys.argv[1]=='--cpu-launch' else _cpu_child)(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        raise SystemExit(2)
