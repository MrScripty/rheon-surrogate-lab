"""Extract captured scientific code/assets and prohibit ambient field fallback."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import zipfile


class PortableSharedRuntime:
    def __init__(self, archive, manifest, forbidden_root, allowed_files, output):
        self.archive, self.manifest = Path(archive), manifest
        self.forbidden = Path(forbidden_root)
        self.allowed = {Path(p).absolute() for p in allowed_files}
        self.output = Path(output).absolute()
        self.active = False
        self.denied = []

    def __enter__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rheon-managed-shared-')
        self.root = Path(self.temporary.name)
        self.cwd, self.paths = Path.cwd(), list(sys.path)
        with zipfile.ZipFile(self.archive) as source:
            names = source.namelist()
            if len(set(names)) != len(names) or set(names) != set(self.manifest['members']):
                raise ValueError('Exact captured source member inventory required')
            if sum(s.file_size for s in source.infolist()) > 128 * 1024**2:
                raise ValueError('Captured scientific source exceeds finite extraction cap')
            for name, expected in self.manifest['members'].items():
                target = self.root / name
                if not target.resolve().is_relative_to(self.root):
                    raise ValueError('Captured source path traversal')
                raw = source.read(name)
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError('Captured source/asset bytes changed')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
        runtime = self.forbidden / 'runs/dependency-migration-v1/app-env'
        sys.path = [p for p in sys.path if p and (not Path(p).absolute().is_relative_to(self.forbidden)
                    or Path(p).absolute().is_relative_to(runtime))]
        sys.path[:0] = [str(self.root / 'experiments' / n) for n in
            ('full-ai-data-diversity-v1', 'full-ai-tight-targets-v1', 'full-ai-next-priorities-v1', 'pressure-methods-v2')]
        os.chdir(self.root)
        self.active = True

        def restrict(event, values):
            if not self.active or event != 'open' or not isinstance(values[0], (str, bytes)):
                return
            path = Path(os.fsdecode(values[0])).absolute()
            denied = ((path.is_relative_to(self.forbidden) and not path.is_relative_to(runtime)
                       and not path.is_relative_to(self.output) and path not in self.allowed)
                      or path.is_relative_to('/tmp/rheon-surrogate-lab-long-reference-240-v1'))
            if denied:
                self.denied.append(str(path))
                raise PermissionError('Captured trainer cannot read ambient scientific files')

        sys.addaudithook(restrict)
        return self

    def __exit__(self, *unused):
        self.active = False
        os.chdir(self.cwd)
        sys.path[:] = self.paths
        self.temporary.cleanup()
