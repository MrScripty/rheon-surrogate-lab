"""Lab wire adapter for the generic glibc native-source policy.

Runs owns descendants and deadlines. This adapter verifies native identity before
the original lab reader receives READY, and retains actual launch diagnostics.
"""
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(raw):
    fd = os.memfd_create("pressure-project", os.MFD_ALLOW_SEALING)
    try:
        with os.fdopen(os.dup(fd), "wb") as stream:
            stream.write(raw)
        fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", 1033), 15)
        return fd
    except BaseException:
        os.close(fd)
        raise


class ReadyStream:
    def __init__(self, prefix, stream):
        self.prefix, self.stream = io.BytesIO(prefix), stream

    def read(self, size):
        if size < 0:
            raise ValueError("Native reads must be bounded")
        head = self.prefix.read(size)
        return head + self.stream.read(size - len(head))

    def close(self):
        self.prefix.close()
        self.stream.close()


class NativeLaunches:
    PIPE = subprocess.PIPE

    def __init__(self, binary, policy):
        self.binary = Path(binary)
        self.policy = json.loads(Path(policy).read_text())
        self.raw = self.binary.read_bytes()
        self.evidence = []
        policy = self.policy
        directory = "/usr/lib/x86_64-linux-gnu"
        arguments = ["--inhibit-cache", "--library-path", directory,
                     "--glibc-hwcaps-mask", ""]
        if (policy.get("schema_version") != 1
                or policy.get("policy") != "glibc_x86_64_no_preload_v1"
                or policy.get("source", {}).get("sha256") != sha(binary)
                or policy.get("library_directory") != directory
                or policy.get("loader_arguments") != arguments
                or policy.get("absent_paths") != ["/etc/ld.so.preload"]
                or policy.get("loader", {}).get("path") != directory + "/ld-linux-x86-64.so.2"):
            raise ValueError("Native policy differs from the supported loader contract")
        self.identities = {}
        files = policy.get("runtime_files", [])
        if not 1 <= len(files) <= 32 or policy["loader"] not in files:
            raise ValueError("Missing native runtime closure")
        for item in files:
            path = Path(item["path"])
            if (str(path.parent) != directory or path.is_symlink()
                    or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]):
                raise ValueError("Native runtime closure changed")
            info = path.stat()
            self.identities[(info.st_dev, info.st_ino)] = str(path)

    def Popen(self, arguments, **kwargs):
        seeds = {str(n) for n in (*range(100, 112), 500, 501, *range(900, 904))}
        if (len(arguments) != 4 or arguments[0] != str(self.binary)
                or arguments[1] != "32" or arguments[2] not in ("jacobi", "sgs", "learned")
                or arguments[3] not in seeds or len(self.evidence) >= 80
                or kwargs != {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE}):
            raise ValueError("Native request differs from fixed pressure experiment")
        if os.path.lexists("/etc/ld.so.preload"):
            raise ValueError("System loader preload is unsupported")
        for item in self.policy["runtime_files"]:
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Native runtime changed before launch")
        loader = Path(self.policy["loader"]["path"]).read_bytes()
        native_fd = sealed(self.raw)
        loader_fd = None
        process = None
        diagnostic = tempfile.TemporaryFile()
        try:
            loader_fd = sealed(loader)
            process = subprocess.Popen(
                [f"/proc/self/fd/{loader_fd}", *self.policy["loader_arguments"],
                 f"/proc/self/fd/{native_fd}", *arguments[1:]],
                pass_fds=(native_fd, loader_fd), stderr=diagnostic, **kwargs)
            prefix = bytearray()
            def read(count):
                value = bytearray()
                while len(value) < count:
                    part = process.stdout.read(count - len(value))
                    if not part:
                        raise ValueError("Native closed before READY")
                    value.extend(part)
                prefix.extend(value)
                return bytes(value)
            size = int.from_bytes(read(4), "little")
            if not 0 < size <= 65536:
                raise ValueError("Invalid native READY size")
            ready = json.loads(read(size))
            expected = {"schema": "rheon-trajectory-v1", "n": 32,
                        "method": arguments[2], "seed": int(arguments[3]),
                        "dt": 1 / 60, "density": 1.25,
                        "spacing": [1 / 32] * 3, "face_lengths": [33 * 32 * 32] * 3}
            if any(ready.get(k) != v for k, v in expected.items()):
                raise ValueError("Native READY differs from declared physics")
            expected_maps = dict(self.identities)
            for label, fd in (("sealed_native", native_fd), ("sealed_loader", loader_fd)):
                info = os.fstat(fd)
                expected_maps[(info.st_dev, info.st_ino)] = label
            mappings = set()
            for line in Path(f"/proc/{process.pid}/maps").read_text().splitlines():
                fields = line.split()
                inode = int(fields[4])
                if inode:
                    major, minor = (int(v, 16) for v in fields[3].split(":"))
                    identity = (os.makedev(major, minor), inode)
                    if identity not in expected_maps:
                        raise ValueError("Native loaded an undeclared mapping")
                    mappings.add(expected_maps[identity])
            required = set(self.identities.values()) - {self.policy["loader"]["path"]}
            if not required | {"sealed_native", "sealed_loader"} <= mappings:
                raise ValueError("Native mappings are incomplete")
            record = {"ready": ready, "mapped_files": sorted(mappings)}
            self.evidence.append(record)
            original_wait = process.wait
            captured = False
            def wait(*args, **kw):
                nonlocal captured
                code = original_wait(*args, **kw)
                if not captured:
                    captured = True
                    diagnostic.seek(0)
                    raw = diagnostic.read(65537)
                    diagnostic.close()
                    if len(raw) > 65536:
                        raise ValueError("Native diagnostic exceeds bound")
                    record.update(exit_code=code, stderr=raw.decode("utf-8", errors="replace"),
                                  stderr_sha256=hashlib.sha256(raw).hexdigest())
                return code
            process.wait = wait
            process.stdout = ReadyStream(bytes(prefix), process.stdout)
            return process
        except BaseException:
            if process is not None:
                process.kill()
                process.wait(timeout=10)
                process.stdin.close()
                process.stdout.close()
            diagnostic.close()
            raise
        finally:
            os.close(native_fd)
            if loader_fd is not None:
                os.close(loader_fd)
