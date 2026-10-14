"""Offline admission of the single retained GroupNorm final200 candidate.

No model, trainer or native executable is imported or invoked. The only executed
legacy source is the byte-pinned standalone data reader.
"""
import ast
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
import types
import zipfile

import numpy as np

LAB = Path(__file__).resolve().parents[3]
MIB = 1024**2
INPUT_LIMIT = 256 * MIB
SOURCE_COMMIT = "fd3a6009e5805978a29ed1df7389ad226a40a914"
TRAIN_RUN = "d73bc2976c4f442f8aefc8f7242bd252"
RESTORE_RUN = "265c9574c06048d8bc91daa1cf8eab02"
ORIGIN_SHA = "dee7811a95b85a4c9fd2a228163ab39288d85bdeee804d09804c24eb7c92663d"
RESIDUAL_ANCHOR = "26c33fac5df17b66fadba036368f2008ec172a860231ff75822cd2ef9dba11b8"
READER_SHA = "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
PREPARE_SHA = "046d99e8d962aea2c6a99aafd41bf6c410dc6461a18655658d21f2b9f2ea80df"
NATIVE_SHA = "3a06964ed4121cde5394e114304abd3f50a569ed3345682d5dccd20c740ba3df"
BUILD_SHA = "70b0e33f5ec13c289e5afd0c38df783655936e06c2dfd32f0509e94d2aef8307"
REPORT_SHAS = {
    "training": "90f2e29b4e0b83dcc3f0ac94fa6583b032c61badb9f1391651f89e1b2af3efbb",
    "restore": "6ba471e68802de80612f75c1065011f4ca5436948cbc31b3b7039d38e67a4e5e",
}
PACKET_SHAS = {
    "train-0.npz": "6b3d0c4a57df0198b4fe13f7e887b8fa238fbf3fd34877f9dc598c157d2b55dc",
    "train-1.npz": "d6d7980a51c17b79a6567c870aca9001108779798cee490101429e00814a9f02",
    "validation.npz": "17a9faa84723208d1bd9d6a3a8b1f5921806e93b85d8983d2041404fb295855d",
    "splits.json": "fbb9edddfa08877e0834fcdf6e0d10e8f4769c5883fc6597b829325c2784650a",
    "origin.json": ORIGIN_SHA,
}
SOURCE_WITNESS_SHA = "549ad91612e7e2607e902bec3ccbf66366f6df1910c5b1755383d691b1f025f6"
RECEIPT_KEYS = set("architecture checkpoints closure diagnostics_sha256 elapsed_s initialization_mapping_sha256 mode optimizer_updates originating_training_run outputs parameter_count per_case_inference_s predictions_inventory predictions_sha256 runtime_observations sample_order_sha256 schema status trained_optimizer_updates training_history validation weights_inventory weights_sha256 witness_cases".split())


class AdmissionError(ValueError):
    """A local authoritative input or native response failed its contract."""


def require(condition, message):
    if not condition:
        raise AdmissionError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def keys(value, expected, name):
    require(type(value) is dict and set(value) == set(expected), name + ": closed fields")


def number(value, name, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value), name + ": finite number")
    require(not nonnegative or value >= 0, name + ": nonnegative")
    return value


def integer(value, name, expected=None):
    require(type(value) is int and value >= 0 and (expected is None or value == expected),
            name + ": exact integer")


def hash_string(value):
    require(type(value) is str and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), "SHA256 type")


def strict_json(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    def floating(text):
        value = float(text)
        number(value, "JSON")
        return value
    def invalid(text):
        raise AdmissionError("nonfinite JSON: " + text)
    try:
        return json.loads(blob, object_pairs_hook=pairs, parse_float=floating,
                          parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise AdmissionError("invalid bounded JSON: " + str(error)[:200]) from error


def check_deadline(deadline):
    require(type(deadline) in (int, float) and math.isfinite(deadline) and
            time.monotonic() < deadline, "admission deadline")


def regular(path):
    path = Path(path).absolute()
    require(path.resolve() == path, "input symlink/traversal: " + str(path))
    require(stat.S_ISREG(path.lstat().st_mode), "regular input required: " + str(path))
    return path


class Inputs:
    """Actual cumulative byte reads, including repeated postverification reads."""
    def __init__(self, deadline):
        self.deadline, self.bytes_read, self.files = deadline, 0, {}

    def charge(self, count):
        check_deadline(self.deadline)
        require(type(count) is int and 0 <= count <= INPUT_LIMIT - self.bytes_read,
                "cumulative input read limit256MiB")
        self.bytes_read += count

    def read(self, path, limit=8*MIB, expected=None):
        check_deadline(self.deadline)
        path = regular(path)
        remaining=INPUT_LIMIT-self.bytes_read
        require(0 < path.stat().st_size <= min(limit,remaining), "bounded file size/read budget: " + str(path))
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            require(stat.S_ISREG(os.fstat(fd).st_mode), "opened regular file required")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                blob = stream.read(min(limit+1,remaining))
            self.charge(len(blob))
            require(os.fstat(fd).st_size==len(blob),"file changed size during bounded read")
        finally:
            os.close(fd)
        require(0 < len(blob) <= limit, "bounded input read")
        sha = digest(blob)
        require(expected is None or sha == expected, "input hash: " + str(path))
        entry = {"sha256": sha, "bytes": len(blob)}
        require(str(path) not in self.files or self.files[str(path)] == entry, "input drift")
        self.files[str(path)] = entry
        return blob

    def external_snapshot(self, path, sha, count):
        # The pinned reader returns hashes of the immutable bytes it decoded.
        path = regular(path)
        require(path.stat().st_size == count, "reader file-size drift")
        self.charge(count)
        self.files[str(path)] = {"sha256": sha, "bytes": count}


def import_pinned(path, expected, inputs, name):
    blob = inputs.read(path, MIB, expected)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module  # dataclass resolves its defining module
    try:
        exec(compile(blob, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def population():
    tids = [f"smoke-train-{s}" for s in range(100000,100012)]
    tids += [f"smoke-validation-{s}" for s in range(100100,100104)]
    return [{"case_id": f"{tid}:frame-{f:03d}", "trajectory_id": tid,
             "frame": f, "row_index": t*24+f,
             "split": "train" if t < 12 else "validation"}
            for t, tid in enumerate(tids) for f in (0,23)]


def descriptors():
    result = {}
    blocks = [(f"encoders.{i}", before, after) for i,(before,after) in
              enumerate(zip((10,8,16,24),(8,16,24,32)))]
    blocks += [("bottleneck",32,48)]
    blocks += [(f"decoders.{i}", before, after) for i,(before,after) in
               enumerate(zip((80,56,40,24),(32,24,16,8)))]
    for name, before, after in blocks:
        for layer, incoming in ((0,before),(3,after)):
            result[f"{name}.layers.{layer}.weight"] = (after,incoming,3,3,3)
            result[f"{name}.layers.{layer}.bias"] = (after,)
    result.update({"head.weight": (1,8,1,1,1), "head.bias": (1,)})
    return result


def validate_inventory(inventory, shapes, dtype):
    keys(inventory, shapes, "numeric inventory")
    for name, shape in shapes.items():
        entry = inventory[name]
        keys(entry, {"shape","dtype","bytes","sha256"}, name)
        require(type(entry["shape"]) is list and all(type(n) is int for n in entry["shape"])
                and tuple(entry["shape"]) == shape, "typed numeric shape")
        require(entry["dtype"] == (dtype[name] if type(dtype) is dict else dtype), "numeric dtype")
        integer(entry["bytes"], "numeric bytes", math.prod(shape)*np.dtype(entry["dtype"]).itemsize)
        hash_string(entry["sha256"])


def decode_npz(blob, inventory):
    """Header/length proof before constructing any ndarray from immutable bytes."""
    require(0 < len(blob) <= 8*MIB, "numeric archive bound")
    result = {}
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            entries = archive.infolist()
            require(len(entries) == len(inventory) and
                    {e.filename for e in entries} == {n+".npy" for n in inventory}, "closed unique ZIP members")
            require(sum(e.file_size for e in entries) <= 8*MIB, "expanded NPZ bound")
            for entry in entries:
                require(entry.compress_type == zipfile.ZIP_STORED and not entry.flag_bits&1,
                        "uncompressed unencrypted NPZ")
                declaration = inventory[entry.filename[:-4]]
                with archive.open(entry) as stream:
                    require(stream.read(8) == b"\x93NUMPY\x01\x00", "NPY1 header")
                    length = stream.read(2)
                    require(len(length)==2, "NPY header length")
                    size = int.from_bytes(length,"little")
                    require(0 < size <= 1024, "NPY header bound")
                    raw = stream.read(size)
                    require(len(raw)==size, "NPY header truncation")
                    tree = ast.parse(raw.decode("latin1").strip(), mode="eval")
                    require(type(tree.body) is ast.Dict and len(tree.body.keys)==3 and
                            {ast.literal_eval(k) for k in tree.body.keys} == {"descr","shape","fortran_order"},
                            "unique NPY header fields")
                    header = ast.literal_eval(tree)
                    require(type(header["shape"]) is tuple and all(type(n) is int for n in header["shape"])
                            and header["shape"] == tuple(declaration["shape"])
                            and header["descr"] == declaration["dtype"] and header["fortran_order"] is False,
                            "NPY exact dtype/shape/order")
                    require(entry.file_size == 10+size+declaration["bytes"], "NPY payload size")
                    body = stream.read(declaration["bytes"]+1)
                    require(len(body)==declaration["bytes"] and digest(body)==declaration["sha256"],
                            "NPY raw hash/size")
                value = np.frombuffer(body,dtype=declaration["dtype"]).reshape(header["shape"])
                require(np.isfinite(value).all(), "finite cached numeric artifact")
                result[entry.filename[:-4]] = value
    except (zipfile.BadZipFile, SyntaxError, TypeError, UnicodeError, OverflowError) as error:
        raise AdmissionError("invalid numeric archive") from error
    return result


def validate_receipt(value, mode):
    keys(value, RECEIPT_KEYS, "GN receipt")
    require(value["schema"] == "rheon-global-pressure-groupnorm-training-receipt-v1"
            and value["status"] == "complete" and value["mode"] == mode, "GN complete receipt identity")
    integer(value["optimizer_updates"], "updates", 200 if mode=="train" else 0)
    integer(value["trained_optimizer_updates"], "trained updates",200)
    integer(value["parameter_count"], "parameters",372377)
    architecture = value["architecture"]
    keys(architecture, {"activation","block_normalization","bottleneck_width","convolutions_per_block",
                       "counts_supported","decoder_widths","dtype","encoder_widths","gauge","global_path",
                       "head","input_channels","kernel","padding","pool","pressure_dtype","schema","upsampling"}, "architecture")
    require(architecture["schema"] == "rheon-global-pressure-groupnorm-model-v1", "GN model identity")
    norm = architecture["block_normalization"]
    keys(norm, {"affine","epsilon","groups","name","position"}, "GN normalization")
    integer(norm["groups"], "GN groups",4)
    require(norm["affine"] is False and type(norm["epsilon"]) is float and norm["epsilon"]==1e-5
            and norm["name"]=="GroupNorm" and norm["position"]=="each block convolution before SiLU", "fixed nonaffine GN")
    for field, expected in {"bottleneck_width":48,"convolutions_per_block":2,"input_channels":10}.items():
        integer(architecture[field], field,expected)
    for field, expected in {"counts_supported":[16,32],"decoder_widths":[32,24,16,8],
                            "encoder_widths":[8,16,24,32],"kernel":[3,3,3]}.items():
        require(type(architecture[field]) is list and all(type(n) is int for n in architecture[field])
                and architecture[field]==expected, "fixed architecture list")
    for field, expected in {"activation":"silu","dtype":"float32","pressure_dtype":"float64",
                            "gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast",
                            "head":"linear_1x1x1","padding":"replicate","pool":"average_2x2x2",
                            "upsampling":"nearest_to_skip_shape"}.items():
        require(architecture[field]==expected,"fixed architecture field")
    validate_inventory(value["weights_inventory"], descriptors(), "<f4")
    validate_inventory(value["predictions_inventory"], {"pressure":(32,16,16,16),"row_indices":(32,)},
                       {"pressure":"<f8","row_indices":"<i8"})
    require(canonical(value["witness_cases"]) == canonical(population()), "fixed case IDs/indices/types")
    require(type(value["per_case_inference_s"]) is list and len(value["per_case_inference_s"])==32, "inference population")
    for timing in value["per_case_inference_s"]:
        number(timing,"recorded inference",True)
    number(value["elapsed_s"],"historical elapsed",True)
    closure = value["closure"]
    keys(closure, {"config_sha256","input_evidence_sha256","input_sha256","normalization_sha256",
                   "runtime_snapshot_sha256","source_commit","source_witness_sha256","splits_sha256","witness_sha256"}, "closure")
    require(closure["source_commit"]==SOURCE_COMMIT and closure["source_witness_sha256"]==SOURCE_WITNESS_SHA,
            "frozen scientific source")
    keys(closure["input_sha256"], {"train0","train1","validation","splits","origin"},"packet closure")
    for slot,name in zip(("train0","train1","validation","splits","origin"),PACKET_SHAS):
        require(closure["input_sha256"][slot]==PACKET_SHAS[name], "packet source hash")
    for name in set(closure)-{"source_commit","input_sha256"}:
        hash_string(closure[name])
    for name in ("weights_sha256","predictions_sha256","sample_order_sha256","diagnostics_sha256","initialization_mapping_sha256"):
        hash_string(value[name])
    expected_checkpoints = {f"checkpoint-{u:03d}.npz" for u in range(0,201,20)}
    keys(value["checkpoints"],expected_checkpoints,"11 observation states")
    for name, entry in value["checkpoints"].items():
        keys(entry,{"bytes","inventory","sha256","update"},"checkpoint")
        integer(entry["update"],"checkpoint update",int(name[11:14]))
        integer(entry["bytes"],"checkpoint bytes")
        hash_string(entry["sha256"])
        validate_inventory(entry["inventory"],descriptors(),"<f4")
    require(canonical(value["checkpoints"]["checkpoint-200.npz"]["inventory"]) == canonical(value["weights_inventory"])
            and value["checkpoints"]["checkpoint-200.npz"]["sha256"]==value["weights_sha256"],"final200 identity")
    keys(value["outputs"], expected_checkpoints | {"diagnostics.json","normalization.json","predictions.npz",
                                                   "splits.json","summary.json","weights.npz"},"receipt outputs")
    for entry in value["outputs"].values():
        keys(entry,{"bytes","role","sha256"},"output declaration")
        integer(entry["bytes"],"output bytes")
        hash_string(entry["sha256"])
        require(type(entry["role"]) is str,"output role")
    require(type(value["training_history"]) is list and len(value["training_history"])==10,"training history")
    for update, entry in zip(range(20,201,20),value["training_history"]):
        keys(entry,{"gradient_norm_before_clip","train_loss","update"},"training observation")
        integer(entry["update"],"history update",update)
        number(entry["train_loss"],"history loss",True)
        number(entry["gradient_norm_before_clip"],"history gradient",True)
    require(type(value["validation"]) is list and len(value["validation"])==2,"retained validation observations")
    for update,entry in zip((0,200),value["validation"]):
        keys(entry,{"loss","update"},"validation observation")
        integer(entry["update"],"validation update",update)
        number(entry["loss"],"validation loss",True)
    runtime=value["runtime_observations"]
    keys(runtime,{"deterministic_algorithms","device","numpy_version","peak_rss_bytes","python_version",
                  "torch_interop_threads","torch_threads","torch_version"},"historical runtime")
    require(runtime["deterministic_algorithms"] is True and runtime["device"]=="cpu"
            and runtime["python_version"]=="3.12.14" and runtime["numpy_version"]=="2.3.5"
            and runtime["torch_version"]=="2.8.0+cpu", "qualified historical CPU")
    for field in ("torch_threads","torch_interop_threads"):
        integer(runtime[field],field,1)
    integer(runtime["peak_rss_bytes"],"historical RSS")
    require(runtime["peak_rss_bytes"]<=4*1024**3,"historical RAM cap")


def zero_disposition(candidate, base, residual):
    require(candidate.dtype.str==base.dtype.str==residual.dtype.str=="<f8"
            and candidate.shape==base.shape==residual.shape==(4096,)
            and all(np.isfinite(v).all() for v in (candidate,base,residual)), "typed cached pressure/residual")
    zero = bool(np.all(residual==0))
    # Match native's serial f64 sum(r*r), not a stable scaled norm: a nonzero
    # vector whose squares underflow must not masquerade as the exact bypass.
    squared = 0.0
    for value in residual:
        squared += float(value)*float(value)
    require(math.isfinite(squared) and (zero or squared>0), "native residual squared norm underflow/overflow")
    if zero:
        require(candidate.tobytes()==base.tobytes(), "zero native input must retain exact base bytes")
    return zero


def residual_metrics(residual):
    squared=0.0
    maximum=0.0
    for item in residual:
        value=float(item)
        squared += value*value
        maximum=max(maximum,abs(value))
    require(math.isfinite(squared),"attested native residual norm overflow")
    return math.sqrt(squared),maximum


def raw_array(inputs, root, entry, shape, dtype, units):
    keys(entry,{"bytes","dtype","layout","path","sha256","shape_xyz","units"},"capture array")
    require(type(entry["shape_xyz"]) is list and all(type(n) is int for n in entry["shape_xyz"])
            and entry["shape_xyz"]==list(shape) and entry["dtype"]==dtype
            and entry["layout"]=="x-fastest" and entry["units"]==units,"capture physical shape/dtype/units")
    integer(entry["bytes"],"raw size",math.prod(shape)*np.dtype(dtype).itemsize)
    relative=Path(entry["path"])
    require(not relative.is_absolute() and ".." not in relative.parts,"capture path escape")
    blob=inputs.read(root/relative,entry["bytes"],entry["sha256"])
    value=np.frombuffer(blob,dtype=dtype)
    require(np.isfinite(value).all(),"capture finite array")
    return value


def validate_originating_inputs(declarations, training_report, originating):
    bindings=training_report["typed_restore_inputs"]
    require(type(bindings) is dict and len(bindings)==17 and type(declarations) is dict,"17 historical typed originating inputs")
    require(set(declarations)==set(bindings)|{"origin","splits","train0","train1","validation"},"closed restore inputs")
    for slot,binding in bindings.items():
        keys(binding,{"artifact_name","expected_sha256","kind","run_id"},"historical Run binding")
        require(binding["kind"]=="run_artifact" and binding["run_id"]==TRAIN_RUN,"original Run artifact")
        declaration=declarations[slot]
        keys(declaration,{"kind","source"},"restore artifact input")
        require(declaration["kind"]=="run_artifact","typed inherited input")
        source=declaration["source"]
        keys(source,{"artifact_bytes","artifact_name","artifact_role","attempt_id","catalog_id","custody_identity",
                     "custody_receipt_base64","custody_receipt_sha256","material_sha256","measured_sha256",
                     "request_sha256","run_id","schema_version"},"historical source artifact")
        integer(source["schema_version"],"source artifact schema",1)
        artifact=training_report["artifact_inventory"][binding["artifact_name"]]
        integer(source["artifact_bytes"],"inherited artifact bytes",artifact["bytes"])
        require(source["artifact_name"]==binding["artifact_name"] and source["artifact_role"]==artifact["role"]
                and source["measured_sha256"]==binding["expected_sha256"]==artifact["sha256"],"inherited artifact relationship")
        for field in ("attempt_id","catalog_id","custody_identity","custody_receipt_base64","custody_receipt_sha256",
                      "material_sha256","request_sha256","run_id"):
            require(canonical(source[field])==canonical(originating[field]),"inherited source custody: "+field)


def validate_runtime_snapshot(runtime):
    """Decode historical snapshot metadata; micro-version belongs to observations.

    The accepted snapshot schema records Python major.minor, as the scientific
    trainer checks it. Admission separately requires local Python3.12.14. This
    validates the already hash-bound manifest and does not reread runtime payloads.
    """
    keys(runtime,{"schema_version","profile","python_version","packages","launch","files","total_bytes"},
         "runtime snapshot metadata")
    integer(runtime["schema_version"],"runtime snapshot schema",1)
    require(runtime["profile"]=="captured_cpu_python_v1" and runtime["python_version"]=="3.12",
            "runtime snapshot major.minor metadata")
    packages=runtime["packages"]
    require(type(packages) is dict and all(type(k) is str and type(v) is str and k and v
                                         for k,v in packages.items())
            and packages.get("numpy")=="2.3.5" and packages.get("torch")=="2.8.0+cpu",
            "runtime snapshot packages")
    launch=runtime["launch"]
    keys(launch,{"dynload","home","library_dirs","loader","packages","python","stdlib"},"runtime launch")
    require(type(launch["library_dirs"]) is list and all(type(v) is str and v for v in launch["library_dirs"]),
            "runtime library directories")
    for value in [*(launch[n] for n in set(launch)-{"library_dirs"}),*launch["library_dirs"]]:
        require(type(value) is str and value and not Path(value).is_absolute()
                and ".." not in Path(value).parts,"relative runtime launch metadata")
    require(type(runtime["files"]) is list and 0<len(runtime["files"])<=25000,"runtime file manifest bound")
    seen=set();total=0
    for entry in runtime["files"]:
        keys(entry,{"bytes","kind","mode","path","sha256"},"runtime file metadata")
        integer(entry["bytes"],"runtime file bytes")
        integer(entry["mode"],"runtime file mode")
        require(entry["mode"] in (256,320) and entry["kind"] in ("data","elf"),"runtime file kind/mode")
        relative=entry["path"]
        require(type(relative) is str and relative and not Path(relative).is_absolute()
                and ".." not in Path(relative).parts and relative not in seen,"runtime unique relative file")
        hash_string(entry["sha256"])
        seen.add(relative);total+=entry["bytes"]
    integer(runtime["total_bytes"],"runtime manifest total",total)


def prepared_modes(prepared, expected, deadline):
    actual={}
    for path in prepared.rglob("*"):
        check_deadline(deadline)
        require(not path.is_symlink(),"prepared source symlink")
        if path.is_dir():
            continue
        regular(path)
        name=path.relative_to(prepared).as_posix()
        if name!="source-receipt.json":
            actual[name]="100755" if path.stat().st_mode&0o111 else "100644"
    require(actual==expected,"prepared source mode/content path closure")


def admit_prepared(inputs, prepared, receipt):
    require(prepared.resolve()==prepared and prepared.is_dir(),"prepared canonical directory")
    expected=strict_json(inputs.read(LAB/"experiments/global-pressure-v1/dependency/source-files.json",2*MIB,
                                    receipt["source_files_sha256"]))
    require(type(expected) is list,"prepared file inventory")
    modes={}
    for entry in expected:
        keys(entry,{"path","mode","sha256"},"prepared entry")
        relative=Path(entry["path"])
        require(not relative.is_absolute() and ".." not in relative.parts and entry["path"] not in modes
                and entry["mode"] in ("100644","100755"),"prepared safe unique file")
        modes[entry["path"]]=entry["mode"]
        inputs.read(prepared/relative,8*MIB,entry["sha256"])
    actual=strict_json(inputs.read(prepared/"source-receipt.json",MIB))
    require(type(actual.get("schema_version")) is int and canonical(actual)==canonical(receipt),"prepared exact typed receipt")
    prepared_modes(prepared,modes,inputs.deadline)
    return modes


def admit(paths, deadline):
    """Read only. Return original requests plus independently bounded custody proof."""
    keys(paths,{"training_artifacts","restored_artifacts","capture","packet","binary"},"admission paths")
    require(sys.version_info[:3]==(3,12,14) and np.__version__=="2.3.5", "qualified Python/NumPy runtime")
    thread_env={name:os.environ.get(name) for name in ("RAYON_NUM_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}
    require(all(value=="1" for value in thread_env.values()),"explicit one-thread local environment")
    paths={k:Path(v).absolute() for k,v in paths.items()}
    require(all(p.resolve()==p for p in paths.values()),"canonical input locators")
    inputs=Inputs(deadline)
    reports={}
    stages={"training":paths["training_artifacts"],"restore":paths["restored_artifacts"]}
    receipts={}
    for kind,root in stages.items():
        reportpath=LAB/"docs/plans/global-pressure-groupnorm-v1/reports"/(kind+".json")
        report=strict_json(inputs.read(reportpath,MIB,REPORT_SHAS[kind]))
        reports[kind]=report
        for name in ("artifact-manifest.json","custody-receipt.json","terminal-run.json"):
            metadata=strict_json(inputs.read(root.parent/name,MIB,report["evidence_file_sha256"][name]))
            if name=="artifact-manifest.json":
                keys(metadata,{"artifacts","custody_receipt_sha256","material_sha256","request_sha256","run_id","schema_version"},"artifact manifest")
                integer(metadata["schema_version"],"artifact manifest schema",1)
                require(metadata["run_id"]==(TRAIN_RUN if kind=="training" else RESTORE_RUN),"artifact originating Run")
                actual={e["name"]:{k:v for k,v in e.items() if k!="name"} for e in metadata["artifacts"]}
                require(len(actual)==18 and len(metadata["artifacts"])==18 and canonical(actual)==canonical(report["artifact_inventory"]),"complete historical artifact manifest")
            elif name=="custody-receipt.json":
                require(metadata["attempt_id"]==(TRAIN_RUN if kind=="training" else RESTORE_RUN)
                        and all(metadata[k] is True for k in ("drained","echild","go_received","spawn_closed"))
                        and type(metadata["leader_returncode"]) is int and metadata["leader_returncode"]==0
                        and metadata["outcome"]=="completed","historical drained custody")
            else:
                require(metadata["id"]==(TRAIN_RUN if kind=="training" else RESTORE_RUN)
                        and metadata["state"]=="completed" and metadata["custody_phase"]=="drained", "historical terminal Run")
        inventory=report["artifact_inventory"]
        require({p.name for p in root.iterdir()}==set(inventory),"closed artifact directory")
        blobs={name:inputs.read(root/name,8*MIB if name.endswith(".npz") else 2*MIB,entry["sha256"])
               for name,entry in inventory.items()}
        require(all(len(blobs[name])==entry["bytes"] for name,entry in inventory.items()),"artifact exact sizes")
        receipt=strict_json(blobs["receipt.json"])
        validate_receipt(receipt,"train" if kind=="training" else "restore")
        for field,name in (("weights_sha256","weights.npz"),("predictions_sha256","predictions.npz"),
                           ("diagnostics_sha256","diagnostics.json")):
            require(receipt[field]==inventory[name]["sha256"],"receipt output hash relationship")
        require(receipt["closure"]["normalization_sha256"]==inventory["normalization.json"]["sha256"]
                and receipt["closure"]["splits_sha256"]==inventory["splits.json"]["sha256"],"receipt retained preprocessing relationship")
        for name,entry in receipt["outputs"].items():
            require(name in inventory and canonical(entry)==canonical({k:v for k,v in inventory[name].items() if k!="media_type"}),"receipt artifact inventory")
        require(set(receipt["outputs"])==set(inventory)-{"receipt.json"},"receipt inherited outputs")
        for name,entry in receipt["checkpoints"].items():
            decode_npz(blobs[name],entry["inventory"])
        decode_npz(blobs["weights.npz"],receipt["weights_inventory"])
        require(blobs["weights.npz"]==blobs["checkpoint-200.npz"],"weights exactly checkpoint200")
        decoded=decode_npz(blobs["predictions.npz"],receipt["predictions_inventory"])
        require(decoded["row_indices"].tolist()==[c["row_index"] for c in population()],"exact int64 witness rows")
        receipts[kind]=receipt
        if kind=="training":
            inherited={n:digest(b) for n,b in blobs.items() if n not in ("receipt.json","summary.json")}
        else:
            require(all(digest(blobs[n])==sha for n,sha in inherited.items()),"16 inherited artifacts byte identity")
            candidates=decoded["pressure"]
        check_deadline(deadline)
    train,restored=receipts["training"],receipts["restore"]
    for field in RECEIPT_KEYS-{"elapsed_s","mode","optimizer_updates","originating_training_run","per_case_inference_s","runtime_observations","closure","outputs"}:
        require(canonical(train[field])==canonical(restored[field]),"train/restore inherited receipt field: "+field)
    for name in set(train["outputs"])-{"summary.json"}:
        require(canonical(train["outputs"][name])==canonical(restored["outputs"][name]),"inherited output declaration")
    require(train["originating_training_run"] is None,"original train provenance")
    originating=restored["originating_training_run"]
    keys(originating,{"attempt_id","catalog_id","custody_identity","custody_receipt_base64","custody_receipt_sha256",
                      "material_sha256","request_sha256","run_id"},"originating train")
    require(originating["run_id"]==originating["attempt_id"]==TRAIN_RUN and
            originating["catalog_id"]=="619034b1109f45faac099835ee9a9065" and
            originating["material_sha256"]==reports["training"]["run"]["material_sha256"] and
            originating["custody_receipt_sha256"]==reports["training"]["evidence_file_sha256"]["custody-receipt.json"] and
            digest(base64.b64decode(originating["custody_receipt_base64"],validate=True))==originating["custody_receipt_sha256"], "restore originating custody")
    for field in set(train["closure"])-{"input_evidence_sha256"}:
        require(canonical(train["closure"][field])==canonical(restored["closure"][field]),"shared frozen closure")
    historical_preflight=strict_json(inputs.read(paths["restored_artifacts"].parent/"fresh-preflight.json",MIB,
                                 reports["restore"]["evidence_file_sha256"]["fresh-preflight.json"]))
    validate_originating_inputs(historical_preflight["material"]["inputs"],reports["training"],originating)
    source_root=LAB/"runs/global-pressure-v1-project/groupnorm-smoke-fd3a600"
    source=strict_json(inputs.read(source_root/"source-witness.json",MIB,SOURCE_WITNESS_SHA))
    require(source["source_commit"]==SOURCE_COMMIT and source["native_residual_origin_sha256"]==RESIDUAL_ANCHOR,"scientific witness source/anchor")
    for name,sha in source["executable_sources"].items():
        require(Path(name).name==name,"flat scientific source")
        inputs.read(source_root/name,MIB,sha)  # bytes only, never import scientific code
    require(source["config_sha256"]==restored["closure"]["config_sha256"] and
            source["runtime_snapshot_sha256"]==restored["closure"]["runtime_snapshot_sha256"] and
            source["native_qualification"]["operator_binary_sha256"]==NATIVE_SHA,"scientific/runtime/native closure")
    inputs.read(source_root/"config.json",MIB,source["config_sha256"])
    runtime=strict_json(inputs.read(source_root/"runtime-manifest.json",8*MIB,source["runtime_snapshot_sha256"]))
    validate_runtime_snapshot(runtime)
    binary=paths["binary"]
    inputs.read(binary,2*MIB,NATIVE_SHA)
    build=strict_json(inputs.read(binary.parent/"build-receipt.json",MIB,BUILD_SHA))
    identity=build["identity"]
    require(build["status"]=="qualified_build" and build["binary_path"]==str(binary)
            and build["binary_sha256"]==NATIVE_SHA and type(identity["native_threads"]) is int
            and identity["native_threads"]==1,"archived native binary closure")
    binding=identity["binding"]
    require(canonical(strict_json(inputs.read(binary.parent/"binding.json",MIB)))==canonical(binding),"native build binding")
    metadata=strict_json(inputs.read(binary.parent/"cargo-metadata.json",8*MIB,binding["metadata_sha256"]))
    native=[p for p in metadata["packages"] if p["name"]=="rheon"]
    require(len(native)==1 and native[0]["source"] is None and
            native[0]["manifest_path"]==binding["resolved_native_manifest"],"actual archived Cargo dependency")
    for name,sha in binding["source_sha256"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts,"native source path")
        inputs.read(LAB/name,2*MIB,sha)
    prepared=Path(binding["prepared_source"])
    modes=admit_prepared(inputs,prepared,identity["native_receipt"])
    reader=import_pinned(LAB/"experiments/global-pressure-v1/training/data.py",READER_SHA,inputs,"gn_native_packet_reader")
    packet=paths["packet"]
    packet_sizes={n:regular(packet/n).stat().st_size for n in PACKET_SHAS}
    require(sum(packet_sizes.values())<=INPUT_LIMIT-inputs.bytes_read,"reader cumulative input preallocation")
    check_deadline(deadline)
    dataset=reader.load_data(*(packet/n for n in PACKET_SHAS),expected_native_residual_origin_sha256=RESIDUAL_ANCHOR)
    check_deadline(deadline)
    for name,sha in PACKET_SHAS.items():
        require(dataset.input_hashes[name]==sha,"packet exact identity")
        slot={"train-0.npz":"train0","train-1.npz":"train1","validation.npz":"validation",
              "splits.json":"splits","origin.json":"origin"}[name]
        integer(packet_sizes[name],"packet expected bytes",source["package_inputs"][slot]["bytes"])
        inputs.external_snapshot(packet/name,sha,packet_sizes[name])
    origin_blob=inputs.read(paths["capture"]/"manifest.json",2*MIB,ORIGIN_SHA)
    origin=reader.read_origin(origin_blob)
    attestation=dataset.splits["native_residual_origin"]
    require(canonical(attestation["native_identity"]["native_receipt"])==canonical(identity["native_receipt"]),"residual/native prepared operator equality")
    require(canonical(attestation["native_identity"]["binding"])==canonical(identity["binding"]),"residual/native compiled operator binding equality")
    require(attestation["native_identity"]["source_sha256"]==identity["source_sha256"],"residual/native full operator source equality")
    expected=population(); cases=[]; requests=[]; bases=[]; references=[]; residuals=[]
    for case in expected:
        t=origin["trajectories"][case["row_index"]//24];frame=t["frames"][case["frame"]]
        previous=t["initial"]["pressure"] if case["frame"]==0 else t["frames"][22]["arrays"]["pressure"]
        require(frame["previous_pressure"]==previous["path"],"previous accepted pressure lineage")
        values={"original_rhs":raw_array(inputs,paths["capture"],frame["arrays"]["rhs"],(16,16,16),"<f8","m3/s2"),
                "reference_pressure":raw_array(inputs,paths["capture"],frame["arrays"]["pressure"],(16,16,16),"<f8","Pa"),
                "previous_pressure":raw_array(inputs,paths["capture"],previous,(16,16,16),"<f8","Pa")}
        arrays=dataset.train if case["split"]=="train" else dataset.validation
        index=case["row_index"] if case["split"]=="train" else case["row_index"]-288
        for packed,raw in (("rhs","original_rhs"),("base","previous_pressure"),("reference","reference_pressure")):
            require(arrays[packed][index].tobytes()==values[raw].tobytes(),"packed/raw selected field correspondence")
        spacing=[h/16 for h in t["geometry"]["extents_m"]]
        require(arrays["spacing"][index].tolist()==spacing and arrays["rho"][index]==t["geometry"]["density_kg_m3"]
                and arrays["dt"][index]==t["reservation_entry"]["dt_s"],"packed/raw physical correspondence")
        candidate=candidates[len(cases)].reshape(-1);residual=arrays["native_input_residual"][index].reshape(-1)
        zero=zero_disposition(candidate,values["previous_pressure"],residual)
        base_l2,base_max=residual_metrics(residual)
        velocity=[raw_array(inputs,paths["capture"],frame["arrays"]["pre_"+axis],shape,"<f4","m/s")
                  for axis,shape in zip("uvw",((17,16,16),(16,17,16),(16,16,17)))]
        request={"schema":"rheon-global-pressure-teacher-case-v1","counts_xyz":[16]*3,"spacing_m":spacing,
                 "density_kg_m3":float(arrays["rho"][index]),"dt_s":float(arrays["dt"][index]),
                 **{n:v.tolist() for n,v in values.items()},"pre_velocity":[v.tolist() for v in velocity],
                 "candidate_pressure":(values["previous_pressure"] if zero else candidate).tolist(),"candidate_error":None}
        require(len(canonical(request))<=MIB,"native request1MiB bound")
        diff=candidate-values["reference_pressure"]
        cases.append({**case,"zero_residual_base_retained":zero,
                      "candidate_raw_sha256":digest(candidate.tobytes()),"base_raw_sha256":digest(values["previous_pressure"].tobytes()),
                      "reference_raw_sha256":digest(values["reference_pressure"].tobytes()),"input_residual_raw_sha256":digest(residual.tobytes()),
                      "native_base_residual_l2":base_l2,"native_base_residual_max":base_max,
                      "raw_pressure_error_l2_pa":float(np.linalg.norm(diff)),
                      "raw_pressure_relative_l2":float(np.linalg.norm(diff)/max(np.linalg.norm(values["reference_pressure"]),1e-12))})
        requests.append(request);bases.append(values["previous_pressure"]);references.append(values["reference_pressure"]);residuals.append(residual)
    proof={"schema":"rheon-groupnorm-native-admission-v1","candidate_update":200,"training_run":TRAIN_RUN,
           "restore_run":RESTORE_RUN,"source_commit":SOURCE_COMMIT,"residual_anchor":RESIDUAL_ANCHOR,
           "zero_disposition_count":sum(c["zero_residual_base_retained"] for c in cases),
           "input_bytes_read":inputs.bytes_read,"read_limit_bytes":INPUT_LIMIT,
           "native_identity":identity,"source_witness_sha256":SOURCE_WITNESS_SHA,
           "runtime_observations":{"python_version":".".join(map(str,sys.version_info[:3])),"python_sys_version":sys.version,
                                   "sys_executable":sys.executable,"numpy_version":np.__version__,"thread_environment":thread_env,
                                   "route":"current local NumPy/stdlib process; not a new Tuldok captured runtime"},
           "custody_scope":"retained hash-bound metadata; no new managed Run or live catalog audit"}
    return {"population":cases,"requests":requests,"raw_candidates":candidates,"base":np.stack(bases),
            "reference":np.stack(references),"native_input_residual":np.stack(residuals),
            "inference":restored["per_case_inference_s"],"identity":identity,"consumed_files":inputs.files,
            "proof":proof,"_inputs":inputs,"_prepared_modes":modes,"_prepared":prepared}


def verify_unchanged(admitted, deadline):
    inputs=admitted["_inputs"];inputs.deadline=deadline
    try:
        for name,entry in list(admitted["consumed_files"].items()):
            inputs.read(Path(name),entry["bytes"],entry["sha256"])
        prepared_modes(admitted["_prepared"],admitted["_prepared_modes"],deadline)
    finally:
        admitted["proof"]["input_bytes_read"]=inputs.bytes_read


QUALITY_KEYS=set("qualified failures residual_l2 residual_max rhs_l2 relative_residual predicted_divergence_max_s_inv actual_divergence_max_s_inv actual_divergence_rms_s_inv net_flux_m3_s wall_normal_flux_exact_zero courant gauge_cell".split())
SOLVE_KEYS=set("qualified solver_error iterations iteration_limit initial_guess_sha256 initial_guess_kind solver_s verification_s quality".split())
CORRECTED_KEYS=set("qualified outcome proposal_rejection warm reset any_pcg reset_fallback elapsed_s".split())


def validate_quality(q, request):
    require(type(q) is dict and type(q.get("qualified")) is bool and type(q.get("failures")) is list
            and all(type(v) is str for v in q["failures"]),"native quality outcome types")
    if set(q)=={"qualified","failures"}:
        require(q["qualified"] is False and len(q["failures"])>0,"native compact failure")
        return
    keys(q,QUALITY_KEYS,"native quality")
    require(q["gauge_cell"]==0 and type(q["gauge_cell"]) in (int,float),"native full-quality pressure gauge")
    require(type(q["wall_normal_flux_exact_zero"]) is bool,"wall bool")
    for name in QUALITY_KEYS-{"qualified","failures","wall_normal_flux_exact_zero"}:
        require(q[name] is None or type(q[name]) in (int,float) and math.isfinite(q[name]),"native quality number/null")
    require(q["qualified"] is (not q["failures"]),"quality status/failures consistency")
    expected_rhs=math.sqrt(sum(float(v)*float(v) for v in request["original_rhs"]))
    require(q["rhs_l2"]==expected_rhs,"reported original-b norm")
    if q["residual_max"] is not None:
        require(q["predicted_divergence_max_s_inv"]==request["dt_s"]*q["residual_max"]/math.prod(request["spacing_m"]),"predicted divergence units/original grid")
    if q["residual_l2"] is not None:
        require(q["relative_residual"]==(q["residual_l2"]/expected_rhs if expected_rhs>0 else None),"reported full relative original-b residual")
    failset=set(q["failures"])
    require(len(failset)==len(q["failures"]) and failset<=set("original_rhs_full_residual predicted_divergence actual_f32_divergence net_volume_flux wall_normal_flux courant".split()),"native quality failure vocabulary")
    metric_gates=(("original_rhs_full_residual","residual_l2",max(1e-12,1e-8*expected_rhs)),
                  ("predicted_divergence","predicted_divergence_max_s_inv",1e-7),
                  ("actual_f32_divergence","actual_divergence_max_s_inv",1e-5),
                  ("courant","courant",.5))
    for reason,metric,limit in metric_gates:
        require((reason in failset) is (q[metric] is None or q[metric]>limit),"native gate flag/metric agreement")
    require(("wall_normal_flux" in failset) is (not q["wall_normal_flux_exact_zero"]),"wall gate flag agreement")
    require(q["net_flux_m3_s"] is not None or "net_volume_flux" in failset,"finite net flux gate")
    if q["net_flux_m3_s"] is not None and abs(q["net_flux_m3_s"])<=1e-12:
        require("net_volume_flux" not in failset,"net flux below absolute floor")
    if q["qualified"]:
        for n in QUALITY_KEYS-{"qualified","failures","wall_normal_flux_exact_zero","relative_residual"}:
            number(q[n],n)
        require(q["residual_l2"]<=max(1e-12,1e-8*q["rhs_l2"])
                and q["predicted_divergence_max_s_inv"]<=1e-7 and q["actual_divergence_max_s_inv"]<=1e-5
                and q["wall_normal_flux_exact_zero"] is True and q["courant"]<=.5 and q["gauge_cell"]==0,
                "native accepted common gates")
    # The absolute-flux denominator is not returned. Its exact gate remains the
    # qualified unchanged native binary's contract, never reconstructed here.
    for n in ("residual_l2","residual_max","rhs_l2","predicted_divergence_max_s_inv","actual_divergence_max_s_inv","actual_divergence_rms_s_inv","courant"):
        require(q[n] is None or q[n]>=0,"nonnegative native metric")


def validate_solve(s, request, limit, guess):
    keys(s,SOLVE_KEYS,"native solve")
    integer(s["iteration_limit"],"native iteration limit",limit)
    require(type(s["qualified"]) is bool and (s["solver_error"] is None or type(s["solver_error"]) is str),"native solve status")
    require(s["iterations"] is None or type(s["iterations"]) is int and 0<=s["iterations"]<=limit,"native attempted/completed iteration bound")
    require(not s["qualified"] or type(s["iterations"]) is int,"successful solve iteration count")
    for field in ("solver_s","verification_s"):
        number(s[field],field,True)
    expected=digest(bytes(4096*8)) if guess is None else digest(np.asarray(guess,dtype="<f8").tobytes())
    require(s["initial_guess_sha256"]==expected and s["initial_guess_kind"]==("zero" if guess is None else "supplied_pressure"),"native initial start custody")
    validate_quality(s["quality"],request)
    require(s["qualified"] is (s["solver_error"] is None and s["quality"]["qualified"]),"solve qualification consistency")


def validate_corrected(s,request,raw,base,guess):
    keys(s,CORRECTED_KEYS,"native corrected policy")
    for n in ("qualified","any_pcg","reset_fallback"):
        require(type(s[n]) is bool,"corrected bool")
    number(s["elapsed_s"],"corrected elapsed",True)
    require(s["proposal_rejection"] is None or type(s["proposal_rejection"]) is str,"proposal rejection type")
    warm,reset=s["warm"],s["reset"]
    if warm is not None:validate_solve(warm,request,32,guess)
    if reset is not None:validate_solve(reset,request,3000,request["previous_pressure"])
    if s["outcome"]=="direct_acceptance":
        require(raw["qualified"] and s["qualified"] and s["proposal_rejection"] is None and warm is None and reset is None and not s["any_pcg"] and not s["reset_fallback"],"direct policy")
    elif s["outcome"]=="warm_correction":
        require(warm is not None and warm["qualified"] and reset is None and s["qualified"] and s["proposal_rejection"] is None and s["any_pcg"] and not s["reset_fallback"],"warm policy")
    else:
        require(s["outcome"] in ("temporal_reset_fallback","terminal_failure") and reset is not None
                and s["qualified"] is reset["qualified"] and s["any_pcg"] and s["reset_fallback"]
                and (warm is None and s["proposal_rejection"] is not None or warm is not None and not warm["qualified"] and s["proposal_rejection"] is None)
                and s["outcome"]==("temporal_reset_fallback" if reset["qualified"] else "terminal_failure"),"single fresh-base reset policy")
    if raw["qualified"] and not (raw.get("residual_l2",math.inf)>base.get("residual_l2",-math.inf)):
        require(s["outcome"]=="direct_acceptance","qualified eligible proposal must accept directly")
    if s["proposal_rejection"] is not None:
        require(warm is None and reset is not None,"rejection skips warm")
    if raw.get("residual_l2") is not None and base.get("residual_l2") is not None and raw["residual_l2"]>base["residual_l2"]:
        require(s["proposal_rejection"]=="proposal_residual_worse_than_temporal" and warm is None,"worse proposal must skip warm")


def validate_result(request,result,case):
    """Validate response contract; caller journals then stops on terminal failure."""
    keys(result,{"schema","scope","reference","raw_learned","raw_temporal","raw_zero","learned_jacobi",
                 "temporal_jacobi","temporal_full_budget","zero_jacobi","native_elapsed_s","state_advanced","cost_scope"},"native result")
    require(result["schema"]=="rheon-global-pressure-teacher-diagnostic-v1" and result["state_advanced"] is False
            and result["scope"]=="teacher-state development snapshots; no autonomous rollout or acceleration qualification"
            and result["cost_scope"]=="native diagnostic includes solver/correction/verification; excludes model inference and generic custody", "teacher native scope")
    number(result["native_elapsed_s"],"native elapsed",True)
    for field in ("reference","raw_learned","raw_temporal","raw_zero"):
        validate_quality(result[field],request)
    require(result["raw_temporal"].get("residual_l2")==case["native_base_residual_l2"]
            and result["raw_temporal"].get("residual_max")==case["native_base_residual_max"],
            "attested/current native full base-residual correspondence")
    require(result["reference"]["qualified"] is True,"reference must qualify")
    validate_corrected(result["learned_jacobi"],request,result["raw_learned"],result["raw_temporal"],request["candidate_pressure"])
    temporal=dict(result["temporal_jacobi"])
    require(temporal.pop("comparison_role",None)=="bounded32-step temporal diagnostic with one p_base reset; not the full-budget temporal baseline","temporal role")
    validate_corrected(temporal,request,result["raw_temporal"],result["raw_temporal"],request["previous_pressure"])
    full=dict(result["temporal_full_budget"])
    require(full.pop("comparison_role",None)=="temporal PCG baseline: one fresh full-budget attempt from previous accepted pressure","full temporal role")
    validate_solve(full,request,3000,request["previous_pressure"])
    validate_solve(result["zero_jacobi"],request,3000,None)
    if case["zero_residual_base_retained"]:
        require(np.asarray(request["candidate_pressure"],dtype="<f8").tobytes()==np.asarray(request["previous_pressure"],dtype="<f8").tobytes()
                and result["raw_temporal"].get("residual_l2")==0
                and canonical(result["raw_temporal"])==canonical(result["raw_learned"]),"zero-base native consistency")
