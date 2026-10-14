#!/usr/bin/env python3
"""Qualified native build and eager immutable DEVELOPMENT-only capture caller."""
import argparse
import hashlib
import contextlib
import signal
import threading
import json
import os
from pathlib import Path
import resource
import stat
import shutil
import struct
import subprocess
import sys
import time

from verify_capture import strict_json

LAB=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RESERVATION=LAB/"docs/plans/global-pressure-correction-v1/reservation.json"
MANIFEST=LAB/"probes/global-pressure-v1/Cargo.toml"
SOURCE_FILES=["probes/global-pressure-v1/Cargo.toml","probes/global-pressure-v1/Cargo.lock","probes/global-pressure-v1/build.rs","probes/global-pressure-v1/src/main.rs","experiments/global-pressure-v1/capture.py","experiments/global-pressure-v1/capture_config.json","experiments/global-pressure-v1/verify_capture.py","experiments/global-pressure-v1/prepare_dependency.py","experiments/global-pressure-v1/dependency/manifest.json","experiments/global-pressure-v1/dependency/source-files.json","experiments/global-pressure-v1/dependency/LICENSE"]

def digest(data):return hashlib.sha256(data).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def require(condition,message):
    if not condition:raise ValueError(message)
def source_hashes():
    from prepare_dependency import load_contract
    dependency, _, _, _ = load_contract()
    files = SOURCE_FILES + ["experiments/global-pressure-v1/dependency/" + dependency["patch_file"]]
    return {relative:digest((LAB/relative).read_bytes()) for relative in files}
def write_new(path,value):
    with path.open("xb") as stream:stream.write(canonical(value)+b"\n")
def new_output(path,namespace):
    require(path.is_absolute(),"output must be absolute")
    expected=LAB/"runs"/namespace
    require(path.parent.resolve()==expected.resolve() and path.name not in (".",".."),f"output must be a fresh direct child of {expected}")
    current=Path(path.anchor)
    for part in path.parts[1:]:
        current=current/part
        require(not current.is_symlink(),f"unsafe output path: {current}")
    expected.mkdir(parents=True,exist_ok=True)
    path.mkdir()  # exclusive creation, refuses overwrite including partial output
    return path

def native_receipt(path):
    from prepare_dependency import verify_prepared
    receipt=verify_prepared(path)
    require(type(receipt.get("schema_version")) is int and receipt["schema_version"]==1 and all(type(value) is str for key,value in receipt.items() if key!="schema_version"),"source_identity_type: native receipt primitives")
    return receipt

def build(args):
    require(not args.prepared_source.is_symlink(),"prepared source symlink prohibited")
    prepared=args.prepared_source.resolve();receipt=native_receipt(prepared)
    require(receipt["base_commit"]=="2209012bc5f902693d0f695ec1c2d854126af317","native pin")
    output=new_output(args.output,"global-pressure-v1-build")
    config=f'patch."https://github.com/MrScripty/Rheon.git".rheon.path={json.dumps(str(prepared))}'
    # Both graph observation and build share exactly these Cargo configuration
    # arguments and this bounded environment; no caller-supplied resolver flags.
    env={k:v for k,v in os.environ.items() if not k.startswith(("CARGO_","RHEON_","RUSTFLAGS","RUSTC","RUSTDOC","RUSTUP_TOOLCHAIN"))}
    for key in ("CARGO_HOME","RUSTUP_HOME"):
        if key in os.environ:env[key]=os.environ[key]
    env["CARGO_TARGET_DIR"]=str(LAB/"runs/global-pressure-v1-target")
    env["RAYON_NUM_THREADS"]="1"
    env["RUSTUP_TOOLCHAIN"]="1.92.0"
    env["RUSTC"]=str(args.cargo.parent/"rustc")
    env["RUSTDOC"]=str(args.cargo.parent/"rustdoc")
    env["RUSTC_WRAPPER"]=""
    env["RUSTC_WORKSPACE_WRAPPER"]=""
    cargo_version=subprocess.check_output([str(args.cargo),"--version"],env=env,text=True).strip()
    rustc_version=subprocess.check_output([str(args.cargo.parent/"rustc"),"-vV"],env=env,text=True).strip()
    require(cargo_version.startswith("cargo 1.92.0 ") and "release: 1.92.0" in rustc_version,"exact Rust/Cargo1.92.0 required")
    common=[str(args.cargo),"--manifest-path",str(MANIFEST),"--config",config]
    # Cargo places the subcommand before its arguments.
    metadata_cmd=[common[0],"metadata",*common[1:],"--format-version","1","--locked"]
    metadata=strict_json(subprocess.check_output(metadata_cmd,env=env,cwd=LAB))
    native=[p for p in metadata["packages"] if p["name"]=="rheon"]
    require(len(native)==1 and native[0]["source"] is None and Path(native[0]["manifest_path"]).resolve()==prepared/"Cargo.toml","actual Cargo graph must select exact prepared manifest")
    require(Path(metadata["workspace_root"]).resolve()==MANIFEST.parent,"actual Cargo workspace mismatch")
    metadata_bytes=canonical(metadata);(output/"cargo-metadata.json").write_bytes(metadata_bytes)
    hashes=source_hashes()
    command=[common[0],"build",*common[1:],"--locked","--release"]
    binding={"schema":"rheon-global-pressure-build-binding-v1","prepared_source":str(prepared),"native_receipt":receipt,"resolved_native_manifest":native[0]["manifest_path"],"metadata_sha256":digest(metadata_bytes),"cargo_command":command,"source_sha256":hashes,"toolchain":{"cargo":cargo_version,"rustc_vV":rustc_version,"rustc_path":env["RUSTC"],"rustdoc_path":env["RUSTDOC"],"rustup_toolchain":env["RUSTUP_TOOLCHAIN"]},"configuration":config}
    write_new(output/"binding.json",binding)
    env["RHEON_BUILD_BINDING"]=str(output/"binding.json");env["RHEON_BUILD_BINDING_SHA256"]=digest((output/"binding.json").read_bytes())
    started=time.monotonic();subprocess.run(command,env=env,cwd=LAB,check=True)
    require(native_receipt(prepared)==receipt and source_hashes()==hashes,"source drift during compilation")
    binary=output/"rheon-global-pressure-capture"
    shutil.copyfile(Path(env["CARGO_TARGET_DIR"])/"release/rheon-global-pressure-capture",binary);binary.chmod(0o555)
    identity=strict_json(subprocess.check_output([str(binary),"--identity"],env=env))
    require(identity["binding"]==binding and identity["native_receipt"]==receipt,"binary/build binding mismatch")
    build_receipt={"status":"qualified_build","binary_path":str(binary),"binary_sha256":digest(binary.read_bytes()),"identity":identity,"elapsed_s":time.monotonic()-started}
    write_new(output/"build-receipt.json",build_receipt)
    print(json.dumps(build_receipt,sort_keys=True))

def admit(args):
    # Admission precedes *all* output creation or scientific process execution.
    require(args.preset=="smoke","development_admission: full capture is prohibited")
    require(args.split in ("train","validation","development"),"development_admission: final-test capture is prohibited")
    reservation=strict_json(RESERVATION.read_bytes())
    entries=[e for e in reservation["presets"]["smoke"]["trajectories"] if e["split"] in ("train","validation") and (args.split=="development" or e["split"]==args.split)]
    if args.trajectory:
        entries=[e for e in entries if e["trajectory_id"]==args.trajectory]
        require(len(entries)==1,"development_admission: unreserved or inadmissible trajectory")
    if args.fixture_frames is not None:
        require(args.trajectory and len(entries)==1 and entries[0]["split"]=="train" and 1<=args.fixture_frames<=24,"development_admission: short fixture must select one TRAIN trajectory")
    return reservation,entries

@contextlib.contextmanager
def bounded_process(command, timeout_s, **kwargs):
    require(timeout_s > 0,"reference_physics: 600s generation budget exhausted; STOP for parent review")
    process=subprocess.Popen(command,start_new_session=True,**kwargs)
    expired=threading.Event()
    def kill_group():
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    def timeout():
        expired.set();kill_group()
    timer=threading.Timer(timeout_s,timeout);timer.daemon=True;timer.start()
    try:
        yield process
    finally:
        timer.cancel();kill_group();process.wait(timeout=2)
        for stream in (process.stdin,process.stdout,process.stderr):
            if stream is not None and not stream.closed:stream.close()
        require(not expired.is_set(),"reference_physics: 600s generation timeout; STOP for parent review")

def read_exact(stream,size):
    result=bytearray()
    while len(result)<size:
        data=stream.read(size-len(result))
        require(data,"producer_stream: truncated")
        result.extend(data)
    return bytes(result)

def capture(args):
    reservation,entries=admit(args)
    require(not args.binary.is_symlink(),"binary symlink prohibited")
    binary=args.binary.resolve();build_receipt=strict_json((binary.parent/"build-receipt.json").read_text())
    require(build_receipt["status"]=="qualified_build" and digest(binary.read_bytes())==build_receipt["binary_sha256"],"binary_hash: build receipt")
    identity=strict_json(subprocess.check_output([str(binary),"--identity"]))
    require(identity==build_receipt["identity"],"binary_identity: build receipt")
    require(digest((binary.parent/"cargo-metadata.json").read_bytes())==identity["binding"]["metadata_sha256"],"build_metadata: actual graph receipt")
    require(strict_json((binary.parent/"binding.json").read_text())==identity["binding"],"build_binding: persisted binding")
    require(identity["binding"]["source_sha256"]==source_hashes(),"source_hash: build source closure drift")
    require(not args.prepared_source.is_symlink(),"prepared source symlink prohibited")
    prepared=args.prepared_source.resolve();receipt=native_receipt(prepared)
    require(identity["native_receipt"]==receipt and identity["binding"]["prepared_source"]==str(prepared),"source_identity: native receipt")
    config_hash=digest((HERE/"capture_config.json").read_bytes());reservation_hash=digest(RESERVATION.read_bytes())
    require(identity["config_sha256"]==config_hash and identity["reservation_sha256"]==reservation_hash,"binary_identity: config/reservation")
    require(identity["native_threads"]==1 and not identity["parallel"],"native one-thread serial binary required")
    output=new_output(args.output,"global-pressure-v1-capture")
    manifest={"schema":"rheon-global-pressure-v1","status":"candidate","preset":"smoke","fixture":bool(args.fixture_frames),"selection":{"split":args.split,"trajectory_id":args.trajectory},"counts_xyz":[16]*3,"density_kg_m3":1.25,"config_sha256":config_hash,"reservation_sha256":reservation_hash,"provenance":{"native_receipt":receipt,"prepared_source":str(prepared),"binary_path":str(binary),"binary_sha256":digest(binary.read_bytes()),"source_sha256":source_hashes(),"build_receipt_sha256":digest((binary.parent/"build-receipt.json").read_bytes()),"lab_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=LAB,text=True).strip()},"trajectories":[]}
    start=time.monotonic()
    for entry in entries:
        tid=entry["trajectory_id"];directory=output/tid;directory.mkdir()
        extents=reservation["geometries_extents_m"][entry["geometry"]]
        geometry={"counts_xyz":[16]*3,"extents_m":extents,"origin_m":[0.0]*3,"density_kg_m3":1.25,"boundary":"sealed_zero_normal_mac","units":"SI"}
        trajectory={"trajectory_id":tid,"reservation_entry":entry,"geometry":geometry,"geometry_sha256":digest(canonical(geometry)),"initial":{},"frames":[]}
        frames=args.fixture_frames if args.fixture_frames is not None else entry["frames"]
        request={"preset":"smoke","trajectory_id":tid,"frames":frames,"fixture":bool(args.fixture_frames)}
        env=os.environ.copy();env["RAYON_NUM_THREADS"]="1"
        with (directory/"producer.stderr.log").open("xb") as stderr:
            with bounded_process([str(binary),"--produce"],strict_json((HERE/"capture_config.json").read_bytes())["generation_wall_limit_s"]-(time.monotonic()-start),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,env=env) as process:
                try:
                    process.stdin.write(canonical(request));process.stdin.close()
                    previous_pressure=None;previous_state=None
                    for stage in range(frames+2):
                        size=struct.unpack("<I",read_exact(process.stdout,4))[0];require(size<=65536,"producer_stream: header size")
                        header=strict_json(read_exact(process.stdout,size))
                        expected="initial" if stage==0 else "complete" if stage==frames+1 else "frame"
                        require(header["kind"]==expected,"producer_stream: stage order")
                        if expected=="complete":require(header["frames"]==frames,"producer_stream: coverage");continue
                        prefix="initial" if stage==0 else f"frame-{stage-1:03d}"
                        inventory={}
                        for descriptor in header.pop("arrays"):
                            name=descriptor.pop("name")
                            require(name in {"rhs","pressure","u","v","w","tracer","pre_u","pre_v","pre_w","corrected_u","corrected_v","corrected_w"} and name not in inventory,"producer_stream: array name")
                            require(isinstance(descriptor["bytes"],int) and 0<descriptor["bytes"]<=40000,"producer_stream: bounded array")
                            data=read_exact(process.stdout,descriptor["bytes"])
                            path=directory/f"{prefix}-{name}.bin"
                            with path.open("xb") as stream:stream.write(data)
                            path.chmod(0o444)
                            descriptor.update(path=str(path.relative_to(output)),sha256=digest(data));inventory[name]=descriptor
                        if expected=="initial":
                            trajectory["initial"]=inventory;previous_pressure=inventory["pressure"]["path"];previous_state={name:inventory[name]["path"] for name in ("u","v","w","tracer")}
                        else:
                            header.update(arrays=inventory,previous_pressure=previous_pressure,previous_state=previous_state)
                            trajectory["frames"].append(header)
                            previous_pressure=inventory["pressure"]["path"];previous_state={name:inventory["corrected_"+name]["path"] for name in ("u","v","w")};previous_state["tracer"]=inventory["tracer"]["path"]
                    require(process.stdout.read(1)==b"","producer_stream: trailing bytes")
                    require(process.wait()==0,"reference_physics: native producer failed; STOP for parent review")
                except BaseException:
                    raise
        manifest["trajectories"].append(trajectory)
    manifest["generation_elapsed_s"]=time.monotonic()-start
    manifest["peak_caller_rss_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    manifest["peak_child_rss_kib"]=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    write_new(output/"candidate.json",manifest)
    # A fresh process reads actual bytes, independently reconstructs native
    # mathematics, and stops publication on any scientific/integrity failure.
    checked=subprocess.run([sys.executable,str(HERE/"verify_capture.py"),str(output),"--candidate"],capture_output=True,text=True,timeout=600)
    require(checked.returncode==0,"independent_verification: "+checked.stderr.strip())
    report=strict_json(checked.stdout);write_new(output/"verification.json",report)
    manifest["verification"]={"path":"verification.json","sha256":digest((output/"verification.json").read_bytes()),"summary":{k:v for k,v in report.items() if k not in ("frame_metrics","shared_payload_groups")}}
    manifest["status"]="fixture_complete" if args.fixture_frames else "development_complete"
    write_new(output/"manifest.json",manifest)
    print(json.dumps({"status":manifest["status"],"output":str(output),"generation_elapsed_s":manifest["generation_elapsed_s"],"verification":manifest["verification"]["summary"]},sort_keys=True))

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="action",required=True)
    b=sub.add_parser("build");b.add_argument("--cargo",type=Path,required=True);b.add_argument("--prepared-source",type=Path,required=True);b.add_argument("--output",type=Path,required=True)
    c=sub.add_parser("generate");c.add_argument("--preset",required=True);c.add_argument("--split",required=True);c.add_argument("--trajectory");c.add_argument("--fixture-frames",type=int);c.add_argument("--binary",type=Path,required=True);c.add_argument("--prepared-source",type=Path,required=True);c.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    try:
        build(args) if args.action=="build" else capture(args)
    except (ValueError,OSError,subprocess.SubprocessError,KeyError) as error:
        print(f"capture_failure: {error}",file=sys.stderr);return 1
    return 0
if __name__=="__main__":sys.exit(main())
