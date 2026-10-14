#!/usr/bin/env python3
"""Independent capture oracle: face-incidence FV action, MAC flux, f32 correction.

Does not import producer/scenario physics. Hash identity and scientific gates are
separate failures. Array allocation is bounded and follows exact shape validation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import subprocess
import time

LAB = Path(__file__).resolve().parents[2]
RESERVATION = LAB / "docs/plans/global-pressure-correction-v1/reservation.json"
CONFIG = LAB / "experiments/global-pressure-v1/capture_config.json"
BASE = "2209012bc5f902693d0f695ec1c2d854126af317"

class CaptureError(ValueError):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"{code}: {detail}")

def require(condition, code, detail):
    if not condition:
        raise CaptureError(code, detail)


def strict_json(data):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,"duplicate_json_key",key)
            result[key]=value
        return result
    def constant(value):
        raise CaptureError("json_nonfinite",value)
    return json.loads(data,object_pairs_hook=pairs,parse_constant=constant)

def keys(value, required, optional=(), detail="object"):
    require(isinstance(value,dict) and set(required)<=set(value) and set(value)<=set(required)|set(optional),"schema_keys",detail)

def typed_number(value, detail):
    require(type(value) in (int,float) and math.isfinite(value),"schema_type",detail)

def declared_types(actual, expected, code, detail):
    """Validate the complete declared object/container/primitive shape.

    JSON integers remain integers. Declared physical f64 values may be emitted
    as JSON integers, but booleans are never numbers. Values are checked separately.
    """
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),code,detail)
        for key,value in expected.items():declared_types(actual[key],value,code,detail+"."+key)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),code,detail)
        for i,value in enumerate(expected):declared_types(actual[i],value,code,detail+f"[{i}]")
    elif type(expected) is float:
        require(type(actual) in (int,float) and math.isfinite(actual),code,detail)
    else:
        require(type(actual) is type(expected),code,detail)

def diagnostic_number(value, detail, nonnegative=False):
    require(type(value) in (int,float) and math.isfinite(value) and (not nonnegative or value>=0),"native_diagnostic_type",detail)

def diagnostics(metrics, detail):
    base={"residual_l2","residual_max","rhs_l2","relative_residual","predicted_divergence_max_s_inv","actual_divergence_max_s_inv","actual_divergence_rms_s_inv","net_flux_m3_s","courant"}
    keys(metrics,base,{"sgs"},detail)
    for key in base:
        if key=="relative_residual" and metrics[key] is None:
            require(metrics["rhs_l2"]==0,"native_diagnostic_type",detail+".relative_residual")
        else:diagnostic_number(metrics[key],detail+"."+key,nonnegative=key!="net_flux_m3_s")
    if "sgs" in metrics:
        sgs=metrics["sgs"]
        keys(sgs,{"iterations","gates","pressure_difference_l2"},detail=detail+".sgs")
        require(type(sgs["iterations"]) is int and 0<=sgs["iterations"]<=3000,"native_diagnostic_type",detail+".sgs.iterations")
        diagnostic_number(sgs["pressure_difference_l2"],detail+".sgs.pressure_difference_l2",True)
        keys(sgs["gates"],base,detail=detail+".sgs.gates")
        diagnostics(sgs["gates"],detail+".sgs.gates")

SOURCE_FILES={"probes/global-pressure-v1/Cargo.toml","probes/global-pressure-v1/Cargo.lock","probes/global-pressure-v1/build.rs","probes/global-pressure-v1/src/main.rs","experiments/global-pressure-v1/capture.py","experiments/global-pressure-v1/capture_config.json","experiments/global-pressure-v1/verify_capture.py","experiments/global-pressure-v1/prepare_dependency.py","experiments/global-pressure-v1/dependency/manifest.json","experiments/global-pressure-v1/dependency/source-files.json","experiments/global-pressure-v1/dependency/LICENSE"}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def regular(path):
    require(stat.S_ISREG(path.lstat().st_mode), "unsafe_path", str(path))
    return path

def safe_file(root, relative):
    require(isinstance(relative, str), "unsafe_path", "non-string path")
    p = Path(relative)
    require(not p.is_absolute() and p.parts and p.as_posix()==relative and all(v not in ("..", ".") for v in p.parts), "unsafe_path", relative)
    current = root
    for part in p.parts:
        current = current / part
        require(not current.is_symlink(), "unsafe_path", relative)
    regular(current)
    require(current.resolve().is_relative_to(root.resolve()), "unsafe_path", relative)
    return current

def shape(axis=None, counts=(16,16,16)):
    a = list(counts)
    if axis is not None:
        a[axis] += 1
    return a

def index(p, dims):
    return p[0] + dims[0] * (p[1] + dims[1] * p[2])

def interior_faces(counts):
    """Yield face and its adjacent cells; no native row-stencil traversal."""
    for axis in range(3):
        dims = shape(axis, counts)
        for z in range(dims[2]):
            for y in range(dims[1]):
                for x in range(dims[0]):
                    p = [x,y,z]
                    if 0 < p[axis] < counts[axis]:
                        low = p.copy(); low[axis] -= 1
                        yield axis, index(p,dims), index(low,counts), index(p,counts)

def full_action(pressure, counts, h, rho):
    # B diag(area/(rho*h)) B^T, contributions accumulated by interior face.
    result = [0.0] * math.prod(counts)
    for axis, _, low, high in interior_faces(counts):
        area = math.prod(h[d] for d in range(3) if d != axis)
        flux = area / (rho*h[axis]) * (pressure[low]-pressure[high])
        result[low] += flux; result[high] -= flux
    return result

def divergence(velocity, counts, h):
    # Direct cell differences in physical inverse seconds, independent of RHS.
    result = []
    for z in range(counts[2]):
        for y in range(counts[1]):
            for x in range(counts[0]):
                p = [x,y,z]; terms = []
                for axis in range(3):
                    high = p.copy(); high[axis] += 1
                    dims = shape(axis, counts)
                    terms.append((velocity[axis][index(high,dims)]-velocity[axis][index(p,dims)])/h[axis])
                result.append(math.fsum(terms))
    return result

def reconstruct_rhs(velocity, counts, h, dt):
    # Scattering interior integrated flux is independent of the native row RHS.
    result = [0.0] * math.prod(counts)
    for axis, face, low, high in interior_faces(counts):
        area = math.prod(h[d] for d in range(3) if d != axis)
        flux = area * velocity[axis][face]
        result[low] -= flux; result[high] += flux
    return [v/dt for v in result]

def f32_bytes(values):
    return b"".join(struct.pack("<f",value) for value in values)

def trajectory_fingerprint(geometry, initial, frames):
    # IDs, seeds, splits, paths and measured diagnostics do not identify physics.
    # Physical identity normalizes permitted JSON numeric representations,
    # including signed origin zero. Raw geometry serialization is hashed separately.
    physical=dict(geometry)
    physical["counts_xyz"]=[int(value) for value in geometry["counts_xyz"]]
    for field in ("extents_m","origin_m"):
        physical[field]=[0.0 if value==0 else float(value) for value in geometry[field]]
    physical["density_kg_m3"]=float(geometry["density_kg_m3"])
    value={"geometry":physical,"initial":{name:descriptor["sha256"] for name,descriptor in initial.items()},"frames":[{name:descriptor["sha256"] for name,descriptor in frame["arrays"].items()} for frame in frames]}
    return digest(canonical(value))

def admit_lineage(seen, fingerprint, split, trajectory_id):
    if fingerprint in seen:
        previous_id,previous_split=seen[fingerprint]
        require(previous_split==split,"cross_split_duplicate",f"{previous_id} and {trajectory_id}")
    else:seen[fingerprint]=(trajectory_id,split)

def f32(v):
    return struct.unpack("<f",struct.pack("<f",v))[0]

def corrected_velocity(pre, pressure, counts, h, rho, dt):
    result = [[0.0]*math.prod(shape(d,counts)) for d in range(3)]
    for axis, face, low, high in interior_faces(counts):
        # Explicit f64 staging then one IEEE binary32 rounding, as required.
        result[axis][face] = f32(pre[axis][face] - dt*(1.0/(rho*h[axis]))*(pressure[high]-pressure[low]))
    return result

def wall_gate(velocity, counts):
    for axis, field in enumerate(velocity):
        dims = shape(axis,counts)
        for z in range(dims[2]):
            for y in range(dims[1]):
                for x in range(dims[0]):
                    p = [x,y,z]
                    if p[axis] in (0,counts[axis]):
                        require(field[index(p,dims)] == 0.0, "boundary_flux", f"axis={axis}, face={p}")

def physics(pre, rhs, pressure, corrected, counts, h, rho, dt):
    wall_gate(pre,counts); wall_gate(corrected,counts)
    require(pressure[0] == 0.0, "pressure_gauge", "cell 0 must be zero")
    reconstructed = reconstruct_rhs(pre,counts,h,dt)
    for i,(a,b) in enumerate(zip(rhs,reconstructed)):
        require(abs(a-b) <= 2e-14*max(abs(a),abs(b),1e-3), "rhs_reconstruction", f"cell={i}, stored={a}, reconstructed={b}")
    action = full_action(pressure,counts,h,rho)
    residual = [b-a for b,a in zip(rhs,action)]
    residual_l2 = math.sqrt(math.fsum(v*v for v in residual))
    rhs_l2 = math.sqrt(math.fsum(v*v for v in rhs))
    maximum = max(map(abs,residual))
    require(residual_l2 <= max(1e-12,1e-8*rhs_l2), "reference_residual", f"l2={residual_l2}, rhs_l2={rhs_l2}")
    volume = math.prod(h)
    predicted = dt*maximum/volume
    require(predicted <= 1e-7, "predicted_divergence", str(predicted))
    expected = corrected_velocity(pre,pressure,counts,h,rho,dt)
    for d in range(3):
        require(f32_bytes(expected[d]) == f32_bytes(corrected[d]), "f32_correction", f"axis={d}")
    div = divergence(corrected,counts,h)
    actual = max(map(abs,div))
    require(actual <= 1e-5, "actual_divergence", str(actual))
    net = math.fsum(div)*volume
    sum_abs = math.fsum(map(abs,div))*volume
    require(abs(net) <= max(1e-12,1e-8*sum_abs), "global_flux", str(net))
    courant = math.fsum(dt*max(map(abs,field))/h[d] for d,field in enumerate(corrected))
    require(courant <= 0.5, "courant", str(courant))
    return {"residual_l2":residual_l2,"rhs_l2":rhs_l2,"relative_residual":residual_l2/rhs_l2 if rhs_l2 else None,"residual_max":maximum,"predicted_divergence_max_s_inv":predicted,"actual_divergence_max_s_inv":actual,"actual_divergence_rms_s_inv":math.sqrt(math.fsum(v*v for v in div)/len(div)),"net_flux_m3_s":net,"courant":courant}

def load_array(root, descriptor, expected_shape, dtype, units):
    keys(descriptor,{"shape_xyz","dtype","layout","units","bytes","path","sha256"},detail="array descriptor")
    require(type(descriptor["bytes"]) is int and all(type(n) is int for n in descriptor["shape_xyz"]),"schema_type","array dimensions/bytes")
    require(descriptor.get("shape_xyz") == expected_shape, "array_shape", str(descriptor.get("path")))
    require(descriptor.get("dtype") == dtype and descriptor.get("layout") == "x-fastest" and descriptor.get("units") == units, "array_format", str(descriptor.get("path")))
    size = math.prod(expected_shape)*(8 if dtype=="<f8" else 4)
    require(descriptor.get("bytes") == size, "array_bytes", str(descriptor.get("path")))
    path = safe_file(root,descriptor["path"])
    require(path.stat().st_size == size, "array_bytes", str(path))
    data = path.read_bytes()
    require(digest(data) == descriptor.get("sha256"), "payload_hash", str(path))
    values = [v[0] for v in struct.iter_unpack("<d" if dtype=="<f8" else "<f",data)]
    require(all(math.isfinite(v) for v in values), "array_nonfinite", str(path))
    return values

def verify(root, manifest, candidate=False, check_sources=True):
    started=time.monotonic()
    require(root.absolute()==root.resolve() and root.is_dir(),"unsafe_path","capture root must be canonical regular directory")
    def finite_tree(value):
        if type(value) in (int,float):require(math.isfinite(value),"json_nonfinite","manifest numeric value")
        elif isinstance(value,dict):
            for v in value.values():finite_tree(v)
        elif isinstance(value,list):
            for v in value:finite_tree(v)
    finite_tree(manifest)
    keys(manifest,{"schema","status","reservation_sha256","config_sha256","preset","counts_xyz","density_kg_m3","provenance","trajectories","selection","fixture"},{"generation_elapsed_s","peak_caller_rss_kib","peak_child_rss_kib","verification"},"manifest")
    require(type(manifest["fixture"]) is bool and all(type(n) is int for n in manifest["counts_xyz"]),"schema_type","fixture/counts")
    typed_number(manifest["density_kg_m3"],"density")
    if not candidate:
        require((manifest["status"]=="fixture_complete")==manifest["fixture"],"status","fixture eligibility")
    require(manifest.get("schema") == "rheon-global-pressure-v1", "schema", "capture schema")
    require(manifest.get("status") in (("candidate",) if candidate else ("development_complete","fixture_complete")), "status", "not complete")
    reservation_bytes = RESERVATION.read_bytes(); reservation = strict_json(reservation_bytes)
    require(manifest.get("reservation_sha256") == digest(reservation_bytes), "reservation_hash", "canonical reservation")
    config_bytes = CONFIG.read_bytes()
    require(manifest.get("config_sha256") == digest(config_bytes), "config_hash", "fixed capture config")
    require(manifest.get("preset") == "smoke", "development_admission", "only smoke")
    require(manifest.get("counts_xyz") == [16]*3 and manifest.get("density_kg_m3") == 1.25, "geometry", "fixed counts/density")
    native=manifest["provenance"]["native_receipt"]
    if check_sources:
        declared_types(native,{"schema_version":1,"base_commit":"","base_tree":"","patch_sha256":"","prepared_tree_sha256":"","source_manifest_sha256":"","source_files_sha256":"","license_sha256":""},"source_identity_type","native receipt")
    require(native.get("base_commit")==BASE,"source_identity","native pin")
    if check_sources:
        provenance=manifest["provenance"]
        keys(provenance,{"native_receipt","source_sha256","prepared_source","binary_path","binary_sha256","build_receipt_sha256","lab_commit"},detail="provenance")
        from prepare_dependency import load_contract, verify_prepared
        dependency, _, _, _ = load_contract()
        expected_files=SOURCE_FILES|{"experiments/global-pressure-v1/dependency/"+dependency["patch_file"]}
        require(isinstance(provenance["source_sha256"],dict),"source_inventory","source hash mapping")
        require(set(provenance["source_sha256"])==expected_files,"source_inventory","complete source closure required")
        for relative, expected in provenance["source_sha256"].items():
            require(digest(safe_file(LAB,relative).read_bytes()) == expected,"source_hash",relative)
        require(verify_prepared(Path(manifest["provenance"]["prepared_source"])) == native,"source_identity","prepared native receipt")
        require(digest(regular(Path(manifest["provenance"]["binary_path"])).read_bytes()) == manifest["provenance"]["binary_sha256"],"binary_hash","compiled producer")
        binary=Path(provenance["binary_path"])
        build_bytes=regular(binary.parent/"build-receipt.json").read_bytes()
        require(digest(build_bytes)==provenance["build_receipt_sha256"],"build_receipt_hash","actual build receipt")
        build=strict_json(build_bytes)
        require(build["status"]=="qualified_build" and build["binary_path"]==str(binary) and build["binary_sha256"]==provenance["binary_sha256"],"build_identity","binary build linkage")
        identity=strict_json(subprocess.check_output([str(binary),"--identity"],timeout=10))
        require(identity==build["identity"],"build_identity","compiled identity")
        binding=identity["binding"]
        require(binding==strict_json(regular(binary.parent/"binding.json").read_bytes()),"build_identity","persisted binding")
        require(binding["native_receipt"]==native and identity["native_receipt"]==native and binding["source_sha256"]==provenance["source_sha256"],"build_identity","native/source closure")
        require(binding["prepared_source"]==provenance["prepared_source"] and binding["resolved_native_manifest"]==str(Path(provenance["prepared_source"])/"Cargo.toml"),"build_identity","resolved source")
        require(identity["config_sha256"]==manifest["config_sha256"] and identity["reservation_sha256"]==manifest["reservation_sha256"] and identity["source_sha256"]==provenance["source_sha256"]["probes/global-pressure-v1/src/main.rs"],"build_identity","compiled inputs")
        require(digest(regular(binary.parent/"cargo-metadata.json").read_bytes())==binding["metadata_sha256"],"build_metadata_hash","actual resolved graph")
        require(identity["native_threads"]==1 and identity["parallel"] is False,"build_identity","serial native")
        if not candidate:
            keys(manifest["verification"],{"path","sha256","summary"},detail="verification report")
            report=safe_file(root,manifest["verification"]["path"])
            require(digest(report.read_bytes())==manifest["verification"]["sha256"],"verification_hash","fresh report bytes")
    entries={e["trajectory_id"]:e for e in reservation["presets"]["smoke"]["trajectories"] if e["split"] in ("train","validation")}
    trajectories=manifest["trajectories"]
    require(isinstance(trajectories,list),"schema_type","trajectory array")
    ids=[t["trajectory_id"] for t in trajectories]
    require(ids and len(ids)==len(set(ids)),"trajectory_ids","empty/duplicate")
    selection=manifest.get("selection")
    keys(selection,{"split","trajectory_id"},detail="selection")
    require(isinstance(selection,dict) and selection.get("split") in ("train","validation","development"),"development_admission","selection")
    expected_ids=[e["trajectory_id"] for e in entries.values() if selection["split"]=="development" or e["split"]==selection["split"]]
    if selection.get("trajectory_id"):
        require(selection["trajectory_id"] in expected_ids,"development_admission","selected trajectory")
        expected_ids=[selection["trajectory_id"]]
    require(ids==expected_ids,"frame_coverage","selected trajectory coverage/order")
    fixture=manifest.get("fixture",False)
    total_bytes=0; reports=[]; paths=set(); payload_hashes={};seen_trajectories={};fingerprints=[]
    for trajectory in trajectories:
        keys(trajectory,{"trajectory_id","reservation_entry","geometry","geometry_sha256","initial","frames"},detail="trajectory")
        tid=trajectory["trajectory_id"]
        require(tid in entries,"development_admission",tid)
        e=entries[tid]
        declared_types(trajectory["reservation_entry"],e,"reservation_type",tid+".reservation_entry")
        require(trajectory["reservation_entry"] == e,"trajectory_identity",tid)
        extents=reservation["geometries_extents_m"][e["geometry"]]
        geometry={"counts_xyz":[16]*3,"extents_m":extents,"origin_m":[0.0]*3,"density_kg_m3":1.25,"boundary":"sealed_zero_normal_mac","units":"SI"}
        geometry_types=dict(geometry);geometry_types["extents_m"]=[float(value) for value in extents]
        declared_types(trajectory["geometry"],geometry_types,"geometry_type",tid+".geometry")
        require(trajectory["geometry"]==geometry and trajectory["geometry_sha256"]==digest(canonical(trajectory["geometry"])),"geometry",tid)
        h=[v/16 for v in extents];dt=e["dt_s"]
        initial=trajectory["initial"]
        require(set(initial)=={"pressure","u","v","w","tracer"},"array_inventory",tid+" initial")
        initial_pressure=load_array(root,initial["pressure"],[16]*3,"<f8","Pa")
        require(all(v==0 for v in initial_pressure),"initial_state","initial pressure")
        for d,name in enumerate(("u","v","w")):
            values=load_array(root,initial[name],shape(d),"<f4","m/s")
            require(all(v==0 for v in values),"initial_state",name)
        tracer0=load_array(root,initial["tracer"],[16]*3,"<f4","dimensionless")
        width=0.08*min(extents)
        expected_tracer=[f32(0.25*math.exp(-math.fsum((((p+0.5)*h[d])-extents[d]*0.5)**2 for d,p in enumerate((x,y,z)))/(2*width*width))) for z in range(16) for y in range(16) for x in range(16)]
        require(all(abs(a-b)<=2e-7*max(abs(a),abs(b),1e-30) for a,b in zip(tracer0,expected_tracer)),"initial_tracer","centered Gaussian")
        frames=trajectory["frames"]
        if fixture:
            require(e["split"]=="train" and len(trajectories)==1 and 1<=len(frames)<=24,"development_admission","TRAIN fixture")
        else:
            require(len(frames)==e["frames"],"frame_coverage",tid)
        previous_pressure=initial["pressure"]["path"]
        previous_state={name:initial[name]["path"] for name in ("u","v","w","tracer")}
        nonzero_rhs_frames=0;nontrivial_state_frames=0
        for f,frame in enumerate(frames):
            keys(frame,{"index","time_s","forcing_time_s","previous_pressure","previous_state","arrays"},{"kind","metrics","iterations","tracer_integral_m3","elapsed_s"},"frame")
            require(type(frame["index"]) is int,"schema_type","frame index")
            typed_number(frame["time_s"],"time");typed_number(frame["forcing_time_s"],"forcing time")
            require(frame["index"]==f and frame["time_s"]==(f+1)*dt and frame["forcing_time_s"]==f*dt,"frame_time",f"{tid}:{f}")
            require(frame["previous_pressure"]==previous_pressure and frame["previous_state"]==previous_state,"history_lineage",f"{tid}:{f}")
            arrays=frame["arrays"]
            require(set(arrays)=={"rhs","pressure","pre_u","pre_v","pre_w","corrected_u","corrected_v","corrected_w","tracer"},"array_inventory",f"{tid}:{f}")
            if "metrics" in frame:diagnostics(frame["metrics"],tid+f".frame[{f}].metrics")
            if "iterations" in frame:require(type(frame["iterations"]) is int and 0<=frame["iterations"]<=3000,"native_diagnostic_type","frame iterations")
            for field in ("elapsed_s","tracer_integral_m3"):
                if field in frame:diagnostic_number(frame[field],"frame "+field,field=="elapsed_s")
            if "kind" in frame:require(type(frame["kind"]) is str and frame["kind"]=="frame","native_diagnostic_type","frame kind")
            rhs=load_array(root,arrays["rhs"],[16]*3,"<f8","m3/s2")
            pressure=load_array(root,arrays["pressure"],[16]*3,"<f8","Pa")
            pre=[load_array(root,arrays["pre_"+name],shape(d),"<f4","m/s") for d,name in enumerate(("u","v","w"))]
            corrected=[load_array(root,arrays["corrected_"+name],shape(d),"<f4","m/s") for d,name in enumerate(("u","v","w"))]
            tracer=load_array(root,arrays["tracer"],[16]*3,"<f4","dimensionless")
            metrics=physics(pre,rhs,pressure,corrected,[16]*3,h,1.25,dt)
            nonzero_rhs_frames+=any(value!=0.0 for value in rhs)
            nontrivial_state_frames+=any(value!=0.0 for field in corrected for value in field) or tracer!=tracer0
            metrics.update(trajectory_id=tid,frame=f,tracer_integral_m3=math.prod(h)*math.fsum(tracer),tracer_min=min(tracer),tracer_max=max(tracer))
            reports.append(metrics)
            previous_pressure=arrays["pressure"]["path"]
            previous_state={name:arrays["corrected_"+name]["path"] for name in ("u","v","w")};previous_state["tracer"]=arrays["tracer"]["path"]
        if not fixture:
            require(nonzero_rhs_frames>0 and nontrivial_state_frames>0,"nontrivial_history",tid+" must have nonzero RHS and nontrivial accepted state")
        fingerprint=trajectory_fingerprint(trajectory["geometry"],initial,frames)
        admit_lineage(seen_trajectories,fingerprint,e["split"],tid)
        fingerprints.append({"trajectory_id":tid,"split":e["split"],"fingerprint_sha256":fingerprint,"nonzero_rhs_frames":nonzero_rhs_frames,"nontrivial_state_frames":nontrivial_state_frames})
        for inventory in [initial]+[f["arrays"] for f in frames]:
            for desc in inventory.values():
                require(desc["path"] not in paths,"payload_path_duplicate",desc["path"]);paths.add(desc["path"]);total_bytes+=desc["bytes"]
                payload_hashes.setdefault(desc["sha256"],[]).append(desc["path"])
    require(total_bytes <= strict_json(config_bytes)["max_payload_bytes"],"resource_cap",str(total_bytes))
    observed={str(p.relative_to(root)) for p in root.rglob("*.bin")}
    require(observed==paths,"array_inventory","unlisted/missing binary payload")
    return {"status":"verified","verifier":"independent-face-incidence-v1","frames":len(reports),"trajectories":len(trajectories),"payload_bytes":total_bytes,"elapsed_s":time.monotonic()-started,"max_residual_l2":max(r["residual_l2"] for r in reports),"max_predicted_divergence_s_inv":max(r["predicted_divergence_max_s_inv"] for r in reports),"max_actual_divergence_s_inv":max(r["actual_divergence_max_s_inv"] for r in reports),"trajectory_fingerprints":fingerprints,"shared_payload_groups":[v for v in payload_hashes.values() if len(v)>1],"frame_metrics":reports}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path);parser.add_argument("--candidate",action="store_true")
    args=parser.parse_args()
    filename="candidate.json" if args.candidate else "manifest.json"
    try:
        regular(args.root/filename)
        require((args.root/filename).stat().st_size<=4*1024*1024,"manifest_size",filename)
        result=verify(args.root,strict_json((args.root/filename).read_text()),candidate=args.candidate)
        print(json.dumps(result,sort_keys=True,allow_nan=False))
    except (CaptureError,KeyError,TypeError,ValueError,OSError,subprocess.SubprocessError) as error:
        print(json.dumps({"status":"failed","error":getattr(error,"code","invalid_manifest"),"detail":str(error)}),file=sys.stderr)
        return 1
    return 0

if __name__=="__main__":
    sys.exit(main())
