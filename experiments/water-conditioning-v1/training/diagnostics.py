"""Apply the unchanged measured native water oracle to fitted/validation/test predictions."""
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from policy import LAB, MIB, canonical, decode, fresh, load_module, read, require, sha, write
import data

_paths = list(sys.path)
try:
    sys.path.insert(0,str(LAB/'experiments/variable-pressure-v2'))
    checks = load_module('sequence_independent_checks',LAB/'experiments/variable-pressure-v2/evaluate.py')
finally:
    sys.path[:] = _paths
targets = load_module('sequence_float32_targets',LAB/'experiments/float32-pressure-target-v1/targets.py')


def admit(binary):
    receipt=decode(read(binary.parent/'build-receipt.json',2*MIB))
    identity=decode(subprocess.check_output([str(binary),'--identity'],timeout=10))
    binding=decode(read(binary.parent/'binding.json',2*MIB))
    require(receipt['status']=='qualified_build' and sha(read(binary))==receipt['binary_sha256'] and receipt['identity']==identity and identity['binding']==binding,'native_identity','unchanged measured evaluator')
    for path,digest in binding['source_sha256'].items():require(sha(read(LAB/path,2*MIB))==digest,'native_source',path)
    dependency=checks.build_native.dependency
    require(dependency.verify(Path(binding['prepared_source']))==binding['native_receipt'],'native_dependency','same qualified native physics')
    return receipt


def run(packet,inference,output,binary,observer=None,training_only=False):
    receipt=decode(read(inference/'receipt.json',MIB))
    if receipt['schema']=='rheon-water-learning-training-v1':
        require(receipt['mode']=='restore' and receipt['execution']['device']=='cpu' and receipt['execution']['cuda_initialized'] is False,'cpu_inference','fresh managed CPU restore')
        weights=receipt['outputs']['weights.npz']['sha256'];prediction_sha=receipt['outputs']['predictions.npz']['sha256']
    else:
        require(receipt['schema']=='rheon-water-learning-inference-v1' and receipt['runtime']['device']=='cpu' and receipt['runtime']['cuda_initialized'] is False,'cpu_inference','fresh managed CPU inference')
        weights=receipt['weights_sha256'];prediction_sha=receipt['predictions_sha256']
    require(receipt['optimizer_updates']==0,'cpu_inference','no optimizer updates')
    raw=read(inference/'predictions.npz',2*MIB);require(sha(raw)==prediction_sha,'prediction_identity','complete managed artifact')
    arrays=data.helpers().read_npz(raw,receipt['prediction_inventory'],2*MIB)
    _reservation,parents=data.read_view(packet)
    cases=[(index,case) for index,case in enumerate(receipt['cases']) if not training_only or case['split']=='train']
    require(cases,'population','nonempty declared measured cases')
    measured=admit(binary);oracle,validator=checks.historical_oracles();profile=targets.load(LAB/'experiments/float32-pressure-target-v1/configs/development.json')
    output=fresh(output);records=[];captures={};process_owner=load_module('sequence_native_process',LAB/'experiments/local-pressure-v1/process.py')
    for index,case in cases:
        parent=parents[case['parent']];root=Path(parent['capture_root'])
        if root not in captures:
            capture_raw=read(root/'manifest.json',4*MIB)
            require(sha(capture_raw)==parent['capture_manifest_sha256'],'capture_identity','measured original source')
            captures[root]=(sha(capture_raw),decode(capture_raw))
        trajectory=next(p for p in captures[root][1]['trajectories'] if p['trajectory_id']==parent['scene']['trajectory_id'])
        require(trajectory['entry']==parent['scene'] and case['native_frame']==parent['frames'][case['frame']]['index'] and case['trajectory_id']==parent['entry']['trajectory_id'],'case_identity','exact source scene and previous-pressure state')
        frame=next(f for f in trajectory['frames'] if f['index']==case['native_frame']);counts=parent['counts_xyz'];slot=case['frame']
        velocity=[np.frombuffer(read(root/frame['arrays']['pre_'+axis]['path'],MIB),dtype='<f4').astype(float).tolist() for axis in 'uvw']
        candidate=arrays[f'case_{index:03d}'].reshape(-1).tolist()
        request={'schema':'rheon-variable-pressure-teacher-case-v1','counts_xyz':counts,'spacing_m':[parent['cell_size_m']]*3,'density_kg_m3':1000.0,'dt_s':parent['entry']['dt_s'],
            'original_rhs':parent['arrays']['rhs'][slot].reshape(-1).tolist(),'reference_pressure':parent['arrays']['reference'][slot].reshape(-1).tolist(),
            'previous_pressure':parent['arrays']['base'][slot].reshape(-1).tolist(),'pre_velocity':velocity,'candidate_pressure':candidate,'candidate_error':None}
        response,process=process_owner.native_call(binary,'--evaluate',request,time.monotonic()+63,60)
        independent=checks.check_response(request,response,oracle,validator)
        record={**case,**response,'float32_acceptance':targets.assess(response['raw_learned'],profile),'independent_candidate_residual_l2':independent,'process':process}
        with (output/'cases.jsonl').open('ab') as stream:stream.write(canonical(record)+b'\n')
        records.append(record)
        if observer:
            preview=observer.frame(root,{'trajectory_id':parent['scene']['trajectory_id'],'counts_xyz':counts,'extents_m':trajectory['geometry']['extents_m'],'dt_s':parent['entry']['dt_s']},case['native_frame'],pressure=np.asarray(candidate),pressure_label='Current trial CPU pressure proposal (Pa); inspect qualification; passive tracer')
            observer.publish('Water learning · native diagnostics',f'{len(records)}/{len(cases)} {case["split"]} cases; {case["trajectory_id"]}',len(records),len(cases),preview=preview)
    require(admit(binary)==measured,'native_identity','evaluator unchanged during complete check')
    for root,(digest,_manifest) in captures.items():require(sha(read(root/'manifest.json',4*MIB))==digest,'capture_drift','each exact source capture unchanged')
    ratios=[record['raw_learned']['relative_residual'] for record in records];finite=[v for v in ratios if v is not None and math.isfinite(v)]
    result={'status':'complete_native_diagnostic','cases':len(records),'populations':sorted({c['split'] for _,c in cases}),
        'raw_native_qualified':sum(r['raw_learned']['qualified'] for r in records),'float32_qualified':sum(r['float32_acceptance']['qualified'] for r in records),
        'relative_residual':{'minimum':min(finite) if finite else None,'median':float(np.median(finite)) if finite else None,'maximum':max(finite) if finite else None},
        'methods':checks.summary(records),'weights_sha256':weights,'scope':'Full original equation and physical gates on supplied preceding native pressure; no cold-start or autonomous claim'}
    write(output/'evaluation.json',result)
    if observer:observer.publish('Water learning · diagnostics complete',f'{result["float32_qualified"]}/{len(records)} pressure proposals qualified',len(records),len(records),state='completed')
    return result
