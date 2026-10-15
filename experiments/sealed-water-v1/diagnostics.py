"""Independent native acceptance of the frozen candidate's reserved CPU predictions."""
import math
from pathlib import Path
import numpy as np
from policy import LAB,MIB,canonical,decode,fresh,load_module,read,require,sha,write
import data

native=load_module('water_existing_native_diagnostics',LAB/'experiments/variable-pressure-v2/build_native.py')
native.HERE=LAB/'experiments/variable-pressure-v2'
checks=load_module('water_existing_diagnostic_checks',LAB/'experiments/variable-pressure-v2/evaluate.py')
targets=load_module('water_float32_targets',LAB/'experiments/float32-pressure-target-v1/targets.py')


def run(packet,inference,output,binary,observer=None):
    receipt=decode(read(inference/'receipt.json',MIB));require(receipt['schema']=='rheon-sealed-water-reserved-inference-v1'
        and receipt['optimizer_updates']==0 and receipt['runtime']['device']=='cpu' and receipt['runtime']['cuda_initialized'] is False,'inference','actual final reserved CPU inference')
    raw=read(inference/'predictions.npz',2*MIB);require(sha(raw)==receipt['predictions_sha256'],'predictions','retained complete artifact')
    arrays=data.helpers().read_npz(raw,receipt['prediction_inventory'],2*MIB)
    reservation,parents=data.load(read(packet/'dataset.npz',32*MIB),decode(read(packet/'splits.json',4*MIB)))
    require(all(p['entry']['split']=='reserved_test' for p in parents),'population','test only')
    expected=[(i,s) for i,p in enumerate(parents) for s in range(len(p['frames']))]
    require([(c['parent'],c['frame']) for c in receipt['cases']]==expected,'case_population','every reserved snapshot exactly once')
    measured=native.admit(binary);oracle,validator=checks.historical_oracles();profile=targets.load(LAB/'experiments/float32-pressure-target-v1/configs/development.json')
    output=fresh(output);records=[];capture_cache={}
    for index,case in enumerate(receipt['cases']):
        parent=parents[case['parent']];root=Path(parent['capture_root']);source=root/'manifest.json'
        if root not in capture_cache:
            capture_raw=read(source,4*MIB);require(sha(capture_raw)==parent['capture_manifest_sha256'],'capture_identity','same qualified corpus')
            capture_cache[root]=decode(capture_raw)
        trajectory=next(p for p in capture_cache[root]['trajectories'] if p['trajectory_id']==case['trajectory_id'])
        require(trajectory['entry']==parent['scene'] and case['native_frame']==parent['frames'][case['frame']]['index'],'case_identity','actual scene and native time')
        frame=next(f for f in trajectory['frames'] if f['index']==case['native_frame']);counts=parent['counts_xyz']
        velocity=[np.frombuffer(read(root/frame['arrays']['pre_'+axis]['path'],MIB),dtype='<f4').astype(float).tolist() for axis in 'uvw']
        candidate=arrays[f'case_{index:03d}'].reshape(-1).tolist();slot=case['frame']
        request={'schema':'rheon-variable-pressure-teacher-case-v1','counts_xyz':counts,'spacing_m':[parent['cell_size_m']]*3,
            'density_kg_m3':parent['density_kg_m3'],'dt_s':parent['entry']['dt_s'],
            'original_rhs':parent['arrays']['rhs'][slot].reshape(-1).tolist(),
            'reference_pressure':parent['arrays']['reference'][slot].reshape(-1).tolist(),
            'previous_pressure':parent['arrays']['base'][slot].reshape(-1).tolist(),'pre_velocity':velocity,
            'candidate_pressure':candidate,'candidate_error':None}
        response,process=native.call(binary,'--evaluate',request);independent=checks.check_response(request,response,oracle,validator)
        record={**case,'force':parent['scene']['force'],**response,'float32_acceptance':targets.assess(response['raw_learned'],profile),
            'independent_candidate_residual_l2':independent,'process':process}
        with (output/'cases.jsonl').open('ab') as stream:stream.write(canonical(record)+b'\n')
        records.append(record)
        if observer:
            preview=observer.frame(root,{'trajectory_id':case['trajectory_id'],'counts_xyz':counts,
                'extents_m':trajectory['geometry']['extents_m'],'dt_s':parent['entry']['dt_s']},case['native_frame'],
                pressure=np.asarray(candidate),pressure_label='Frozen water model pressure proposal (Pa); inspect native acceptance; passive tracer')
            observer.publish('Reserved water pressure diagnostics',f'{index+1}/{len(expected)} snapshots; {case["trajectory_id"]}',index+1,len(expected),preview=preview)
    require(native.admit(binary)==measured,'native_identity','unchanged qualified diagnostic binary')
    for root in capture_cache:require(sha(read(root/'manifest.json',4*MIB))==parents[0]['capture_manifest_sha256'],'capture_drift','unchanged corpus')
    ratios=[r['raw_learned']['relative_residual'] for r in records];finite=[v for v in ratios if v is not None and math.isfinite(v)]
    result={'status':'complete_reserved_test_diagnostic','parents':len(parents),'cases':len(records),
        'raw_native_qualified':sum(r['raw_learned']['qualified'] for r in records),
        'float32_qualified':sum(r['float32_acceptance']['qualified'] for r in records),
        'relative_residual':{'minimum':min(finite) if finite else None,'median':float(np.median(finite)) if finite else None,'maximum':max(finite) if finite else None},
        'methods':checks.summary(records),'weights_sha256':receipt['weights_sha256'],
        'scope':'Independent reserved teacher states; no autonomous rollout or acceleration claim'}
    write(output/'evaluation.json',result)
    if observer:observer.publish('Water model training and test complete',f'{len(records)} reserved snapshots; learned pressure passes {result["float32_qualified"]}/{len(records)}',len(records),len(records),state='completed')
    return result
