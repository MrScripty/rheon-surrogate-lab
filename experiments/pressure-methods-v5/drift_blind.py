"""Compare independently advanced full native fields at matched accepted times."""
import argparse
import hashlib
from pathlib import Path
import math
from v5_support import ROOT,BASE,DOCS,read,write,sha

def metrics(reference,candidate,pressure=False):
    import numpy as np
    ref=reference.astype('float64');actual=candidate.astype('float64')
    if pressure:ref-=ref.mean();actual-=actual.mean()
    error=actual-ref;denominator=float(np.linalg.norm(ref));numerator=float(np.linalg.norm(error))
    return dict(relative_l2=numerator/denominator if denominator else 0. if not numerator else None,
        absolute_l2=numerator,reference_l2=denominator,max_absolute=float(np.abs(error).max()))

def states(directory,row):
    import numpy as np
    path=directory/row['path']
    if sha(path)!=row['sha256']:raise ValueError('Full-state artifact changed')
    result={}
    with np.load(path,allow_pickle=False)as archive:
        if set(archive.files)!=set(row['arrays']):raise ValueError('State field inventory changed')
        for key,spec in row['arrays'].items():
            value=archive[key]
            if value.dtype.str!=spec['dtype']or list(value.shape)!=spec['shape']or hashlib.sha256(value.tobytes()).hexdigest()!=spec['sha256']:
                raise ValueError('Full native field bytes changed')
            result[key]=value
    return result

def compare(trajectory,policy,output):
    import numpy as np
    if output.exists():raise ValueError('Fresh drift result required')
    refdir=BASE/'rollouts-blind'/(trajectory+'-spectral');aidir=BASE/'rollouts-blind'/(trajectory+'-'+policy)
    reference=read(refdir/'receipt.json');candidate=read(aidir/'receipt.json')
    if reference['scene']!=candidate['scene']:raise ValueError('Drift requires matching initial conditions and forcing')
    if not all(d['native_process']['reaped']for d in [reference,candidate]):raise ValueError('Both trajectories must be terminal and reaped')
    rows=[];reference_states={r['index']:r for r in reference['states']}
    for row in candidate['states']:
        i=row['index']
        if i not in reference_states:break
        if not reference['frames'][i]['advanced']or not candidate['frames'][i]['advanced']:break
        r=states(refdir,reference_states[i]);a=states(aidir,row)
        velocity_ref=np.concatenate([r[k]for k in ['vx','vy','vz']]);velocity_ai=np.concatenate([a[k]for k in ['vx','vy','vz']])
        rows.append(dict(index=i,time_s=(i+1)/60,pressure=metrics(r['pressure'],a['pressure'],True),
            velocity=metrics(velocity_ref,velocity_ai),tracer=metrics(r['tracer'],a['tracer']),
            reference_quality=reference['frames'][i]['quality'],candidate_quality=candidate['frames'][i]['quality']))
    crossings={}
    for field in ['pressure','velocity','tracer']:
        crossings[field]=[]
        for threshold in [1e-6,1e-5,1e-4,1e-3,1e-2]:
            first=next((r for r in rows if r[field]['relative_l2']is not None and r[field]['relative_l2']>threshold),None)
            crossings[field].append(dict(relative_l2_threshold=threshold,first_crossing_step=first['index']+1 if first else None,
                first_crossing_time_s=first['time_s']if first else None))
    write(output,dict(status='completed',trajectory=trajectory,policy=policy,rows=rows,first_crossings=crossings,
        requested_steps=reference['scene']['steps'],matched_accepted_steps=len(rows),reference_status=reference['status'],candidate_status=candidate['status'],
        reference_receipt_sha256=sha(refdir/'receipt.json'),candidate_receipt_sha256=sha(aidir/'receipt.json'),source_sha256=sha(__file__),
        no_fallback=True,diagnostic_only=True,application_acceptance_claimed=False,
        interpretation='Trajectory differences between independently advanced tracks; pressure gauges removed. Rejected steps excluded. Relative errors undefined for nonzero error against zero norm.'))
    print('Measured full-field drift',trajectory,len(rows),'accepted steps',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--trajectory',required=True);parser.add_argument('--policy',default='learned');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compare(args.trajectory,args.policy,args.output)
