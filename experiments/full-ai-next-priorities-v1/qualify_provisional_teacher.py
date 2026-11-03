"""Qualify offline provisional labels against independent native primitives."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from provisional_teacher import padded_velocity,pressure_contribution,reconstructed_provisional,vectors,packet,native
from temporal_features_v2 import geometry
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def flat(values):return np.concatenate([np.asarray(v,dtype='f8')for v in values])


def run(args):
    if args.output.exists():raise ValueError('Fresh teacher qualification required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    protocol=DOCS/'temporal-coupled-transport-diagnostic-protocol-01.json';reg=read(protocol)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered diagnostic source changed')
    assets=BASE/'temporal-coupled-assets-01';provenance=read(assets/'request-initialization.json')
    if sha(assets/'request-weights.npz')!=provenance['weights_sha256']:raise ValueError('Learned flux changed')
    with np.load(assets/'request-weights.npz',allow_pickle=False)as arrays:weight=torch.from_numpy(arrays['flux.weight'].copy())
    rng=np.random.Generator(np.random.PCG64(2026102011));fixtures=[];sources={}
    with torch.inference_mode():
        for counts in ([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
            nx,ny,nz=counts
            for cpm in (8,16):
                scene=dict(counts=counts,cells_per_metre=cpm)
                for pattern in ('random','zero_pressure','zero_provisional'):
                    fields=dict(pressure=rng.normal(size=nx*ny*nz),
                        vx=(.006*rng.normal(size=nz*ny*(nx+1))).astype('f4'),
                        vy=(.006*rng.normal(size=nz*(ny+1)*nx)).astype('f4'),
                        vz=(.006*rng.normal(size=(nz+1)*ny*nx)).astype('f4'))
                    if pattern=='zero_pressure':fields['pressure'].fill(0)
                    original=padded_velocity(fields,counts)
                    if pattern=='zero_provisional':original.zero_()
                    result=native([packet(original,fields,scene)])[0]
                    corrected=dict(pressure=fields['pressure'],**{name:np.asarray(v,dtype='f4')for name,v in zip(('vx','vy','vz'),result['corrected_velocity'])})
                    recovered=reconstructed_provisional(corrected,scene,weight)
                    a=flat(vectors(original,counts));b=flat(vectors(recovered,counts));out=flat(result['corrected_velocity'])
                    contribution=flat(vectors(pressure_contribution(corrected,scene,weight),counts))
                    error=float(np.linalg.norm(a-b))/max(float(np.linalg.norm(a)),float(np.linalg.norm(out)),float(np.linalg.norm(contribution)),1e-30)
                    if error>reg['native_rounding_scaled_recovery_tolerance']:raise ValueError('Native provisional inverse exceeds registered rounding tolerance')
                    fixtures.append(dict(counts_xyz=counts,cells_per_metre=cpm,pattern=pattern,rounding_scaled_recovery_error=error))
        pending=[];identities=[]
        for row in read(BASE/'temporal-data-01/bank.json')['trajectories']:
            if row['split']!='train':continue
            receipt_path=ROOT/row['receipt']
            if sha(receipt_path)!=row['receipt_sha256']:raise ValueError('Training receipt changed')
            receipt=read(receipt_path);scene=receipt['scene'];sources[str(receipt_path.relative_to(ROOT))]=sha(receipt_path)
            for start in reg['starts']:
                for horizon in (1,2):
                    spec=receipt['states'][start+horizon];path=receipt_path.parent/spec['path']
                    if sha(path)!=spec['sha256']:raise ValueError('Native target changed')
                    sources[str(path.relative_to(ROOT))]=sha(path)
                    with np.load(path,allow_pickle=False)as arrays:fields={n:arrays[n].copy()for n in ('pressure','vx','vy','vz','rhs')}
                    value=reconstructed_provisional(fields,scene,weight)
                    pending.append((packet(value,fields,scene),fields))
                    identities.append(dict(identifier=row['identifier'],start=start,horizon=horizon,counts_xyz=scene['counts']))
        rows=[]
        for start in range(0,len(pending),16):
            batch=pending[start:start+16];outputs=native([v[0]for v in batch])
            for offset,((_,fields),result)in enumerate(zip(batch,outputs)):
                truth=flat([fields[n]for n in ('vx','vy','vz')]);actual=flat(result['corrected_velocity'])
                velocity_error=float(np.linalg.norm(actual-truth))/max(float(np.linalg.norm(truth)),1e-30)
                reference=fields['rhs'];rhs=np.asarray(result['rhs'],dtype='f8')
                rhs_error=float(np.linalg.norm(rhs-reference))/max(float(np.linalg.norm(reference)),1e-30)
                if velocity_error>reg['captured_velocity_roundtrip_relative_tolerance']or rhs_error>reg['captured_RHS_relative_tolerance']:
                    write(args.output.with_suffix('.failure.json'),dict(status='failed',case=identities[start+offset],velocity_error=velocity_error,rhs_error=rhs_error))
                    raise ValueError('Captured provisional reconstruction differs from native state/RHS')
                rows.append(dict(**identities[start+offset],velocity_roundtrip_relative_error=velocity_error,native_RHS_relative_error=rhs_error))
    write(args.output,dict(status='passed',fixtures=fixtures,rows=rows,
        maximum_rounding_scaled_recovery_error=max(r['rounding_scaled_recovery_error']for r in fixtures),
        maximum_captured_velocity_roundtrip_relative_error=max(r['velocity_roundtrip_relative_error']for r in rows),
        maximum_captured_RHS_relative_error=max(r['native_RHS_relative_error']for r in rows),
        protocol_sha256=sha(protocol),source_sha256=reg['source_sha256'],source_evidence=sources,
        native_build_receipt_sha256=sha(BASE/'provisional-teacher-build-01/build-receipt.json'),
        learned_flux_sha256=sha(assets/'request-weights.npz'),new_optimizer_updates=0,pressure_solves=0,native_steps_executed=0,
        training_queries_only=True,candidate_frontier_point=False,reserved_scenes_consumed=False))
    print('Qualified provisional reconstruction:',len(fixtures),'native inverse fixtures and',len(rows),'captured training targets',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
