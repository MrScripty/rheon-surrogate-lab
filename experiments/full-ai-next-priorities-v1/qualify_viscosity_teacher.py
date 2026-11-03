"""Check public native viscosity transactions, units and exact dyadic labels."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from prepare_flux_teacher import fields
from viscosity_teacher_support import native,from_carrier
from study_support import HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh viscosity teacher qualification required')
    protocol_path=DOCS/'displacement-component-diagnostic-protocol-01.json';protocol=read(protocol_path)
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered diagnostic source changed')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    rng=np.random.Generator(np.random.PCG64(protocol['fixture_seed']));cases=[]
    for counts in protocol['counts_xyz']:
        for cpm in (8,16):
            value,_=fields(rng,counts,1)
            quantized=(value*16).round().clamp(-16,16)/16
            quantized=torch.where(quantized==0,torch.zeros_like(quantized),quantized)
            for pattern,current in (('zero',torch.zeros_like(value)),('random',.01*value),('quantized',quantized)):
                for mu in (0.,.001):
                    cases.append((from_carrier(current,counts,cpm,mu=mu),dict(counts_xyz=counts,cpm=cpm,pattern=pattern,mu=mu,dyadic=False)))
            cases.append((from_carrier(quantized,counts,cpm,density=1.,mu=1.,dt=(1/cpm)**2/64),
                dict(counts_xyz=counts,cpm=cpm,pattern='quantized',mu=1.,dyadic=True)))
    rows=[]
    for start in range(0,len(cases),2):
        selected=cases[start:start+2];outputs=native([c[0]for c in selected])
        for (case,identity),result in zip(selected,outputs):
            actual=[np.asarray(v,dtype='f4')for v in result['updated_velocity']]
            original=[np.asarray(v,dtype='f4')for v in case['velocity']]
            if any(not np.isfinite(v).all()for v in actual):raise ValueError('Native viscosity output nonfinite')
            unchanged=all(np.array_equal(a.view('u4'),b.view('u4'))for a,b in zip(actual,original))
            if (case['dynamic_viscosity']==0 or identity['pattern']=='zero')and not unchanged:
                raise ValueError('Zero input/coefficient must preserve stored fields exactly')
            if result['kinetic_after']>result['kinetic_before']+1e-12*max(result['kinetic_before'],1e-30):
                raise ValueError('Native viscosity increases kinetic energy')
            lattice_error=None
            if identity['dyadic']:
                target=np.concatenate([(a.astype('f8')-b.astype('f8'))/result['beta']for a,b in zip(actual,original)])
                lattice_error=float(np.max(np.abs(target*16-np.rint(target*16))))
                if lattice_error!=0:raise ValueError('Dyadic label is not exact at the registered lattice')
            rows.append(dict(**identity,zero_or_mu0_bitwise_preserved=unchanged if case['dynamic_viscosity']==0 or identity['pattern']=='zero'else None,
                kinetic_before=result['kinetic_before'],kinetic_after=result['kinetic_after'],dyadic_label_lattice_error=lattice_error))
    write(args.output,dict(status='passed',rows=rows,source_sha256=protocol['source_sha256'],protocol_sha256=sha(protocol_path),
        native_build_receipt_sha256=sha(BASE/'viscosity-teacher-build-01/build-receipt.json'),optimizer_updates=0,
        native_pressure_solves=0,native_steps_executed=0,candidate_inference_component=False,CPU_latency_claim=False,reserved_scenes_consumed=False))
    print('Qualified native viscosity primitive and dyadic label path:',len(rows),'cases; zero optimizer updates',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
