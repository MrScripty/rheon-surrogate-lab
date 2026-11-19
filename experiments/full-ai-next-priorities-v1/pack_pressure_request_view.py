"""Training-only causal pressure-request data; future RHS is a target, never input."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import zipfile
import numpy as np
from study_support import ROOT,HERE,BASE,read,write,sha
from temporal_codec import encode,decode


def run(args):
    if args.output.exists():raise ValueError('Fresh causal request training view required')
    args.output.mkdir(parents=True);os.sched_setaffinity(0,{23})
    bank_path=BASE/'temporal-data-01/bank.json';bank=read(bank_path)
    original=BASE/'temporal-training-view-02'
    manifest=read(original/'manifest.json');archive=original/'trajectories.npz'
    if sha(archive)!=manifest['archive_sha256']:raise ValueError('Original training transport changed')
    parents=[];squared=0.;elements=0;worst=0.
    target=args.output/'trajectories.npz'
    with np.load(archive,allow_pickle=False)as old,zipfile.ZipFile(target,'w',compression=zipfile.ZIP_LZMA,allowZip64=True)as packed:
        selected=[row for row in bank['trajectories']if row['split']=='train']
        if len(selected)!=16:raise ValueError('Exactly16 whole training scenes required')
        for index,row in enumerate(selected):
            receipt_path=ROOT/row['receipt']
            if sha(receipt_path)!=row['receipt_sha256']:raise ValueError('Native training receipt changed')
            receipt=read(receipt_path);scene=receipt['scene']
            source_parent=manifest['parents'][index]
            if source_parent['identifier']!=row['identifier']:raise ValueError('Training parent order changed')
            fields={}
            for name in ('vx','vy','vz'):
                spec=source_parent['fields'][name];encoded=old[spec['key']]
                restored=decode(encoded)
                if hashlib.sha256(restored.tobytes()).hexdigest()!=spec['full_float32_sha256']:
                    raise ValueError('Current velocity transport differs')
                fields[name]=dict(spec)
                with packed.open(spec['key']+'.npy','w',force_zip64=True)as member:
                    np.lib.format.write_array(member,encoded,allow_pickle=False)
            values=[]
            for state in receipt['states']:
                path=receipt_path.parent/state['path']
                if sha(path)!=state['sha256']:raise ValueError('Original state changed')
                with np.load(path,allow_pickle=False)as source:
                    rhs=source['rhs']
                    if hashlib.sha256(rhs.tobytes()).hexdigest()!=state['arrays']['rhs']['sha256']:
                        raise ValueError('Native RHS field bytes differ')
                    physical=rhs.astype('<f8')*(1000.*scene['cells_per_metre'])
                stored=physical.astype('<f4')
                error=float(np.linalg.norm(stored.astype('<f8')-physical))/max(float(np.linalg.norm(physical)),1e-30)
                worst=max(worst,error)
                squared+=float(np.dot(physical,physical));elements+=physical.size
                values.append(stored)
            value=np.stack(values);encoded=encode(value)
            if not np.array_equal(value.view('<u4'),decode(encoded).view('<u4')):
                raise ValueError('Pressure-request float32 transport changed bits')
            key=f'p{index:03d}_q'
            with packed.open(key+'.npy','w',force_zip64=True)as member:
                np.lib.format.write_array(member,encoded,allow_pickle=False)
            fields['q']=dict(key=key,shape=list(value.shape),dtype='<f4',
                full_float32_sha256=hashlib.sha256(value.tobytes()).hexdigest())
            parents.append(dict(identifier=row['identifier'],split='train',scene=scene,states=len(values),
                fields=fields,receipt_source=row['receipt'],receipt_sha256=row['receipt_sha256']))
            write(args.output/'packing-progress.json',dict(status='running',completed_parents=len(parents),
                required_parents=16,development_or_reserved_used=False))
            print('Packed causal request training parent',row['identifier'],flush=True)
    if worst>1e-6:raise ValueError('Request target quantization exceeds predeclared tolerance')
    # The smaller velocity/request-only view must fit the existing managed limit.
    if target.stat().st_size+6*1024**2>256*1024**2:
        raise ValueError('Full training view exceeds managed limit; retain before redesign')
    velocity=read(original/'normalization.json')['scales']['velocity']
    normalization=dict(schema='rheon-pressure-request-scales-v1',scales=dict(velocity=velocity,q=math.sqrt(squared/elements)),
        statistics=dict(q=dict(squared=squared,elements=elements)),source='All129states from16 training scenes only',
        development_or_reserved_used=False,velocity_scale='Exact unchanged original temporal training scale')
    write(args.output/'normalization.json',normalization)
    shards=[]
    with target.open('rb')as source:
        while block:=source.read(24*1024**2):
            path=args.output/f'part-{len(shards):02d}.bin';path.write_bytes(block)
            shards.append(dict(path=path.name,bytes=len(block),sha256=sha(path)))
    write(args.output/'manifest.json',dict(schema='rheon-causal-pressure-request-training-view-v1',status='verified',
        parents=parents,shards=shards,archive_sha256=sha(target),archive_bytes=target.stat().st_size,
        codec='lossless-float32-temporal-xor-byteplanes',archive_compression='ZIP_LZMA',
        bank_sha256=sha(bank_path),original_velocity_view_sha256=sha(original/'manifest.json'),
        normalization_sha256=sha(args.output/'normalization.json'),source_sha256=sha(__file__),
        codec_sha256=sha(HERE/'temporal_codec.py'),worst_q_quantization_relative_l2=worst,
        q_definition='q=(rho/h)*native_rhs; rho1000; native RHS at each exported completed step',
        causal_inputs='Current staggered velocity only, geometry, physical scalars and prescribed known forcing',
        offline_targets='Future q; never part of candidate inference inputs',
        current_pressure_input=False,reference_pressure_input=False,development_or_reserved_used=False,
        temporal_candidate=False,optimizer_updates=0,
        objective_scope='Component fitting diagnostic before full-state endpoint/reuse models; does not satisfy temporal objectives'))
    write(args.output/'packing-progress.json',dict(status='completed',completed_parents=16,required_parents=16,
        development_or_reserved_used=False,archive_bytes=target.stat().st_size))
    print('Verified causal pressure request view',target.stat().st_size,'bytes',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
