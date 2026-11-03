"""Freeze training-only full trajectories, exact float32 transport and train scales."""
import hashlib
import math
import os
from pathlib import Path
import time
import zipfile
import numpy as np
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha
from temporal_codec import encode,decode

def run():
    output=BASE/'temporal-training-view-02'
    if output.exists():raise ValueError('Fresh temporal fitting view required')
    output.mkdir(parents=True)
    bank_path=BASE/'temporal-data-01/bank.json';bank=read(bank_path)
    if bank['status']!='completed':raise ValueError('Completed trajectory bank required')
    arrays={};parents=[];statistics={name:dict(squared=0.,elements=0)for name in ('pressure','velocity','tracer')}
    worst_quantization={name:0. for name in ('pressure','velocity','tracer')}
    for index,row in enumerate(r for r in bank['trajectories']if r['split']=='train'):
        receipt_path=ROOT/row['receipt']
        if sha(receipt_path)!=row['receipt_sha256']:raise ValueError('Source trajectory receipt changed')
        receipt=read(receipt_path);scene=receipt['scene'];fields={k:[]for k in ('pressure','vx','vy','vz','tracer')}
        for state in receipt['states']:
            path=receipt_path.parent/state['path']
            if sha(path)!=state['sha256']:raise ValueError('Source native state changed')
            with np.load(path,allow_pickle=False)as source:
                for field in fields:
                    original=source[field]
                    if hashlib.sha256(original.tobytes()).hexdigest()!=state['arrays'][field]['sha256']:
                        raise ValueError('Source full field byte hash differs')
                    value=original.astype('<f4')
                    fields[field].append(value)
                    group='velocity'if field in ('vx','vy','vz')else field
                    physical=original.astype('<f8')
                    difference=value.astype('<f8')-physical
                    if field=='pressure':physical=physical-physical.mean();difference=difference-difference.mean()
                    norm=float(np.linalg.norm(physical));error=float(np.linalg.norm(difference))
                    relative=error/max(norm,1e-30)
                    worst_quantization[group]=max(worst_quantization[group],relative)
                    statistics[group]['squared']+=float(np.dot(physical,physical));statistics[group]['elements']+=physical.size
        specs={}
        for field,values in fields.items():
            value=np.stack(values);key=f'p{index:03d}_{field}'
            encoded=encode(value);restored=decode(encoded)
            if not np.array_equal(value.view('<u4'),restored.view('<u4')):
                raise ValueError('Trajectory transport changed float32 bits')
            arrays[key]=encoded
            specs[field]=dict(key=key,shape=list(value.shape),dtype='<f4',
                full_float32_sha256=hashlib.sha256(value.tobytes()).hexdigest())
        parents.append(dict(identifier=row['identifier'],split='train',scene=scene,states=len(receipt['states']),
            receipt_source=row['receipt'],receipt_sha256=row['receipt_sha256'],fields=specs))
        print('Packed complete fitting trajectory',row['identifier'],flush=True)
    if len(parents)!=16:raise ValueError('All16 registered training trajectories required')
    scales={k:math.sqrt(v['squared']/v['elements'])for k,v in statistics.items()}
    if any(not math.isfinite(v)or v<=0 for v in scales.values()):raise ValueError('Positive train-only field scales required')
    archive=output/'trajectories.npz'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_LZMA,allowZip64=True)as target:
        for key,value in arrays.items():
            with target.open(key+'.npy','w',force_zip64=True)as member:
                np.lib.format.write_array(member,value,allow_pickle=False)
    if archive.stat().st_size>240*1024**2:
        raise ValueError('Lossless float32 training view exceeds reserved managed input budget; retain and redesign transport')
    shards=[]
    with archive.open('rb')as source:
        while block:=source.read(24*1024**2):
            p=output/f'part-{len(shards):02d}.bin';p.write_bytes(block)
            shards.append(dict(path=p.name,bytes=len(block),sha256=sha(p)))
    normalization=dict(schema='rheon-temporal-field-scales-v1',source='All states from16 whole training trajectories only',
        scales=scales,statistics=statistics,relative_denominator_rms_floor={k:v*1e-8 for k,v in scales.items()},
        degenerate_absolute_rms_tolerance={k:v*1e-8*4e-3 for k,v in scales.items()},
        pressure_gauge='Mean-free for training losses and field error; cell-zero for native state interchange',
        development_or_reserved_used=False)
    write(output/'normalization.json',normalization)
    manifest=dict(schema='rheon-temporal-training-view-v1',status='verified',parents=parents,shards=shards,
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,codec='lossless-float32-temporal-xor-byteplanes',archive_compression='ZIP_LZMA',
        original_precision='Authoritative native pressure/RHS float64 and velocities/tracer float32 retained in source captures',
        training_precision='Float32 pressure and state for fitting; reference evaluation uses original native precision',
        worst_field_quantization_relative_l2=worst_quantization,all_float32_roundtrips_bitwise=True,
        bank_sha256=sha(bank_path),normalization_sha256=sha(output/'normalization.json'),
        source_sha256=sha(__file__),codec_sha256=sha(HERE/'temporal_codec.py'),
        windows='Construct only within each declared training trajectory; all K outputs come after current state',
        development_or_reserved_used=False,optimizer_updates=0)
    write(output/'manifest.json',manifest)
    print('Verified temporal fitting view bytes',archive.stat().st_size,'parts',len(shards),flush=True)

if __name__=='__main__':run()
