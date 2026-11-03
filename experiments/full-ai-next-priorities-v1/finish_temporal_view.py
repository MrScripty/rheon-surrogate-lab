"""Independently verify the retained complete archive within the actual managed input limit."""
import hashlib
import math
import numpy as np
from study_support import ROOT,HERE,BASE,read,write,sha
from temporal_codec import decode

def run():
    output=BASE/'temporal-training-view-02';archive=output/'trajectories.npz'
    if (output/'manifest.json').exists():raise ValueError('View already finalized')
    bank_path=BASE/'temporal-data-01/bank.json';bank=read(bank_path)
    runtime_path=ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json'
    # Keep a conservative six MiB allowance for sources, metadata and CPU restore
    # inputs; the actual recipe preflight must enforce its own 256 MiB input cap.
    upper=archive.stat().st_size+runtime_path.stat().st_size+6*1024**2
    if upper>256*1024**2:raise ValueError('Actual managed input limit would be exceeded')
    statistics={k:dict(squared=0.,elements=0)for k in ('pressure','velocity','tracer')}
    worst={k:0. for k in statistics};parents=[]
    with np.load(archive,allow_pickle=False)as packed:
        for index,row in enumerate(r for r in bank['trajectories']if r['split']=='train'):
            receipt_path=ROOT/row['receipt']
            if sha(receipt_path)!=row['receipt_sha256']:raise ValueError('Source receipt changed')
            receipt=read(receipt_path)
            fields={name:decode(packed[f'p{index:03d}_{name}'])for name in ('pressure','vx','vy','vz','tracer')}
            for step,spec in enumerate(receipt['states']):
                source_path=receipt_path.parent/spec['path']
                if sha(source_path)!=spec['sha256']:raise ValueError('Original native state changed')
                with np.load(source_path,allow_pickle=False)as source:
                    for name,values in fields.items():
                        original=source[name]
                        if hashlib.sha256(original.tobytes()).hexdigest()!=spec['arrays'][name]['sha256']:
                            raise ValueError('Original full field hash changed')
                        expected=original.astype('<f4')
                        if not np.array_equal(expected.view('<u4'),values[step].view('<u4')):
                            raise ValueError('Transport changed a float32 field bit')
                        group='velocity'if name in ('vx','vy','vz')else name
                        physical=original.astype('<f8');difference=values[step].astype('<f8')-physical
                        if name=='pressure':physical-=physical.mean();difference-=difference.mean()
                        worst[group]=max(worst[group],float(np.linalg.norm(difference))/max(float(np.linalg.norm(physical)),1e-30))
                        statistics[group]['squared']+=float(np.dot(physical,physical));statistics[group]['elements']+=physical.size
            parents.append(dict(identifier=row['identifier'],split='train',scene=receipt['scene'],states=len(receipt['states']),
                receipt_source=row['receipt'],receipt_sha256=row['receipt_sha256'],fields={name:dict(key=f'p{index:03d}_{name}',
                    shape=list(value.shape),dtype='<f4',full_float32_sha256=hashlib.sha256(value.tobytes()).hexdigest())for name,value in fields.items()}))
            print('Independently verified all original state fields',row['identifier'],flush=True)
    if len(parents)!=16:raise ValueError('All16 fitting trajectories required')
    scales={k:math.sqrt(v['squared']/v['elements'])for k,v in statistics.items()}
    write(output/'normalization.json',dict(schema='rheon-temporal-field-scales-v1',scales=scales,statistics=statistics,
        source='All original full states from16 training trajectories only',
        relative_denominator_rms_floor={k:v*1e-8 for k,v in scales.items()},
        degenerate_absolute_rms_tolerance={k:v*1e-8*4e-3 for k,v in scales.items()},
        pressure_gauge='Mean-free for field loss/error; cell-zero for native interchange',development_or_reserved_used=False))
    shards=[]
    with archive.open('rb')as source:
        while block:=source.read(24*1024**2):
            path=output/f'part-{len(shards):02d}.bin'
            if path.exists():raise ValueError('Do not overwrite any retained shard')
            path.write_bytes(block);shards.append(dict(path=path.name,bytes=len(block),sha256=sha(path)))
    write(output/'manifest.json',dict(schema='rheon-temporal-training-view-v1',status='verified',parents=parents,shards=shards,
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,archive_compression='ZIP_LZMA',
        codec='lossless-float32-temporal-xor-byteplanes',all_float32_roundtrips_bitwise=True,
        training_precision='Native pressure converted to float32 for training; all authoritative native fields retained for evaluation',
        worst_field_quantization_relative_l2=worst,bank_sha256=sha(bank_path),normalization_sha256=sha(output/'normalization.json'),
        completed_archive_reused=True,original_packer_sha256=sha(HERE/'pack_temporal.py'),
        verification_source_sha256=sha(__file__),codec_sha256=sha(HERE/'temporal_codec.py'),
        managed_input_limit_bytes=256*1024**2,measured_archive_plus_runtime_plus_six_mib_reserve_bytes=upper,
        resource_note='Original240MiB packing reserve was conservative; full verification uses the actual256MiB recipe input limit with sixMiB additional allowance, before any temporal fitting.',
        development_or_reserved_used=False,optimizer_updates=0))
    print('Verified full temporal view within actual managed input budget',upper,flush=True)

if __name__=='__main__':run()
