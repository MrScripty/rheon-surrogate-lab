"""Lossless CPU data packaging only; no Torch/model import or inference.

Three groups retain all29active training scenes and all9native current states.
Dense teacher/features must later be regenerated and checked against the full
CUDA reference view, never read secretly outside managed captured inputs.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import time
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FIELDS=('pressure','vx','vy','vz','tracer')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(args):
    if args.output.exists():raise ValueError('Fresh lossless source snapshot package required')
    reg=read(args.protocol)
    for p,h in reg['source_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen snapshot packing dependency changed: '+p)
    bank=read(ROOT/reg['bank']);cases=[c for c in bank['trajectories'] if c['reference_substeps']==8]
    if bank['split']!='train' or len(bank['trajectories'])!=148 or len(cases)!=29:raise ValueError('Original29active scenes within148training bank required')
    os.sched_setaffinity(0,{reg['host_cpu']});args.output.mkdir(parents=True);began=time.monotonic()
    result=dict(status='running',active=None,segments=[],source_evidence={},cpu_model_forward=False,model_forward=False,
        optimizer_updates=0,coefficient_fitting=False,development_or_reserved_used=False,expanded_reserved52_references_opened=0,
        tensor_precision='OriginalP64andV/T32bytes preserved; no quantization.',protocol_sha256=sha(args.protocol),source_sha256=reg['source_sha256'])
    def save():write(args.output/'progress.json',result)
    def bounded():
        if time.monotonic()-began>reg['maximum_wall_seconds']:raise TimeoutError('Finite lossless packing deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>reg['maximum_peak_rss_bytes']:raise ValueError('Snapshot packing host memory cap')
        if sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())>reg['maximum_output_bytes']:raise ValueError('Snapshot packing disk cap')
    try:
        for segment,starts in enumerate(reg['start_groups']):
            path=args.output/f'snapshots-segment{segment}.zip';members=[]
            with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as archive:
                for case in cases:
                    receipt=ROOT/case['receipt']
                    if sha(receipt)!=case['receipt_sha256']:raise ValueError('Training receipt changed')
                    scene=read(receipt)['scene'];specs={s['step']:s for s in case['states']}
                    archive.writestr('scenes/'+case['identifier']+'.json',json.dumps(scene,sort_keys=True,allow_nan=False).encode())
                    for start in starts:
                        result['active']=dict(segment=segment,identifier=case['identifier'],start=start)
                        spec=specs[start];source=ROOT/spec['path']
                        if sha(source)!=spec['sha256']:raise ValueError('Original current training state changed')
                        result['source_evidence'][spec['path']]=spec['sha256']
                        with np.load(source,allow_pickle=False) as a:arrays={n:a[n].copy() for n in FIELDS}
                        if any(a.dtype.str!=('<f8' if n=='pressure' else '<f4') or not np.isfinite(a).all() for n,a in arrays.items()):raise ValueError('Exact native current tensor dtype/values required')
                        buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);blob=buffer.getvalue()
                        member=f"states/{case['identifier']}-start{start:03d}.npz";archive.writestr(member,blob)
                        members.append(dict(identifier=case['identifier'],start=start,member=member,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),
                            original_current_path=spec['path'],original_current_sha256=spec['sha256'],counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],
                            fields={n:dict(dtype=a.dtype.str,shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest()) for n,a in arrays.items()}))
                        bounded();save()
            with zipfile.ZipFile(path) as archive:
                for row in members:
                    blob=archive.read(row['member'])
                    if hashlib.sha256(blob).hexdigest()!=row['sha256']:raise ValueError('Snapshot ZIPmember changed')
                    with np.load(io.BytesIO(blob),allow_pickle=False) as a:
                        for n,spec in row['fields'].items():
                            if a[n].dtype.str!=spec['dtype'] or list(a[n].shape)!=spec['shape'] or hashlib.sha256(a[n].tobytes()).hexdigest()!=spec['sha256']:
                                raise ValueError('Snapshot roundtrip changed original physical tensor bytes')
            if len(members)!=87 or len({r['identifier'] for r in members})!=29:raise ValueError('All29scenes x3native current starts required persegment')
            chunks=[];index=0
            with path.open('rb') as source:
                while blob:=source.read(reg['maximum_chunk_bytes']):
                    chunk=args.output/f'segment{segment}-part{index:02d}.bin';chunk.write_bytes(blob)
                    chunks.append(dict(path=chunk.name,bytes=len(blob),sha256=sha(chunk)));index+=1
            aggregate=hashlib.sha256()
            for chunk in chunks:
                with (args.output/chunk['path']).open('rb') as source:
                    while blob:=source.read(1024*1024):aggregate.update(blob)
            if aggregate.hexdigest()!=sha(path):raise ValueError('Captured chunk reconstruction changed ZIPbytes')
            if path.stat().st_size+reg['reserved_asset_and_metadata_bytes']>reg['managed_total_input_bytes']:
                raise ValueError('Source snapshot segment plus asset reserve exceeds managed input bound')
            result['segments'].append(dict(segment=segment,starts=starts,archive=dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path)),chunks=chunks,
                members=members,all_current_tensor_bytes_exact=True,managed_input_asset_reserve_bytes=reg['reserved_asset_and_metadata_bytes']))
            bounded();save();print('Lossless native snapshot segment',segment,'all29scenes/87snapshots',path.stat().st_size,'bytes',flush=True)
        if len({(r['identifier'],r['start']) for s in result['segments'] for r in s['members']})!=261:raise ValueError('All29scenes/9native snapshots must be preserved')
        for p,h in reg['source_sha256'].items():
            if sha(ROOT/p)!=h:raise ValueError('Frozen packing dependency changed during execution')
        result.update(status='verified_all261_lossless_native_current_snapshots_in_three_managed_size_groups',active=None,
            total_current_snapshots=261,all_nine_current_starts_preserved=True,managed_teacher_regeneration_verified=False,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_packaging_seconds=time.monotonic()-began,
            limitation='Packaging only. Captured frozen CUDA teacher assets/known controls and exact783example regeneration still required before managed training. Dense11GBview is a reference, not an uncaptured input read inside a fit. No training/fit/promotion.')
        write(args.output/'manifest.json',result)
    except BaseException as e:
        result.update(status='failed',error=repr(e));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
