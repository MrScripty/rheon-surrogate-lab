"""Losslessly compress the retained native view within aggregate managed limits."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def compact(source,output):
    if output.exists():raise ValueError('Fresh compact view required')
    os.sched_setaffinity(0,{19});manifest=read(source/'manifest.json');output.mkdir(parents=True);shards=[]
    for shard in manifest['shards']:
        original=source/shard['path']
        if sha(original)!=shard['sha256']:raise ValueError('Source view changed')
        with np.load(original,allow_pickle=False)as archive:values={k:archive[k].copy()for k in archive.files}
        target=output/shard['path'];np.savez_compressed(target,**values)
        with np.load(target,allow_pickle=False)as archive:
            for key in archive.files:
                if hashlib.sha256(archive[key].tobytes()).hexdigest()!=manifest['arrays'][key]['sha256']:
                    raise ValueError('Compression changed decoded field bytes')
        shards.append(dict(shard,bytes=target.stat().st_size,sha256=sha(target)))
        print('Compressed',target.name,round(target.stat().st_size/1024**2,2),'MiB',flush=True)
    total=sum(s['bytes']for s in shards)
    write(output/'manifest.json',dict(manifest,shards=shards,source_sha256=sha(__file__),
        lossless_compression=True,source_manifest_sha256=sha(source/'manifest.json'),encoded_total_bytes=total))
    print('Total compact inputs',round(total/1024**2,3),'MiB',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compact(args.source,args.output)
