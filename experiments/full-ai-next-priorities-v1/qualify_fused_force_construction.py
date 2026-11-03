"""Require exact float32 forcing throughout the consumed development population."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from temporal_features_v2 import force_features as original
from fused_force_construction import force_features as candidate
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh forcing qualification required')
    protocol_path=DOCS/'fused-force-construction-protocol-01.json';protocol=read(protocol_path)
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered forcing source changed')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    bank_path=BASE/'temporal-development-240-01/bank.json'
    if sha(bank_path)!=protocol['bank_sha256']:raise ValueError('Registered forcing scenes changed')
    norm=read(BASE/'b2-displacement-pair-01/restore/normalization.json');scale=norm['scales']['velocity']
    rows=[];cases=0;worst=0.;exact=True;physical_values=0
    with torch.inference_mode():
        for row in read(bank_path)['trajectories']:
            if row['split']!='development':raise ValueError('Only consumed scenes required')
            path=ROOT/row['receipt']
            if sha(path)!=row['receipt_sha256']:raise ValueError('Reference scene changed')
            scene=read(path)['scene'];scene_exact=True;scene_worst=0.
            for k in (1,2,3,4):
                for step in range(241-k):
                    a=original(scene,step,k,scale);b=candidate(scene,step,k,scale)
                    same=np.array_equal(a.numpy().view('u1'),b.numpy().view('u1'))
                    scene_exact &= same;exact &= same
                    error=float((a-b).abs().max());worst=max(worst,error);scene_worst=max(scene_worst,error)
                    cases+=1;physical_values+=a.numel()
            rows.append(dict(identifier=row['identifier'],byte_exact=scene_exact,maximum_absolute_error=scene_worst))
            print('Prescribed-force execution:',row['identifier'],'byte exact:',scene_exact,'max abs:',scene_worst,flush=True)
    write(args.output,dict(status='passed'if exact else'failed',rows=rows,cases=cases,scalar_values=physical_values,
        all_float32_bytes_exact=exact,maximum_absolute_error=worst,optimizer_updates=0,
        source_sha256=protocol['source_sha256'],protocol_sha256=sha(protocol_path),CPU_latency_claim=False,reserved_scenes_consumed=False))
    if not exact:raise ValueError('Force execution does not qualify for exact model equivalence')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
