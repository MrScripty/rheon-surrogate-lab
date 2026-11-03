"""Read-only CPU profile of exact installed-runtime verification; no Torch import."""
import argparse
import cProfile
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','owner','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cpu',type=int,required=True);a=p.parse_args()
    assert not a.output.exists() and a.cpu in os.sched_getaffinity(0)
    os.sched_setaffinity(0,{a.cpu});os.nice(10)
    spec=importlib.util.spec_from_file_location('measured_runtime_owner',a.owner.resolve(strict=True))
    owner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=owner;spec.loader.exec_module(owner)
    before=time.perf_counter();profile=cProfile.Profile();profile.enable()
    runtime=owner.load(a.runtime.resolve(strict=True));owner.verify(runtime)
    profile.disable();elapsed=time.perf_counter()-before
    stats=__import__('pstats').Stats(profile).stats
    rows=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],total_calls=v[1],self_seconds=v[2],cumulative_seconds=v[3]) for k,v in stats.items()]
    result=dict(format='rheon_runtime_verification_profile_v1',status='passed',elapsed_s=elapsed,cpu_affinity=[a.cpu],
        owner_sha256=hashlib.sha256(a.owner.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(a.runtime.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch_imported='torch' in sys.modules,
        top_cumulative=sorted(rows,key=lambda r:r['cumulative_seconds'],reverse=True)[:35],
        top_self=sorted(rows,key=lambda r:r['self_seconds'],reverse=True)[:35],
        scope='cProfile overhead included; one warm/uncontrolled-cache CPU verification while managed CUDA work/admission may run; no runtime files modified or optimization performed')
    assert not result['torch_imported']
    with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'elapsed_s':elapsed,'top_self':result['top_self'][:7]}))

if __name__=='__main__':main()
