"""Read-only prototype: cached path components preserve runtime containment.

Writes a separate candidate owner under a fresh artifact directory. Never changes
Tuldok, the deployed service, queued material, the Pumas runtime or model code.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('owner','runtime','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cpu',type=int,required=True);a=p.parse_args()
    assert a.cpu in os.sched_getaffinity(0) and not a.output.exists()
    os.sched_setaffinity(0,{a.cpu});os.nice(10);a.output.mkdir(parents=True)
    original=a.owner.read_text();predicate="any(Path(path).is_relative_to(value[root]) for root in ('python_root','site_packages'))"
    assert original.count(predicate)==2
    helper="def _manifest_contains(path, roots):\n    parts=Path(path).parts\n    return any(parts[:len(root)]==root for root in roots)\n\n\n"
    prototype=original.replace('def validate_manifest(value):',helper+'def validate_manifest(value):')
    prototype=prototype.replace("    require(Path(value['python']).is_relative_to(value['python_root'])", "    roots=tuple(Path(value[root]).parts for root in ('python_root','site_packages'))\n    require(Path(value['python']).is_relative_to(value['python_root'])",1)
    prototype=prototype.replace(predicate,'_manifest_contains(path, roots)')
    candidate=a.output/'torch_runtime.py';candidate.write_text(prototype)
    owners=[load_module('reference_runtime',a.owner.resolve()),load_module('prototype_runtime',candidate.resolve())]
    manifest=owners[0].load(a.runtime.resolve());roots=tuple(Path(manifest[root]).parts for root in ('python_root','site_packages'))
    paths=[row['path'] for row in manifest['files']+manifest['file_links']]
    paths += ['/','//','/elsewhere/file',manifest['python_root']+'-other/escape',manifest['python_root'],manifest['python_root']+'/child']
    for path in paths:
        assert any(Path(path).is_relative_to(manifest[root]) for root in ('python_root','site_packages'))==owners[1]._manifest_contains(path,roots)
    timings=[]
    for label,owner in zip(('reference','prototype'),owners):
        before=time.perf_counter();value=owner.load(a.runtime.resolve());owner.verify(value)
        timings.append(dict(owner=label,elapsed_s=time.perf_counter()-before,manifest_sha256=hashlib.sha256(owner.encoded(value)).hexdigest()))
    assert timings[0]['manifest_sha256']==timings[1]['manifest_sha256'] and 'torch' not in sys.modules
    result=dict(format='rheon_runtime_path_prototype_v1',status='passed',scope='separate unpromoted prototype; original then candidate, uncontrolled warm cache; illustrative single pair, not an isolated repeated benchmark',
        cpu_affinity=[a.cpu],containment_pairs_checked=len(paths),full_installed_asset_verification_both=True,torch_imported=False,
        source_owner_sha256=hashlib.sha256(original.encode()).hexdigest(),prototype_owner_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(a.runtime.read_bytes()).hexdigest(),timings=timings,
        deployment_or_original_sources_changed=False,model_or_runtime_files_changed=False)
    (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
