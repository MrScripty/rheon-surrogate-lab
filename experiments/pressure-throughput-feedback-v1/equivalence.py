"""Reuse this run's complete CPU control only after exact inference equivalence.

Metadata identity may differ. Every executable/numerical byte, physical contract
and fresh first-frame input/output must match. Historical receipts never change.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_bytes())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure_package(path):
    manifest=read(path);root=path.parent
    assert {p.name for p in root.iterdir() if p.is_file()}==set(manifest['files'])|{'manifest.json'}
    for name,expected in manifest['files'].items():
        file=root/name;assert file.is_file() and not file.is_symlink()
        assert digest(file)==expected['sha256'] and file.stat().st_size==expected['bytes']
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','campaign','fixture','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    reference=measure_package(a.reference);candidate=measure_package(a.candidate)
    def numerical(value):
        return {**{k:v for k,v in value.items() if k not in ('id','version','files')},
            'files':{k:v for k,v in value['files'].items() if k!='provenance.json'}}
    assert numerical(reference)==numerical(candidate),'numerical/package contract differs; run full independent regression'
    campaign=read(a.campaign/'receipt.json');assert campaign['status']=='passed' and len(campaign['records'])==74
    assert campaign['manifest_sha256']==digest(a.reference)
    fixture=read(a.fixture/'episode.json');assert fixture['status']=='passed' and fixture['fixture'] is True
    assert fixture['manifest_sha256']==digest(a.candidate) and fixture['binary_sha256']==campaign['binary_sha256']
    previous=a.campaign/'case-00-scene-00/state-0016.npz';fresh=a.fixture/'state-0016.npz'
    with np.load(previous,allow_pickle=False) as left,np.load(fresh,allow_pickle=False) as right:
        assert set(left.files)==set(right.files)
        for name in left.files:
            assert left[name].shape==right[name].shape and left[name].dtype==right[name].dtype
            assert left[name].tobytes()==right[name].tobytes(),'fresh first-frame array differs: '+name
    result=dict(format='rheon_pressure_exact_control_equivalence_v1',status='passed',reference_manifest_sha256=digest(a.reference),candidate_manifest_sha256=digest(a.candidate),
        source_campaign_sha256=digest(a.campaign/'receipt.json'),fresh_fixture_sha256=digest(a.fixture/'episode.json'),
        weights_sha256=candidate['files']['weights.npz']['sha256'],numerical_files_exact=sorted(numerical(candidate)['files']),
        metadata_only_differences=['id','version','files.provenance.json'],fresh_first_frame_all_arrays_exact=True,
        reused_qualified_cases=74,additional_full_trajectories_executed=0,additional_first_frame_fixture=True,
        scope='Same numerical executable bytes, weights, runtime, physical and stopping contracts; reuse this turn\'s complete74 control after a fresh current-package fixture. No accuracy or timing improvement attributed to metadata.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
