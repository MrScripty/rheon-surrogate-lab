"""Freeze full-fit observer beside source-identical qualified clipping owners."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('qualified-project','entry-qualification','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    qualification=json.loads(args.entry_qualification.read_text())
    reference=json.loads((args.qualified_project/'gradient-reference.json').read_text())
    assert qualification['status']=='passed'and qualification['full_fit_finish_boundary_qualified']
    assert qualification['exact_predecessor_update_results']and qualification['exact_weights_gradients_adam_rng']
    assert qualification['repair_qualification_sha256']==reference['repair_qualification_sha256']
    for name,identity in qualification['sources'].items():assert digest(args.qualified_project/name)==identity
    args.output.mkdir(parents=True)
    for name in ('gradient_screen.py','stability_screen.py','capture_policy.py','baseline_train.py'):
        shutil.copyfile(args.qualified_project/name,args.output/name)
    shutil.copyfile(Path(__file__).with_name('full_fit.py'),args.output/'full_fit.py')
    assert digest(args.output/'full_fit.py')==qualification['source_sha256']
    reference['format']='pressure-coarse-full-fit-reference-v1'
    reference['predecessor_reference_sha256']=digest(args.qualified_project/'gradient-reference.json')
    reference['full_fit_qualification_sha256']=digest(args.entry_qualification)
    reference['repair_sources']['full_fit.py']=qualification['source_sha256']
    reference['scope']='Full original5000 configuration and streams; qualified clipping and observation unchanged. Existing evidence owner publishes final checkpoints only on completed training; CPU physics remains independent.'
    (args.output/'gradient-reference.json').write_text(json.dumps(reference,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status='passed',reference_sha256=digest(args.output/'gradient-reference.json'),optimizer_updates=0)))


if __name__=='__main__':main()
