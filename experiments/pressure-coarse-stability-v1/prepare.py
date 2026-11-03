"""Prepare an explicit repair overlay without replacing historical identities."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from capture_policy import repaired_trainer


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-project','qualification','training-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    reference=json.loads((args.baseline_project/'gradient-reference.json').read_text())
    qualification=json.loads(args.qualification.read_text())
    assert reference['format']=='pressure-coarse-gradient-reference-v1'
    assert qualification['status']=='passed'and qualification['exact_weights_gradients_adam_rng']
    assert qualification['integrated_observer_and_cuda_cap_qualified']
    assert qualification['baseline_trainer_sha256']==reference['original_sources']['train.py']==digest(args.baseline_project/'train.py')
    assert (args.training_source/'train.py').read_bytes()==repaired_trainer((args.baseline_project/'train.py').read_bytes())
    assert qualification['trainer_sha256']==digest(args.training_source/'train.py')
    assert qualification['clipping_sha256']==digest(args.training_source/'gradient_clipping.py')
    args.output.mkdir(parents=True)
    for name in ('stability_screen.py','capture_policy.py'):
        shutil.copyfile(Path(__file__).with_name(name),args.output/name)
    shutil.copyfile(args.baseline_project/'gradient_screen.py',args.output/'gradient_screen.py')
    shutil.copyfile(args.baseline_project/'train.py',args.output/'baseline_train.py')
    reference['format']='pressure-coarse-clipping-reference-v1'
    reference['baseline_reference_sha256']=digest(args.baseline_project/'gradient-reference.json')
    reference['repair_qualification_sha256']=digest(args.qualification)
    reference['repair_sources']={name:digest(args.output/name)for name in ('stability_screen.py','capture_policy.py','gradient_screen.py','baseline_train.py')}
    reference['repair_sources'].update({name:digest(args.training_source/name)for name in ('train.py','gradient_clipping.py')})
    assert qualification['source_sha256']==reference['repair_sources']['stability_screen.py']
    assert qualification['observer_sha256']==reference['repair_sources']['gradient_screen.py']
    reference['scope']='Original full5000 configuration/streams/initial tensors retained; exact clipping owner replacement only. This1500 cap is incomplete training, never a deployable final model.'
    (args.output/'gradient-reference.json').write_text(json.dumps(reference,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status='passed',reference_sha256=digest(args.output/'gradient-reference.json'),optimizer_updates=0)))


if __name__=='__main__':main()
