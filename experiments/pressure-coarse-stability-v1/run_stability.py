"""One qualified clipping screen through the existing journalled owner lifecycle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pressure-coarse-gradient-v1'))
import run_diagnostic as driver
from capture_policy import verify_sources


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_admission(args):
    root=args.root.resolve(strict=True)
    def bound(relative):
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Root-relative captured source required')
        result=(root/path).resolve(strict=True)
        if not result.is_relative_to(root):raise ValueError('Source escaped artifact root')
        return result
    assembly=json.loads(args.assembly.read_text())
    if assembly['limits']['attempt_slots']!=1:raise ValueError('One screen Run only')
    project_path=bound(assembly['project']['path'])
    if digest(project_path)!=assembly['project']['sha256']:raise ValueError('Project identity changed')
    project=json.loads(project_path.read_text())
    row=next(r for r in project['recipes']if r['id']=='train')
    recipe_path=bound(row['path'])
    if digest(recipe_path)!=row['sha256']:raise ValueError('Recipe identity changed')
    recipe=json.loads(recipe_path.read_text())
    if Path(recipe['entrypoint']).name!='stability_screen.py':raise ValueError('Qualified clipping stability entry required')
    arguments=recipe['arguments']
    def selected(flag,kind):
        positions=[i for i,item in enumerate(arguments)if item=={'literal':flag}]
        if len(positions)!=1 or positions[0]+1>=len(arguments)or set(arguments[positions[0]+1])!={kind}:
            raise ValueError('Unique explicit screen argument required: '+flag)
        return arguments[positions[0]+1][kind]
    if selected('--gradient-limit','literal')!='1500':raise ValueError('Exact1500-update screen required')
    reference_path=bound(selected('--gradient-reference','source'))
    if reference_path.name!='gradient-reference.json':raise ValueError('Captured repair reference required')
    sources={r['path']:r['sha256']for r in recipe['sources']}
    if recipe['entrypoint']not in sources or str(reference_path.relative_to(root))not in sources:
        raise ValueError('Captured screen/reference required')
    for name,identity in sources.items():
        if digest(bound(name))!=identity:raise ValueError('Captured source changed')
    reference=json.loads(reference_path.read_text())
    if reference['format']!='pressure-coarse-clipping-reference-v1':raise ValueError('Qualified repair reference required')
    verify_sources(project_path.parent,reference)
    verification=json.loads(args.capture_verification.read_text())
    qualification=json.loads(args.probe_qualification.read_text())
    if not(verification['status']==qualification['status']=='passed'
        and verification['assembly_sha256']==digest(args.assembly)
        and verification['project_manifest_sha256']==digest(project_path)
        and verification['witness_sha256']==digest(project_path.with_name('source-witness.json'))
        and verification['reference_sha256']==digest(reference_path)
        and verification['verified_prefix_updates']==1500
        and verification['all_full_length_streams_exact']and verification['original_initial_tensors_exact']
        and verification['allowed_clipping_owner_change_only']
        and reference['repair_qualification_sha256']==digest(args.probe_qualification)
        and qualification['source_sha256']==sources[recipe['entrypoint']]
        and qualification['trainer_sha256']==digest(project_path.with_name('train.py'))
        and qualification['clipping_sha256']==digest(project_path.with_name('gradient_clipping.py'))
        and qualification['observer_sha256']==digest(project_path.with_name('gradient_screen.py'))
        and qualification['exact_original_update_results']and qualification['exact_weights_gradients_adam_rng']
        and qualification['integrated_observer_and_cuda_cap_qualified']
        and qualification['repaired_fixture_optimizer_steps']==1
        and qualification['unrelated_error_identity_preserved']):
        raise ValueError('Bound integrated repair/capture qualification required before admission')
    return 1500


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','tuldok','assembly','session-file','output','capture-verification','probe-qualification'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('endpoint','catalog-id','project-alias'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();args.collect_existing=False
    original=driver.validate_admission
    driver.validate_admission=validate_admission
    try:driver.execute(args)
    finally:driver.validate_admission=original


if __name__=='__main__':main()
