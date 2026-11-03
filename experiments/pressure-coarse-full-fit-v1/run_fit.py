"""Admit one qualified full CUDA fit; existing owner restores on CPU on success."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'pressure-throughput-feedback-v1'),str(HERE.parent/'pressure-coarse-stability-v1')]
import run_study
from capture_policy import verify_sources


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(args):
    root=args.root.resolve(strict=True)
    def bound(relative):
        path=Path(relative)
        if path.is_absolute()or'..'in path.parts:raise ValueError('Root-relative captured source required')
        result=(root/path).resolve(strict=True)
        if not result.is_relative_to(root):raise ValueError('Captured source escaped root')
        return result
    assembly=json.loads(args.assembly.read_text())
    if assembly['limits']['attempt_slots']!=1 or assembly['updates']!=5000:
        raise ValueError('One full5000-fit attempt required')
    project_path=bound(assembly['project']['path'])
    if digest(project_path)!=assembly['project']['sha256']:raise ValueError('Project identity changed')
    project=json.loads(project_path.read_text());row=next(r for r in project['recipes']if r['id']=='train')
    recipe_path=bound(row['path'])
    if digest(recipe_path)!=row['sha256']:raise ValueError('Recipe identity changed')
    recipe=json.loads(recipe_path.read_text())
    if Path(recipe['entrypoint']).name!='full_fit.py':raise ValueError('Qualified full-fit entry required')
    arguments=recipe['arguments'];positions=[i for i,r in enumerate(arguments)if r=={'literal':'--fit-reference'}]
    if len(positions)!=1 or positions[0]+1>=len(arguments)or set(arguments[positions[0]+1])!={'source'}:
        raise ValueError('Unique captured full-fit reference required')
    if any(r=={'literal':'--gradient-limit'}for r in arguments):raise ValueError('Capped diagnostic cannot be a full fit')
    reference_path=bound(arguments[positions[0]+1]['source'])
    if reference_path.name!='gradient-reference.json':raise ValueError('Captured fit reference required')
    sources={r['path']:r['sha256']for r in recipe['sources']}
    if recipe['entrypoint']not in sources or str(reference_path.relative_to(root))not in sources:
        raise ValueError('Entry/reference absent from captured sources')
    for name,identity in sources.items():
        if digest(bound(name))!=identity:raise ValueError('Captured source changed')
    reference=json.loads(reference_path.read_text())
    if reference['format']!='pressure-coarse-full-fit-reference-v1':raise ValueError('Qualified full-fit reference required')
    verify_sources(project_path.parent,reference)
    verification=json.loads(args.capture_verification.read_text());qualification=json.loads(args.entry_qualification.read_text())
    control=json.loads(args.control_comparison.read_text())
    if not(verification['status']==qualification['status']==control['status']=='passed'
        and verification['assembly_sha256']==digest(args.assembly)
        and verification['project_manifest_sha256']==digest(project_path)
        and verification['witness_sha256']==digest(project_path.with_name('source-witness.json'))
        and verification['reference_sha256']==digest(reference_path)
        and verification['verified_prefix_updates']==5000 and verification['all_full_length_streams_exact']
        and verification['original_initial_tensors_exact']and verification['allowed_clipping_owner_change_only']
        and reference['full_fit_qualification_sha256']==digest(args.entry_qualification)
        and qualification['source_sha256']==sources[recipe['entrypoint']]
        and qualification['exact_predecessor_update_results']and qualification['exact_weights_gradients_adam_rng']
        and qualification['full_fit_finish_boundary_qualified']
        and qualification['repair_qualification_sha256']==reference['repair_qualification_sha256']
        and all(digest(project_path.with_name(name))==identity for name,identity in qualification['sources'].items())
        and control['qualified_reference_sha256']==reference['predecessor_reference_sha256']
        and all(digest(Path(path))==identity for path,identity in control['sources'].items())
        and control['campaign_package_weights_bound']
        and control['all_five_full_orders_exact']and control['inputs_normalization_runtime_exact']):
        raise ValueError('Bound full-fit/capture/control qualification required before admission')
    return assembly


class FailedRun(RuntimeError):
    def __init__(self,owner,run):
        self.owner,self.run=owner,run
        super().__init__('Managed fit/restore failed; retained terminal record, no retry')


def execute(args):
    original=run_study.await_drained
    def wait(owner,run_id,timeout=2100):
        deadline=time.monotonic()+timeout
        while True:
            run=owner.call('runs.get',dict(run_id=run_id))
            if run['state']in('completed','failed','cancelled','interrupted','unknown'):
                run_study.write(args.output/('terminal-'+run_id+'.json'),run)
                if run['custody_phase']!='drained':raise RuntimeError('Cessation unknown; inspect without resubmission')
                if run['state']!='completed':raise FailedRun(owner,run)
                return run
            if time.monotonic()>=deadline:raise TimeoutError('Run wait expired; inspect without resubmission')
            time.sleep(2)
    run_study.await_drained=wait
    try:run_study.execute(args)
    except FailedRun as failure:
        active=json.loads((args.output/'active.json').read_text())
        owner=failure.owner;state=owner.call('studies.get',dict(study_id=active['study_id']))['study']
        if state['open']:
            journal=run_study.Journal(args.output/'mutations');rows=list((args.output/'mutations').glob('*.json'))
            if not all(json.loads(path.read_text())['status']=='acknowledged'for path in rows):
                raise RuntimeError('Unknown mutation prevents automatic closure; inspect journal')from failure
            journal.index=max(int(path.name.split('-',1)[0])for path in rows)
            key='band-'+run_study.sha(dict(catalog=args.catalog_id,output=str(args.output.resolve())))[:24]
            body=dict(study_id=state['id'],request_key=key+'-stop',revision=state['revision'])
            journal.mutation('close-study','studies.stop',body,lambda:owner.call('studies.stop',body))
        final=owner.call('studies.get',dict(study_id=state['id']))['study']
        if final['open']:raise RuntimeError('Study closure unconfirmed')from failure
        run_study.write(args.output/'failed-run.json',dict(run_id=failure.run['id'],state=failure.run['state'],
            custody_phase='drained',study_id=state['id'],study_open=False,automatic_retry=False,
            cpu_restore_admitted=failure.run['id']!=active['training_run_id']))
        raise
    finally:run_study.await_drained=original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','tuldok','assembly','session-file','output','capture-verification','entry-qualification','control-comparison'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('endpoint','catalog-id','project-alias'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();args.collect_existing=False
    if args.output.exists():raise ValueError('Fresh command directory required; never replay writes')
    validate(args)
    execute(args)


if __name__=='__main__':main()
