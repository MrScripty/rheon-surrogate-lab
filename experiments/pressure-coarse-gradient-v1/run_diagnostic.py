"""One original journalled admission, terminal readback and Study closure.

Reuse the existing admission controller. Replace only its completed-fit wait
with a terminal diagnostic boundary that always prevents CPU restore/promotion.
Tuldok remains the execution, custody and scheduler owner. Never replay writes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pressure-throughput-feedback-v1'))
import run_study


class TerminalDiagnostic(Exception):
    def __init__(self, owner, run):
        self.owner, self.run = owner, run


def diagnostic_arguments(recipe):
    """Reject ordinary training before the controller can perform any mutation."""
    if Path(recipe['entrypoint']).name != 'gradient_screen.py':
        raise ValueError('Captured gradient diagnostic entrypoint required')
    arguments = recipe['arguments']
    def selected(flag, kind):
        matches = [i for i,item in enumerate(arguments) if item == {'literal':flag}]
        if len(matches)!=1 or matches[0]+1>=len(arguments) or set(arguments[matches[0]+1])!={kind}:
            raise ValueError('One explicit diagnostic argument required: '+flag)
        return arguments[matches[0]+1][kind]
    raw_limit = selected('--gradient-limit','literal')
    if not raw_limit.isdecimal() or not 1<=int(raw_limit)<=1500:
        raise ValueError('Diagnostic cap must be1..1500')
    reference = selected('--gradient-reference','source')
    if Path(reference).name!='gradient-reference.json':
        raise ValueError('Captured gradient reference source required')
    sources = {row['path']:row['sha256'] for row in recipe['sources']}
    if recipe['entrypoint'] not in sources or reference not in sources:
        raise ValueError('Diagnostic entrypoint and reference must belong to captured sources')
    return int(raw_limit), reference, sources[recipe['entrypoint']]


def validate_admission(args):
    root = args.root.resolve(strict=True)
    def bound(relative):
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Root-relative captured diagnostic source required')
        result = (root/path).resolve(strict=True)
        if not result.is_relative_to(root):
            raise ValueError('Captured diagnostic escaped artifact root')
        return result
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    assembly = json.loads(args.assembly.read_text())
    if assembly['limits']['attempt_slots']!=1:
        raise ValueError('Diagnostic admits one Run only')
    project_path = bound(assembly['project']['path'])
    if digest(project_path)!=assembly['project']['sha256']:
        raise ValueError('Diagnostic Project identity changed')
    project = json.loads(project_path.read_text())
    selected = next(row for row in project['recipes'] if row['id']=='train')
    recipe_path = bound(selected['path'])
    if digest(recipe_path)!=selected['sha256']:
        raise ValueError('Diagnostic recipe changed')
    recipe = json.loads(recipe_path.read_text())
    limit, reference, entry_sha = diagnostic_arguments(recipe)
    for row in recipe['sources']:
        if digest(bound(row['path']))!=row['sha256']:
            raise ValueError('Diagnostic captured source changed')
    verification = json.loads(args.capture_verification.read_text())
    qualification = json.loads(args.probe_qualification.read_text())
    if not (verification['status']=='passed' and verification['assembly_sha256']==digest(args.assembly)
        and verification['project_manifest_sha256']==digest(project_path)
        and verification['witness_sha256']==digest(project_path.with_name('source-witness.json'))
        and verification['reference_sha256']==digest(bound(reference))
        and verification['verified_prefix_updates']>=limit
        and verification['all_full_length_streams_exact'] and verification['original_initial_tensors_exact']
        and qualification['status']=='passed' and qualification['source_sha256']==entry_sha
        and qualification['exact_original_update_results'] and qualification['exact_weights_gradients_adam_rng']
        and qualification['actual_original_guard_observed_without_change']):
        raise ValueError('Exact capture/prefix/observation qualification required before admission')
    return limit


def execute(args):
    if args.output.exists():
        raise ValueError('Fresh diagnostic command directory required; inspect existing journals, never replay')
    started = time.perf_counter()
    limit = validate_admission(args)
    def wait_terminal(owner, run_id, timeout=2100):
        deadline = time.monotonic()+timeout
        while True:
            run = owner.call('runs.get',dict(run_id=run_id))
            if run['state'] in ('completed','failed','cancelled','interrupted','unknown'):
                run_study.write(args.output/'terminal-run.json',run)
                if run['custody_phase'] != 'drained':
                    raise RuntimeError('Diagnostic cessation unknown; inspect without resubmission')
                raise TerminalDiagnostic(owner,run)
            if time.monotonic() >= deadline:
                raise TimeoutError('Diagnostic wait expired; inspect without resubmission')
            time.sleep(2)
    original_wait = run_study.await_drained
    run_study.await_drained = wait_terminal
    try:
        run_study.execute(args)
    except TerminalDiagnostic as terminal:
        active = json.loads((args.output/'active.json').read_text())
        owner, run = terminal.owner, terminal.run
        assert run['id'] == active['training_run_id']
        state = owner.call('studies.get',dict(study_id=active['study_id']))['study']
        journal = run_study.Journal(args.output/'mutations')
        existing = list((args.output/'mutations').glob('*.json'))
        assert existing and all(json.loads(path.read_text())['status']=='acknowledged' for path in existing)
        journal.index = max(int(path.name.split('-',1)[0]) for path in existing)
        key = 'band-'+run_study.sha(dict(catalog=args.catalog_id,output=str(args.output.resolve())))[:24]
        if state['open']:
            body = dict(study_id=state['id'],request_key=key+'-stop',revision=state['revision'])
            journal.mutation('close-study','studies.stop',body,lambda:owner.call('studies.stop',body))
        final = owner.call('studies.get',dict(study_id=state['id']))['study']
        assert not final['open']
        assert not (args.output/'mutations/13-admit-restore.json').exists()
        result = dict(format='pressure-coarse-gradient-terminal-v1',status='terminal_collected',
            run_id=run['id'],run_state=run['state'],custody_phase=run['custody_phase'],
            study_id=state['id'],study_open=False,elapsed_s=time.perf_counter()-started,
            artifacts_published=len(run['artifacts']),cpu_restore_admitted=False,
            diagnostic_update_limit=limit,
            candidate_promoted=False,untouched_scenes_consumed=0,
            scientific_result='unclassified until hash-verified diagnostic logs are inspected',
            scope='One original managed admission and terminal/closed Study readback; success means diagnostic custody collected, not a completed pressure model.')
        run_study.write(args.output/'diagnostic-terminal.json',result)
        print(json.dumps(result),flush=True)
    finally:
        run_study.await_drained = original_wait


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','tuldok','assembly','session-file','output','capture-verification','probe-qualification'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('endpoint','catalog-id','project-alias'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    args.collect_existing = False
    execute(args)


if __name__ == '__main__':
    main()
