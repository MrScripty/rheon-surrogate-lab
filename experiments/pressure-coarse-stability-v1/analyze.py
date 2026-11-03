"""Classify the hash-verified capped replay; never infer final pressure accuracy."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pressure-coarse-gradient-v1'))
from analyze import progress, PREFIX


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('custody','capture-verification','probe-qualification','reference','baseline-custody','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    summary=json.loads((args.custody/'summary.json').read_text())
    receipt=json.loads((args.custody/'receipt.json').read_text())
    baseline=json.loads((args.baseline_custody/'summary.json').read_text())
    baseline_receipt=json.loads((args.baseline_custody/'receipt.json').read_text())
    verification=json.loads(args.capture_verification.read_text())
    qualification=json.loads(args.probe_qualification.read_text())
    reference=json.loads(args.reference.read_text())
    binding=summary['capture_binding']
    assert summary['status']==baseline['status']==verification['status']==qualification['status']=='passed'
    assert binding['assembly_sha256']==verification['assembly_sha256']
    assert binding['project_manifest_sha256']==verification['project_manifest_sha256']
    assert binding['source_witness_sha256']==verification['witness_sha256']
    assert binding['reference_sha256']==verification['reference_sha256']==digest(args.reference)
    assert binding['capture_verification_sha256']==digest(args.capture_verification)
    assert binding['qualification_sha256']==reference['repair_qualification_sha256']==digest(args.probe_qualification)
    assert binding['observer_sha256']==qualification['source_sha256']
    assert receipt['attempt_id']==summary['run_id']and receipt['spec_sha256']==binding['actual_spec_sha256']
    assert receipt['drained']and receipt['echild']and receipt['spawn_closed']and receipt['leader_returncode']==1
    assert not summary['study_open']and not summary['artifacts_published']and not summary['cpu_restore_admitted']
    for directory,record,custody in ((args.custody,summary,receipt),(args.baseline_custody,baseline,baseline_receipt)):
        assert digest(directory/'receipt.json')==record['custody_receipt_sha256']
        assert custody['attempt_id']==record['run_id']
        for name in ('stdout','stderr'):
            path=directory/(name+'.log')
            assert path.stat().st_size==record['verified_streams'][name]['bytes']
            assert digest(path)==record['verified_streams'][name]['sha256']==custody['streams'][name]['sha256']
    events=[json.loads(line[len(PREFIX):])for line in(args.custody/'stdout.log').read_text().splitlines()if line.startswith(PREFIX)]
    assert len(events)<1600
    sampling=[r for r in events if r['event']=='sampling_verified']
    expected={r['stream']:r for r in verification['events']if r['event']=='sampling_verified'}
    assert len(sampling)==5 and {r['stream']for r in sampling}==set(reference['sampling_order_sha256'])
    assert all(r['full_length']==5000 and r['prefix_length']==1500
        and r['full_order_sha256']==reference['sampling_order_sha256'][r['stream']]
        and r['prefix_sha256']==expected[r['stream']]['prefix_sha256']for r in sampling)
    initial=[r for r in events if r['event']=='initial_tensors_verified']
    assert len(initial)==1 and initial[0]['device'].startswith('cuda')
    repairs=[r for r in events if r['event']=='clip_repaired']
    assert [r['repair_index']for r in repairs]==list(range(1,len(repairs)+1))
    assert all(r['norms']['ordinary_float32_norm']in('inf','nan')and math.isfinite(r['norms']['repaired_float32_norm'])for r in repairs)
    caps=[r for r in events if r['event']=='diagnostic_cap_reached']
    failed=[r for r in events if r['event']=='update_failed']
    assert len(caps)+len(failed)==1
    terminal=(caps or failed)[0]
    if caps:
        assert terminal['completed_updates']==terminal['successful_cuda_checks']==1500
        assert terminal['clipping_repairs']==len(repairs)and not terminal['numerical_failure']
        classification='1500_updates_finite_with_clipping_repair'
    else:classification='numerical_or_execution_failure_before_cap'
    previous=progress(args.baseline_custody/'stdout.log');current=progress(args.custody/'stdout.log')
    comparisons=[dict(update=n,baseline_loss=previous[n],repaired_loss=current[n],exact=previous[n]==current[n])for n in sorted(previous.keys()&current.keys())]
    baseline_first=[json.loads(line[len(PREFIX):])for line in(args.baseline_custody/'stdout.log').read_text().splitlines()if line.startswith(PREFIX)]
    baseline_failure=next(r for r in baseline_first if r['event']=='clip_failure')
    first_repair_matches=bool(repairs)and repairs[0]['attempted_update']==baseline_failure['attempted_update']
    run_s=(datetime.fromisoformat(summary['run_finished_at'])-datetime.fromisoformat(summary['run_started_at'])).total_seconds()
    sources={str(path):dict(sha256=digest(path),bytes=path.stat().st_size)for path in (
        args.custody/'summary.json',args.custody/'receipt.json',args.custody/'stdout.log',args.custody/'stderr.log',
        args.baseline_custody/'summary.json',args.baseline_custody/'receipt.json',args.baseline_custody/'stdout.log',
        args.baseline_custody/'stderr.log',args.capture_verification,args.probe_qualification,args.reference,Path(__file__))}
    result=dict(format='pressure-coarse-clipping-analysis-v1',status='analyzed',classification=classification,
        run_id=summary['run_id'],study_id=summary['study_id'],completed_updates=terminal['completed_updates'],
        successful_cuda_checks=terminal.get('successful_cuda_checks'),clipping_repair_count=len(repairs),
        first_repair_update=repairs[0]['attempted_update']if repairs else None,
        first_repair_matches_diagnostic_failure=first_repair_matches,repairs=repairs,terminal_observation=terminal,
        baseline_diagnostic_run_id=baseline['run_id'],historical_progress_comparison=comparisons,
        all_baseline_logged_losses_exact=bool(comparisons)and all(r['exact']for r in comparisons),
        all_baseline_logged_updates_compared=set(previous).issubset(current),logged_losses=current,
        actual_capture_binding=binding,all_original_streams_and_initial_tensors_verified=True,
        run_start_through_drainage_s=run_s,instrumented_loop_s=terminal['elapsed_s']-initial[0]['elapsed_s'],
        optimizer_only_s=None,training_completed=False,candidate_pressure_accuracy=None,
        final_candidate_checkpoint_published=False,cpu_restore_admitted=False,holdouts_consumed=0,candidate_promoted=False,
        analysis_elapsed_s=time.perf_counter()-started,sources=sources,
        scope='One capped1500 CUDA stability replay preserving original5000 configuration and streams. Cap is intentional nonzero diagnostic cessation, not numerical failure. Finite optimizer state does not establish pressure accuracy or full5000 stability; instrumented timing includes probes and periodic diagnostics.')
    args.output.mkdir(parents=True)
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k]for k in ('classification','completed_updates','successful_cuda_checks','clipping_repair_count','first_repair_update','all_baseline_logged_losses_exact','run_start_through_drainage_s','instrumented_loop_s')}))


if __name__=='__main__':main()
