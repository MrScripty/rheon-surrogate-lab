"""Classify exact retained diagnostic observations, without inferring accuracy."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

PREFIX = '[DEBUG-coarse-gradient-v1]'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return path.read_text().splitlines()


def progress(path):
    result={}
    for line in lines(path):
        if line.startswith(PREFIX):
            continue
        try:
            value=json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get('format')=='tuldok_progress_v1' and value['stage']=='water-learning-sequence':
            assert value['total']==5000 and value['completed']not in result
            result[value['completed']]=value['metrics']['loss']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('custody','capture-verification','probe-qualification','reference','historical-stdout','historical-custody','historical-summary','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    summary=json.loads((args.custody/'summary.json').read_text())
    custody=json.loads((args.custody/'receipt.json').read_text())
    verification=json.loads(args.capture_verification.read_text())
    reference=json.loads(args.reference.read_text())
    qualification=json.loads(args.probe_qualification.read_text())
    historical_custody=json.loads(args.historical_custody.read_text())
    historical_summary=json.loads(args.historical_summary.read_text())
    assert digest(args.historical_custody)==historical_summary['raw_custody_sha256']
    assert historical_custody['attempt_id']==historical_summary['run_id']
    assert digest(args.historical_stdout)==historical_custody['streams']['stdout']['sha256']
    assert summary['status']==verification['status']=='passed'
    binding=summary['capture_binding']
    assert binding['status']=='passed'and binding['diagnostic_limit']==1500
    assert binding['assembly_sha256']==verification['assembly_sha256']
    assert binding['project_manifest_sha256']==verification['project_manifest_sha256']
    assert binding['source_witness_sha256']==verification['witness_sha256']
    assert binding['reference_sha256']==digest(args.reference)==verification['reference_sha256']
    assert binding['capture_verification_sha256']==digest(args.capture_verification)
    assert binding['qualification_sha256']==digest(args.probe_qualification)
    assert binding['observer_sha256']==qualification['source_sha256']and qualification['status']=='passed'
    assert binding['actual_spec_sha256']==custody['spec_sha256']
    assert digest(args.custody/'receipt.json')==summary['custody_receipt_sha256']
    assert custody['attempt_id']==summary['run_id'] and custody['drained']and custody['echild']and custody['spawn_closed']
    assert custody['leader_returncode']==1 and not summary['artifacts_published']and not summary['cpu_restore_admitted']
    for name in ('stdout','stderr'):
        path=args.custody/(name+'.log')
        assert path.stat().st_size==summary['verified_streams'][name]['bytes']
        assert digest(path)==summary['verified_streams'][name]['sha256']==custody['streams'][name]['sha256']
    assert digest(args.reference)==verification['reference_sha256']
    events=[json.loads(line[len(PREFIX):])for line in lines(args.custody/'stdout.log')if line.startswith(PREFIX)]
    assert len(events)<=32
    sampling=[row for row in events if row['event']=='sampling_verified']
    assert len(sampling)==5 and {row['stream']for row in sampling}==set(reference['sampling_order_sha256'])
    expected={row['stream']:row for row in verification['events']if row['event']=='sampling_verified'}
    assert all(row['full_length']==5000 and row['prefix_length']==1500
               and row['full_order_sha256']==reference['sampling_order_sha256'][row['stream']]
               and row['prefix_sha256']==expected[row['stream']]['prefix_sha256']for row in sampling)
    initial=[row for row in events if row['event']=='initial_tensors_verified']
    assert len(initial)==1 and initial[0]['parameters']==374112 and initial[0]['device'].startswith('cuda')
    failures=[row for row in events if row['event']=='clip_failure']
    caps=[row for row in events if row['event']=='diagnostic_cap_reached']
    assert len(failures)+len(caps)==1, 'Original clipping failure or explicit cap must identify this diagnostic'
    event=(failures or caps)[0]
    snapshot=event['update']['gradient_snapshot']if failures else event['final_gradient_snapshot']
    if failures:
        assert event['attempted_update']==event['update']['update']==event['completed_updates']+1
        if snapshot['all_gradient_components_finite']:
            assert isinstance(snapshot['reference_float64_total_norm'],float)
            assert math.isfinite(snapshot['reference_float64_total_norm'])
            assert snapshot['ordinary_float32_total_norm']in('inf','nan')
            classification='finite_components_norm_reduction_overflow'
        elif any(row.get('gradient_present')and not row['finite']for row in snapshot['parameters']):
            classification='nonfinite_gradient_components'
        else:
            classification='missing_gradient_observation'
    else:
        assert event['completed_updates']==1500 and event['numerical_failure']is False
        classification='cap_reached_without_gradient_guard_failure'
    historical=progress(args.historical_stdout)
    actual=progress(args.custody/'stdout.log')
    common=sorted(set(historical)&set(actual))
    comparisons=[dict(update=update,historical_loss=historical[update],diagnostic_loss=actual[update],
                      exact=historical[update]==actual[update])for update in common]
    norms={}
    for label in ('local','coarse'):
        selected=[row['reference_float64_norm']for row in snapshot['parameters']
                  if row['gradient_present']and(row['parameter'].startswith('spatial_coarse.')==(label=='coarse'))]
        norms[label]=math.hypot(*selected)if all(isinstance(value,float)for value in selected)else None
    run_seconds=(datetime.fromisoformat(summary['run_finished_at'])-datetime.fromisoformat(summary['run_started_at'])).total_seconds()
    sources={str(path):dict(sha256=digest(path),bytes=path.stat().st_size)for path in (
        args.custody/'summary.json',args.custody/'receipt.json',args.custody/'stdout.log',args.custody/'stderr.log',
        args.capture_verification,args.probe_qualification,args.reference,args.historical_stdout,args.historical_custody,args.historical_summary,Path(__file__))}
    result=dict(format='pressure-coarse-gradient-analysis-v1',status='analyzed',classification=classification,
        run_id=summary['run_id'],study_id=summary['study_id'],exact_failed_update=event['attempted_update']if failures else None,
        historical_failed_run_id=historical_summary['run_id'],
        completed_updates=event['completed_updates'],diagnostic_cap=1500,
        original_configured_updates=5000,all_five_original_streams_and_prefixes_verified=True,
        historical_progress_comparison=comparisons,all_shared_logged_losses_exact=bool(common)and all(row['exact']for row in comparisons),
        all_original_logged_updates_compared=set(historical).issubset(actual),
        failure_observation=event if failures else None,cap_observation=event if caps else None,
        gradient_group_reference_norms=norms,
        actual_run_capture_binding=binding,
        parameter_float32_norms_nonfinite=[row['parameter']for row in snapshot['parameters']
            if row['gradient_present']and isinstance(row['ordinary_float32_norm'],str)],
        run_start_through_drainage_s=run_seconds,
        instrumented_loop_before_terminal_s=event['elapsed_s']-initial[0]['elapsed_s'],
        optimizer_only_s=None,training_completed=False,final_candidate_checkpoint_published=False,
        candidate_pressure_accuracy=None,cpu_restore_admitted=False,holdouts_consumed=0,candidate_promoted=False,
        sources=sources,
        scope='One hash-verified diagnostic replay. Wider norms are diagnostic only; original float32 network/Adam/clipping decision unchanged. Instrumented loop includes observation, periodic original training diagnostics and checks; no throughput/accuracy claim.')
    args.output.mkdir(parents=True)
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if failures:
        for name,rows in [('gradients',snapshot['parameters']),('forward-contributions',event['update']['forwards'])]:
            with(args.output/(name+'.csv')).open('x',newline='')as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({key:result[key]for key in ('classification','exact_failed_update','completed_updates',
        'all_shared_logged_losses_exact','run_start_through_drainage_s','instrumented_loop_before_terminal_s')}))


if __name__=='__main__':
    main()
