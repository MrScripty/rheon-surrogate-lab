"""Freeze the fixed primary, control and completed calibrated seeds before capture."""
import argparse
from pathlib import Path
import time
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    args.output=args.output.resolve()
    if args.output.exists():raise ValueError('Fresh frozen selection required')
    folder=DOCS/'pressure-confirmation-v2'
    decision=read(folder/'primary-model-decision.json')
    if decision['primary_model']!='a-tail8':raise ValueError('Fixed primary changed')
    if read(BASE/'tail8-replication-batch-01/execution.json')['status']!='completed':
        raise ValueError('All registered continuation fits/restores/evaluations must complete')
    qualifications={
        'C107':BASE/'precise-pressure-family-qualification-01/C107-v2/qualification.json',
        'a-tail8':BASE/'precise-routed-pressure-qualification-01/a-tail8.json',
        'a-tail8-s2':BASE/'precise-pressure-family-qualification-01/a-tail8-s2/qualification.json',
        'a-tail8-s3':BASE/'precise-pressure-family-qualification-01/a-tail8-s3/qualification.json'}
    models=[]
    for identifier in ('C107','a-tail8','a-tail8-s2','a-tail8-s3'):
        entry_path=ROOT/'runs/pressure-methods-v4/entries/v4-student50-control.json'if identifier=='C107'else BASE/'entries'/(identifier+'.json')
        entry=read(entry_path);qualification=read(qualifications[identifier])
        if qualification['status']!='passed'or qualification['tested_requests']!=2176:
            raise ValueError('Every candidate/control must pass full numerical qualification')
        expected=HERE/('precise_routed_baseline_pressure.py'if identifier=='C107'else'precise_routed_tail_pressure.py')
        if qualification['execution_sha256']!=sha(expected)or qualification['materialization_sha256']!=sha(HERE/'materialized_fine_precise.py'):
            raise ValueError('Qualified learned execution changed')
        if qualification['source_entry']!=entry:raise ValueError('Qualified checkpoint entry changed')
        if not all(row['pressure_bitwise_equal']for row in qualification['rows']if max(row['counts_xyz'])>16):
            raise ValueError('Larger-grid parent retention failed')
        if identifier!='C107':
            pair_path=BASE/f'{identifier}-pair-01/receipt.json';pair=read(pair_path)
            if pair['status']!='completed'or pair['optimizer_updates']!=10000 or not all(pair['artifact_parity'].values()):
                raise ValueError('Exact independent CPU restore required')
            if entry['originating_pair_sha256']!=sha(pair_path):raise ValueError('Restored entry pair changed')
            if any(row['compiled_equation_error']>7e-4 for row in qualification['rows']):
                raise ValueError('Consumed pressure target must pass before independent capture')
        models.append(dict(id=identifier,entry_path=str(entry_path.relative_to(ROOT)),entry_sha256=sha(entry_path),
            role='baseline'if identifier=='C107'else'primary'if identifier=='a-tail8'else'continuation_replica',
            qualification_path=str(qualifications[identifier].relative_to(ROOT)),qualification_sha256=sha(qualifications[identifier])))
    files=['pressure_family_execution.py','precise_routed_baseline_pressure.py','precise_routed_tail_pressure.py',
        'materialized_fine.py','materialized_fine_precise.py','whole_pressure_trace.py','tail_whole_pressure.py',
        'capture_precise_pressure_confirmation.py','capture_pressure_fresh.py','evaluate_precise_pressure_confirmation.py',
        'benchmark_precise_pressure_confirmation.py','fresh_pressure_data.py','native_client.py','study_support.py',
        'freeze_precise_pressure_finalists.py']
    write(args.output,dict(status='frozen_before_reserved_capture',frozen_unix=time.time(),models=models,
        primary_model='a-tail8',primary_decision_sha256=sha(folder/'primary-model-decision.json'),
        primary_selection='Original successful tail seed remains primary; no post-confirmation reselection',
        scene_reservation_sha256=sha(folder/'pressure-scene-reservation.json'),
        source_sha256={name:sha(HERE/name)for name in files},
        weights='Verified managed CUDA fits and exact CPU restores; zero evaluation updates',
        execution='Same precise learned fine policy for C107 and tail models; dense<=2048cells, compiled CNN above, eagerabove16',
        numerical_tolerance=dict(pressure_relative=1e-6,equation_absolute=1e-7,equation_relative=1e-4),
        targets=dict(median_CPU_ms=4.,worst_equation_error=7e-4),
        populations='768small+25632cubedpreparedrequests; fullnative64stepsattemptedoneveryscene; nofallback',
        timing='Five balanced serial repeats, steps1/16/32/64eachscene; fullvalidation andprediction; coldsetupseparate',
        candidate_numerical_pressure_solve=False,model_promotion=False,reserved_temporal_scenes_consumed=False))
    print('Frozen original primary, three exact restored tail seeds and consistently optimized C107 before new capture',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
