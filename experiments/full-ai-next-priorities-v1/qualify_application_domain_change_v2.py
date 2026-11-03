"""Verify fresh predictions after a supported unannounced domain boundary rescale.

Logical cell counts and uploaded concentrations/velocities are unchanged. The
physical closed box is rescaled through the already trained 8/16 cells/metre
settings. This is a functional/accuracy diagnostic, not a CPU speed comparison.
"""
import argparse
import hashlib
import os
from pathlib import Path
import signal
import numpy as np
import torch
import temporal_displacement_objective as teaching
from application_model_support import load_model
from application_model_evidence import require
from forecast_domain_change_episode import episode
from evaluate_coupled import references
from temporal_metrics import metrics,physical_arrays,FIELDS
from temporal_train_displacement import inventory
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    args.output=args.output.resolve()
    if args.output.exists():raise ValueError('Fresh supported domain-change qualification required')
    repair=read(DOCS/'application-domain-change-repair-registration-02.json')
    protocol_path=DOCS/'application-domain-change-qualification-protocol-02.json';protocol=read(protocol_path)
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered domain-change source changed')
    for relative,digest in protocol['execution_assets_sha256'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Qualified execution artifact changed')
    bank_path=BASE/'application-domain-change-reference-01/bank.json'
    if sha(bank_path)!=protocol['bank_sha256']:raise ValueError('Actual supported domain-change references changed')
    registry_path=BASE/'temporal-finalists-01/application-models.json'
    if sha(registry_path)!=protocol['registry_sha256']:raise ValueError('Original primary checkpoint selection changed')
    methods=[m for m in read(registry_path)['methods']if m['kind']=='forecast']
    if len(methods)!=4 or {m['model']['horizons']for m in methods}!={1,2,3,4}:
        raise ValueError('All four actual trained primary horizons required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1);args.output.mkdir(parents=True)
    def stop(*_):raise InterruptedError('Domain-change qualification interrupted; reaping native child')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    rows=[];sources=[protocol_path,bank_path,registry_path]
    def forbidden(*args,**kwargs):raise ValueError('Training labels reached supported-boundary inference')
    teaching.displacement_labels=forbidden
    for method in methods:
        require(method,protocol['development_bank_sha256'],protocol['scene_identifiers'])
        network,norm,identity=load_model(method['model']);before=inventory(network);k=network.horizons
        for row in read(bank_path)['trajectories']:
            scene,truth=references(row);receipt_path=ROOT/row['receipt'];receipt=read(receipt_path)
            changed=receipt['post_event_scene'];original_cpm=scene['cells_per_metre'];new_cpm=changed['cells_per_metre']
            if new_cpm!=24-original_cpm or changed['counts']!=scene['counts']:
                raise ValueError('Registered physical-box rescale at unchanged cell counts required')
            calls=[]
            class ObservedNetwork:
                horizons=k
                def predict(self,fields,active_scene,step,normalization):
                    calls.append(dict(start_step=step,cells_per_metre=active_scene['cells_per_metre']))
                    return network.predict(fields,active_scene,step,normalization)
            identifier=row['original_identifier'];directory=args.output/method['identifier']/identifier
            prior=BASE/'application-domain-change-qualification-01'/method['identifier']/identifier/'receipt.json'
            reused=prior.exists()
            if reused:
                expected=repair['retained_case']
                if prior!=ROOT/expected['path'] or sha(prior)!=expected['sha256']:
                    raise ValueError('Only the source-bound completed original case may be reused')
                result=read(prior);calls=result['calls'];diagnostics=result['diagnostics'];path=prior
                if result['model']!=identity or not result['completed_physical_interval'] or result['physical_steps']!=48:
                    raise ValueError('Retained case model/physical interval differs')
                if result['event_discarded_states']!=k-1 or result['native_numerical_steps_executed'] or result['fallback']:
                    raise ValueError('Retained case has invalidation/correction differences')
                if not result['native_final_state_bytes_exact'] or not all(c['reaped']and c['returncode']==0 and c['graceful_protocol_complete']for c in result['native_processes']):
                    raise ValueError('Retained native execution did not complete/reap exactly')
                if not any(c['start_step']==13 and c['cells_per_metre']==new_cpm for c in calls) or any(c['cells_per_metre']!=(original_cpm if c['start_step']<13 else new_cpm)for c in calls):
                    raise ValueError('Retained geometry calls are not causal and regenerated')
                if [d['step']for d in diagnostics]!=list(range(1,49)) or any(d['cells_per_metre']!=(original_cpm if d['step']<=13 else new_cpm)for d in diagnostics):
                    raise ValueError('Retained physical diagnostics/geometry incomplete')
                if result['accuracy_pass']!=all(d['metrics']['all_accuracy_criteria_pass']for d in diagnostics) or result['worst_error']!=max(v for d in diagnostics for v in d['metrics']['relative'].values()):
                    raise ValueError('Retained diagnostic summaries differ')
                if inventory(network)!=before or torch.cuda.is_initialized():raise ValueError('Reuse altered model or initialized CUDA')
            else:
                result,outputs=episode(ObservedNetwork(),norm,scene,{n:truth[0][n]for n in FIELDS},directory,steps=48)
                write(directory/'episode-execution.json',dict(result,model=identity,calls=calls,performance_claim=False))
                if not result['completed_physical_interval']or result['physical_steps']!=48:
                    raise ValueError('Supported boundary rescale must continue with fresh predictions through48')
                if result['event_discarded_states']!=k-1 or result['native_numerical_steps_executed']or result['fallback']:
                    raise ValueError('Every stale prediction must be discarded without numerical correction')
                if not any(c['start_step']==13 and c['cells_per_metre']==new_cpm for c in calls):
                    raise ValueError('Event must trigger a fresh model call on the changed geometry')
                if any(c['cells_per_metre']!=(original_cpm if c['start_step']<13 else new_cpm)for c in calls):
                    raise ValueError('Future geometry leaked before the event or stale geometry reused afterward')
                if [r['step']for r in outputs]!=list(range(1,49)):
                    raise ValueError('Every consumed physical output required')
                diagnostics=[]
                initial_sum=float(truth[0]['tracer'].astype('f8').sum())
                for output in outputs:
                    step=output['step'];active=scene if step<=13 else changed;revision=int(step>13)
                    if output['geometry_revision']!=revision:raise ValueError('Stale physical output consumed')
                    # A user box rescale changes physical cell volume. Concentration
                    # arrays are copied, so map the initial mass by that same volume;
                    # retain the original and external volume factor explicitly.
                    mass=initial_sum/active['cells_per_metre']**3
                    value=metrics(output['fields'],truth[step],truth[step]['rhs'],active,norm,mass)
                    reference_check=metrics({n:truth[step][n]for n in FIELDS},truth[step],truth[step]['rhs'],active,norm,mass)
                    if any(reference_check['absolute_rms'].values())or reference_check['relative']['equation']>1e-6:
                        raise ValueError('Piecewise native pressure/request/geometry alignment differs')
                    arrays=physical_arrays(output['fields'])
                    diagnostics.append(dict(step=step,time_s=step/60,geometry_revision=revision,
                        cells_per_metre=active['cells_per_metre'],external_volume_factor=(original_cpm/active['cells_per_metre'])**3,
                        unscaled_initial_mass=initial_sum/original_cpm**3,metrics=value,
                        predicted_field_sha256={n:hashlib.sha256(v.tobytes()).hexdigest()for n,v in arrays.items()}))
                if inventory(network)!=before or torch.cuda.is_initialized():raise ValueError('Qualification changed weights or initialized CUDA')
                path=directory/'receipt.json';write(path,dict(result,model=identity,calls=calls,diagnostics=diagnostics,
                    accuracy_pass=all(r['metrics']['all_accuracy_criteria_pass']for r in diagnostics),
                    worst_error=max(v for r in diagnostics for v in r['metrics']['relative'].values()),
                    fresh_prediction_after_supported_boundary_change=True,future_reference_inputs=False,
                    performance_claim=False,original_primary_reselected=False,optimizer_updates=0))
            sources.extend([path,receipt_path])
            rows.append(dict(model=method['identifier'],identifier=identifier,horizons=k,physical_steps=48,
                cells_per_metre_before=original_cpm,cells_per_metre_after=new_cpm,
                event_discarded_states=result['event_discarded_states'],endpoint_discarded_states=result['endpoint_discarded_states'],
                fresh_prediction_at_step13=True,geometry_arguments_causal=True,
                native_numerical_steps_executed=0,native_processes=result['native_processes'],
                accuracy_pass=all(r['metrics']['all_accuracy_criteria_pass']for r in diagnostics),
                receipt=str(path.relative_to(ROOT)),receipt_sha256=sha(path),reused_completed_original_episode=reused))
            write(args.output/'progress.json',dict(status='running',rows=rows))
            print('Qualified supported boundary rescale:',method['identifier'],identifier,'discarded',k-1,'and regenerated after13',flush=True)
    if len(rows)!=32 or sum(r['reused_completed_original_episode']for r in rows)!=1:raise ValueError('Exactly32cases, one retained and31new, required')
    sources.extend([DOCS/'application-domain-change-repair-registration-02.json',ROOT/repair['original_failure']['execution_path'],ROOT/repair['original_failure']['log_path']])
    write(args.output/'receipt.json',dict(status='passed',rows=rows,source_sha256=protocol['source_sha256'],
        source_evidence={str(p.relative_to(ROOT)):sha(p)for p in sources},protocol_sha256=sha(protocol_path),
        actual_primary_horizons=[1,2,3,4],all_supported_boundary_changes_trigger_fresh_predictions=True,
        completed_original_episodes_reused=1,new_episodes_executed=31,original_failed_owner_preserved=True,
        unchanged_closed_boundary_type=True,physical_domain_size_changed=True,logical_cell_counts_unchanged=True,
        application_acceptance_limits=False,performance_claim=False,optimizer_updates=0,reserved_scenes_consumed=False))
    print('Qualified32actual supported domain-change episodes; original research errors remain diagnostics',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:run(args)
    except Exception as error:
        path=args.output/'receipt.json'
        if not path.exists():write(path,dict(status='failed',error=type(error).__name__+': '+str(error),source_sha256=sha(__file__),optimizer_updates=0,performance_claim=False))
        raise
