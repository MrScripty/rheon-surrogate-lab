"""Consume neural physical states in Rheon, with explicit event invalidation.

No future reference state or metric enters this module. Independent accuracy
evaluation happens after the measured episode. Native state restoration is a
structural research interface; it does not certify native physical acceptance.
"""
import time
import numpy as np
import torch
from bounded_native_client import BoundedNativeClient
from native_client import FIELDS
from temporal_metrics import physical_arrays


class ForecastConsumer(BoundedNativeClient):
    def advance(self,*args,**kwargs):
        raise ValueError('Numerical advance is forbidden in the forecast consumer')


def episode(network,norm,scene,initial,directory,*,steps=48,scenario='steady'):
    if scenario not in ('steady','forcing','unsupported_boundary')or not 14<=steps<=240:
        raise ValueError('Bounded registered application scenario required')
    k=network.horizons
    if k not in (1,2,3,4):raise ValueError('Registered forecast horizon required')
    event_step=None if scenario=='steady'else 13
    changed=dict(scene,steps=steps,acceleration=1.5*scene['acceleration'])
    scene=dict(scene,steps=steps);counts=tuple(scene['counts']);nx,ny,nz=counts
    sizes=dict(pressure=nx*ny*nz,vx=(nx+1)*ny*nz,vy=nx*(ny+1)*nz,vz=nx*ny*(nz+1),tracer=nx*ny*nz)
    expected_dtypes={n:np.dtype('<f8'if n=='pressure'else'<f4')for n in FIELDS}
    directory.mkdir(parents=True,exist_ok=False)
    setup_start=time.perf_counter();client=ForecastConsumer(scene,directory/'before-event')
    setup_s=time.perf_counter()-setup_start
    step=0;revision=0;current=initial;calls=0;generated=0;event_discard=0;end_discard=0
    clock=[];outputs=[];custody=[];model_s=0.;checks_s=0.;upload_s=0.;event_s=0.;final_s=0.
    status='completed';failure=None;unavailable=None;all_numerical_acks_zero=True
    started=time.perf_counter()
    try:
        # The same structural restore used for every later state also initializes
        # the actual consumer from the supplied causal state.
        tick=time.perf_counter();ack=client.restore(0,physical_arrays(initial));upload_s+=time.perf_counter()-tick
        all_numerical_acks_zero &= ack['numerical_steps_executed']==0
        with torch.inference_mode():
            while step<steps and status=='completed':
                active_scene=scene if revision==0 else changed
                tick=time.perf_counter()
                try:prediction=network.predict(current,active_scene,step,norm)
                except (ValueError,FloatingPointError)as error:
                    known={'Finite input and colocated float64 neural map required','Finite request on the component device required'}
                    if not isinstance(error,FloatingPointError)and str(error)not in known:raise
                    elapsed=time.perf_counter()-tick;model_s+=elapsed;clock.append(elapsed);calls+=1
                    status='incomplete';unavailable=step+1
                    failure=dict(kind='model_block_failed',start_step=step,requested_states=k,returned_states=0,error=str(error))
                    break
                elapsed=time.perf_counter()-tick;model_s+=elapsed;clock.append(elapsed);calls+=1;generated+=len(prediction)
                tick=time.perf_counter();validated=[]
                if len(prediction)!=k:raise ValueError('Every intermediate physical state required')
                for offset,predicted in enumerate(prediction,1):
                    values=physical_arrays(predicted)
                    if set(values)!=set(FIELDS):raise ValueError('Incomplete physical-state prediction')
                    if any(v.size!=sizes[n]or v.dtype!=expected_dtypes[n]for n,v in values.items()):
                        raise ValueError('Physical state shape/dtype differs from the qualified contract')
                    if any(not np.isfinite(v).all()for v in values.values()):
                        failure=dict(kind='nonfinite_generated_block',start_step=step,invalid_generated_step=step+offset)
                        status='incomplete';unavailable=step+1;break
                    validated.append(values)
                checks_s+=time.perf_counter()-tick
                if status!='completed':break
                block_revision=revision
                for index,(predicted,values)in enumerate(zip(prediction,validated)):
                    next_step=step+1
                    expected_revision=int(event_step is not None and next_step>event_step)
                    if block_revision!=expected_revision:raise ValueError('Stale prediction consumed after an event')
                    tick=time.perf_counter();ack=client.restore(next_step,values);upload_s+=time.perf_counter()-tick
                    all_numerical_acks_zero &= ack['numerical_steps_executed']==0
                    step=next_step;current=predicted
                    outputs.append(dict(step=step,forcing_revision=revision,fields=predicted))
                    if event_step is not None and step==event_step:
                        event_discard+=len(prediction)-index-1
                        if scenario=='unsupported_boundary':
                            status='unsupported_boundary';unavailable=step+1
                            failure=dict(kind='boundary_contract_changed',detail='The model is qualified for the closed regular domain; unused predictions invalidated before any unsupported continuation')
                            break
                        tick=time.perf_counter();custody.append(client.close())
                        client=ForecastConsumer(changed,directory/'after-event')
                        ack=client.restore(step,values);all_numerical_acks_zero &= ack['numerical_steps_executed']==0
                        revision=1;event_s+=time.perf_counter()-tick
                        break
                    if step==steps:
                        end_discard+=len(prediction)-index-1;break
        tick=time.perf_counter();packet,final=client.inspect();expected=physical_arrays(current)
        if packet['step']!=step or packet['time_s']!=step/60:raise ValueError('Native consumer clock differs from accepted neural state')
        if any(not np.array_equal(final[n].view('u1'),expected[n].view('u1'))for n in FIELDS):
            raise ValueError('Native consumer physical bytes differ from neural output')
        final_s+=time.perf_counter()-tick
        tick=time.perf_counter();custody.append(client.close());final_s+=time.perf_counter()-tick
        total_s=time.perf_counter()-started
    finally:
        if client.last_custody is None:client.close()
    complete=status=='completed'and step==steps
    if not all_numerical_acks_zero:raise ValueError('Native numerical work entered a candidate forecast episode')
    if not all(c['reaped']and c['returncode']==0 and c['graceful_protocol_complete']for c in custody):
        raise ValueError('Every native consumer must complete and reap')
    if status in ('completed','unsupported_boundary'):
        if generated!=step+event_discard+end_discard:raise ValueError('Generated/discarded physical-state accounting differs')
        if event_step is not None and event_discard!=k-1:raise ValueError('Step13must invalidate every unused pre-event state')
    result=dict(status=status,horizons=k,scenario=scenario,physical_steps=step,requested_steps=steps,
        completed_physical_interval=complete,first_unavailable_step=unavailable,failure=failure,network_calls=calls,
        generated_states=generated,event_discarded_states=event_discard,endpoint_discarded_states=end_discard,
        rejected_block_states=generated-step-event_discard-end_discard,
        full_block_seconds=clock,prediction_seconds=model_s,finite_shape_dtype_check_seconds=checks_s,
        native_state_upload_seconds=upload_s,event_reinitialization_seconds=event_s,
        native_final_verification_and_close_seconds=final_s,native_setup_seconds=setup_s,
        application_seconds=total_s,application_seconds_per_physical_second=total_s/(steps/60)if complete else None,
        application_with_native_setup_seconds=total_s+setup_s,
        model_seconds_per_physical_second=model_s/(steps/60)if complete else None,
        event_step=event_step,event_acceleration_multiplier=1.5 if scenario=='forcing'else None,
        native_processes=custody,native_final_state_bytes_exact=True,native_numerical_steps_executed=0,
        all_intermediate_states_exposed=True,unused_prediction_invalidation_verified=True,
        fallback=False,reference_inputs=False,offline_metric_seconds_in_application=0.,
        native_state_setter_is_structural_only=True,application_physical_acceptance_claim=False,
        retained_history='References retained for deferred offline metrics; no future references used by the episode')
    return result,outputs
