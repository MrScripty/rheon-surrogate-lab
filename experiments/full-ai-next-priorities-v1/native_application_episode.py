"""Native numerical/C107 controls for the same application intervals and events."""
import time
import numpy as np
from bounded_native_client import BoundedNativeClient
from temporal_metrics import physical_arrays


def episode(method,predict,scene,initial,directory,*,steps=240,scenario='steady'):
    if method not in ('native-pcg','spectral','C107')or scenario not in ('steady','forcing')or not 14<=steps<=240:
        raise ValueError('Bounded declared native control scenario required')
    if (method=='C107')!=(predict is not None):raise ValueError('Real C107 predictor required only for its control')
    event_step=None if scenario=='steady'else 13;scene=dict(scene,steps=steps)
    changed=dict(scene,acceleration=1.5*scene['acceleration']);directory.mkdir(parents=True,exist_ok=False)
    tick=time.perf_counter();client=BoundedNativeClient(scene,directory/'before-event');setup_s=time.perf_counter()-tick
    step=0;revision=0;frames=[];attempt_s=[];outputs=[];custody=[];failed=None;event_s=0.;offline_export_s=0.;pressure_s=0.
    started=time.perf_counter()
    try:
        client.restore(0,physical_arrays(initial))
        while step<steps:
            tick=time.perf_counter();frame=client.advance(method=method,predict=predict)
            attempt_s.append(time.perf_counter()-tick);frames.append(frame)
            if client.last_request is not None:pressure_s+=client.last_request['predict_seconds']
            if not frame['frame']['advanced']:
                failed=dict(attempted_step=step+1,native_frame=frame);break
            step+=1
            if event_step is not None and step==event_step:
                tick=time.perf_counter();packet,fields=client.inspect()
                if packet['step']!=step:raise ValueError('Event native control clock differs')
                custody.append(client.close());client=BoundedNativeClient(changed,directory/'after-event');client.restore(step,fields)
                event_s+=time.perf_counter()-tick;revision=1
                outputs.append(dict(step=step,forcing_revision=0,fields=fields))
            elif step in (12,48,240)and step!=steps:
                tick=time.perf_counter();packet,fields=client.inspect()
                if packet['step']!=step:raise ValueError('Native control validation clock differs')
                outputs.append(dict(step=step,forcing_revision=revision,fields=fields));offline_export_s+=time.perf_counter()-tick
        tick=time.perf_counter();packet,final=client.inspect()
        if packet['step']!=step or packet['time_s']!=step/60:raise ValueError('Native final control clock differs')
        if step>0 and not any(r['step']==step for r in outputs):outputs.append(dict(step=step,forcing_revision=revision,fields=final))
        custody.append(client.close());final_s=time.perf_counter()-tick
        total_s=time.perf_counter()-started-offline_export_s
    finally:
        if client.last_custody is None:client.close()
    if not all(c['reaped']and c['returncode']==0 and c['graceful_protocol_complete']for c in custody):
        raise ValueError('Every native control process must complete and reap')
    complete=failed is None and step==steps
    return dict(status='completed'if complete else'native_rejected',method=method,scenario=scenario,horizons=1,
        physical_steps=step,requested_steps=steps,completed_physical_interval=complete,first_unavailable_step=None if complete else step+1,
        native_failure=failed,frames=frames,per_attempt_seconds=attempt_s,pressure_prediction_or_reference_seconds=pressure_s,
        native_setup_seconds=setup_s,event_reinitialization_seconds=event_s,native_final_verification_and_close_seconds=final_s,
        offline_state_export_seconds=offline_export_s,application_seconds=total_s,
        application_seconds_per_physical_second=total_s/(steps/60)if complete else None,
        application_with_native_setup_seconds=total_s+setup_s,event_step=event_step,
        event_acceleration_multiplier=1.5 if scenario=='forcing'else None,event_discarded_states=0,endpoint_discarded_states=0,
        native_processes=custody,native_acceptance_gates_unchanged=True,fallback=False,
        timing_scope='Actual native advance, pressure exchange/solver, physical gates, event restart and final readback/close; pure offline validation exports excluded',
        role='Numerical/native-pressure control, never a full-state neural inference component'),outputs
