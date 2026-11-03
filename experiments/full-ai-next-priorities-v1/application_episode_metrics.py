"""Independent deferred episode diagnostics, outside application timing."""
import hashlib
import time
import numpy as np
from temporal_metrics import metrics,physical_arrays


DIAGNOSTIC_LEVELS=(1e-6,1e-5,1e-4,1e-3,1e-2)
ERRORS=('pressure','equation','velocity','tracer')


def evaluate_forecast(result,outputs,truth,scene,norm):
    tick=time.perf_counter();rows=[]
    mass=float(np.asarray(truth[0]['tracer'],dtype='f8').sum()/scene['cells_per_metre']**3)
    thresholds={name:{str(level):None for level in DIAGNOSTIC_LEVELS}for name in ERRORS}
    if [r['step']for r in outputs]!=list(range(1,result['physical_steps']+1)):
        raise ValueError('Every consumed physical state must be evaluated')
    for row in outputs:
        step=row['step'];reference=truth[step]
        revision=int(result['event_step']is not None and step>result['event_step'])
        if row['forcing_revision']!=revision:raise ValueError('Offline evaluation found a stale forcing revision')
        value=metrics(row['fields'],reference,reference['rhs'],scene,norm,mass)
        fields=physical_arrays(row['fields'])
        rows.append(dict(step=step,time_s=step/60,forcing_revision=revision,metrics=value,
            predicted_field_sha256={n:hashlib.sha256(v.tobytes()).hexdigest()for n,v in fields.items()}))
        for name,levels in thresholds.items():
            for level,first in levels.items():
                if first is None and value['relative'][name]>float(level):levels[level]=dict(step=step,time_s=step/60)
    observed={name:max((r['metrics']['relative'][name]for r in rows),default=None)for name in ERRORS}
    complete=result['completed_physical_interval']and len(rows)==result['requested_steps']
    return dict(rows=rows,complete=complete,worst=observed if complete else None,observed_prefix_worst=observed,
        accuracy_pass=complete and all(r['metrics']['all_accuracy_criteria_pass']for r in rows),
        worst_scope='Complete declared interval'if complete else'Observed prefix only; full-interval accuracy unproven',
        diagnostic_drift_thresholds=thresholds,application_acceptance_limits=False,
        offline_metric_seconds=time.perf_counter()-tick,offline_metrics_in_application_time=False)


def evaluate_native(result,outputs,truth,scene,norm):
    tick=time.perf_counter();rows=[]
    mass=float(np.asarray(truth[0]['tracer'],dtype='f8').sum()/scene['cells_per_metre']**3)
    steps=[r['step']for r in outputs]
    if steps!=sorted(set(steps))or any(not 0<s<=result['physical_steps']for s in steps):
        raise ValueError('Native diagnostics must use unique accepted physical timestamps')
    for row in outputs:
        step=row['step'];revision=int(result['event_step']is not None and step>result['event_step'])
        if row['forcing_revision']!=revision:raise ValueError('Native diagnostic forcing revision differs')
        value=metrics(row['fields'],truth[step],truth[step]['rhs'],scene,norm,mass)
        rows.append(dict(step=step,time_s=step/60,forcing_revision=revision,metrics=value))
    required={s for s in (12,48,240)if s<=result['requested_steps']}
    complete=result['completed_physical_interval']and required.issubset(steps)
    observed={name:max((r['metrics']['relative'][name]for r in rows),default=None)for name in ERRORS}
    return dict(rows=rows,complete=complete,worst_at_diagnostic_checkpoints=observed if complete else None,
        observed_prefix_checkpoint_worst=observed,diagnostic_accuracy_pass=complete and all(r['metrics']['all_accuracy_criteria_pass']for r in rows),
        error_scope='Native accepted checkpoint diagnostics; not an every-step worst-error claim',
        offline_metric_seconds=time.perf_counter()-tick,offline_metrics_in_application_time=False)
