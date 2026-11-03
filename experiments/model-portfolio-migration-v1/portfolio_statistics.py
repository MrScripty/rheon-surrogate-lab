"""Small scalar-only benchmark summaries;no model or prediction persistence."""
import math
import statistics


def distribution(values):
    values=list(values)
    if not values or any(type(v) not in (float,int) or not math.isfinite(v) for v in values):
        raise ValueError('Nonempty finite scalar observations required')
    ordered=sorted(values)
    position=.95*(len(ordered)-1);lo=math.floor(position);hi=math.ceil(position)
    return dict(samples=len(values),best=ordered[0],worst=ordered[-1],mean=statistics.fmean(values),
        median=statistics.median(values),p95=ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo),
        p95_definition='Linear interpolation at0.95*(n-1) in sorted observed values')


def physical_metrics(row):
    measured=row['metrics']
    result={name+'_relative_l2':value for name,value in measured['relative'].items()}
    result.update({name+'_absolute_rms':value for name,value in measured['absolute_rms'].items()})
    if 'equation_absolute_rms' in measured:result['equation_absolute_rms']=measured['equation_absolute_rms']
    if 'pressure_cell_zero_absolute_error' in measured:result['pressure_cell_zero_absolute_error']=measured['pressure_cell_zero_absolute_error']
    return result


def summarize(rows):
    if not rows:raise ValueError('Actual scalar rows required')
    metrics={}
    names=set().union(*(r['values'] for r in rows))
    for name in sorted(names):
        samples=[r for r in rows if name in r['values']]
        result=distribution(r['values'][name] for r in samples)
        result['best_case']=min(samples,key=lambda r:r['values'][name])['identity']
        result['worst_case']=max(samples,key=lambda r:r['values'][name])['identity']
        metrics[name]=result
    return dict(observations=len(rows),independent_scene_ids=len({r['identity']['identifier'] for r in rows}),metrics=metrics)


def grouped(rows,selectors):
    result={}
    for name,select in selectors.items():
        groups={}
        for row in rows:
            key=select(row)
            if key is not None:groups.setdefault(str(key),[]).append(row)
        result[name]={key:summarize(items) for key,items in sorted(groups.items())}
    return result
