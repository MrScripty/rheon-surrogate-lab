"""Fit an explicit conservative cost rule using consumed native profiles only."""
import math,statistics
from v6_support import ROOT,HERE,BASE,DOCS,read,write,sha
def run():
    out=DOCS/'reuse-calibration.json'
    if out.exists():raise ValueError('Reuse calibration already frozen')
    execution=read(BASE/'quiet-reuse-cost-batch-01/execution.json')
    if execution['status']!='completed'or not execution['owned_child_reaped']:raise ValueError('All cost profiles must finish and drain')
    reservation=read(DOCS/'reuse-cost-reservation.json');bands=[];evidence={}
    for row in reservation['rows']:
        results={}
        for label in ['fresh','pressure4','coefficient4']:
            p=BASE/'quiet-reuse-cost'/(row['trajectory_id']+'-'+label)/'receipt.json';d=read(p);evidence[str(p.relative_to(ROOT))]=sha(p)
            if d['row']!=row or d['status']!='completed'or not d['native_process']['reaped']or not all(f['advanced']and f['quality']['qualified']for f in d['frames']):raise ValueError('Complete accepted development profile required')
            results[label]=d
        fresh=[c['work']for c in results['fresh']['callbacks']];cached=[c['work']for c in results['coefficient4']['callbacks']];cells=math.prod(row['scene']['counts'])
        original=sum(w['coarse_calls']for w in fresh);recycled=sum(w['coarse_calls']for w in cached)
        fraction=max(0.,min(.75,(original-recycled)/max(original,1)))
        marginal=[]
        for w in fresh:
            n=w['coarse_calls']
            if not n:continue
            last_neural=bool(w['neural_decisions']and w['neural_decisions'][-1]['neural'])
            cnn=w['network_seconds']/w['network_calls']if last_neural and w['network_calls']else 0.
            op=w['operator_seconds']/max(w['operator_applications'],1)
            marginal.append(w['coarse_seconds']/n+cnn+(2+int(last_neural))*op)
        if not marginal:raise ValueError('No observed marginal solver work')
        overhead=[w.get('recycle_projection_seconds',0.)+w.get('cached_orthogonalization_seconds',0.)+w.get('retention_seconds',0.)for w in cached if w['cache_in']]
        if not overhead:raise ValueError('No real cached-space overhead observed')
        baseline=statistics.median(w['coarse_calls']for w in fresh)
        bands.append(dict(cells=cells,counts_xyz=row['scene']['counts'],steps=row['scene']['steps'],baseline_iterations=baseline,
          saved_fraction=fraction,cycle_seconds_per_cell=statistics.median(marginal)/cells,overhead_seconds_per_cell=statistics.median(overhead)/cells,
          fresh_total_cycles=original,cached_total_cycles=recycled,profile='Full native step input and solver work; median marginal tail-cycle cost and measured cached-space overhead.'))
    write(out,dict(status='frozen_from_consumed_development',point=reservation['point'],bands=bands,minimum_cosine=.9,overhead_margin=1.15,
      evidence=evidence,preselection_sha256=sha(DOCS/'preselection.json'),reservation_sha256=sha(DOCS/'reuse-cost-reservation.json'),
      source_sha256=sha(__file__),adaptive_source_sha256=sha(HERE/'adaptive.py'),solver_source_sha256=sha(HERE/'solver_v2.py'),
      inference='Expected saved cycles from paired development totals; recent fresh-solve work is tracked. Removing a numerical-only tail does not claim savings from earlier neural calls.',
      limitations=['Cost extrapolation to other cell counts is a hypothesis','Single calibration episode per grid/method; final repeated timing remains independent','Paired independently advanced trajectories do not provide identical later pressure equations'],promotion=False))
    print('Frozen development-only cost calibration with15% overhead margin',flush=True)
if __name__=='__main__':run()
