"""Choose two learned operating points only after common repeated development."""
from v6_support import ROOT,HERE,BASE,DOCS,read,write,sha,fit_record
def run():
    output=DOCS/'preselection.json'
    if output.exists():raise ValueError('Preselection already frozen')
    evidence={}
    def keep(p):evidence[str(p.relative_to(ROOT))]=sha(p);return read(p)
    campaign=keep(DOCS/'campaign.json');keep(DOCS/'fit-records.json');keep(DOCS/'selection-activity-clarification.json');points=keep(DOCS/'calibration-points-03.json')['points'];all_points=[];signatures=set()
    for arm in campaign['fits']:
        d=keep(ROOT/fit_record(arm)['study']/'receipt.json')
        if d['status']!='completed'or not all(d['artifact_parity'].values()):raise ValueError('All4 completed matched fits/restores required')
        if keep(BASE/'transfer'/(arm+'.json'))['status']!='passed':raise ValueError('Full128³ deployed transfer required')
    timing=keep(BASE/'quiet-development-batch-03/execution.json')
    if timing['status']!='completed'or not timing['owned_child_reaped']:raise ValueError('Repeated development timing has not drained')
    for point in points:
        d=keep(BASE/'development'/(point['identifier']+'.json'));t=keep(BASE/'quiet-development'/(point['identifier']+'.json'))
        if d['cases']!=254 or t['summary']['cases']!=20 or t['points_sha256']!=sha(DOCS/'calibration-points-03.json'):raise ValueError('Common full development/timing populations required')
        signatures.add((d['population_sha256'],t['population_sha256']))
        all_points.append(dict(point=point,development_qualified=d['qualified'],timing_qualified=t['summary']['qualified'],
          median_ms=t['summary']['median_ms'],actual_network_calls=sum(r['network_calls']for r in t['rows']),worst_pressure_relative_l2=t['summary']['worst_pressure_relative_l2'],
          score=[-d['qualified'],-t['summary']['qualified'],t['summary']['median_ms'],t['summary']['worst_pressure_relative_l2']]))
    if len(signatures)!=1:raise ValueError('Different development/timing input populations')
    candidates=[r for r in all_points if not r['point']['numerical']and r['actual_network_calls']>0]
    best=sorted(candidates,key=lambda r:tuple(r['score']))[:2]
    numerical_best=min([r for r in all_points if r['point']['numerical']and r['point']['method']=='gcr'],key=lambda r:tuple(r['score']))
    if any(r['development_qualified']!=254 or r['timing_qualified']!=20 for r in best):raise ValueError('Frozen learned candidates must retain all gates')
    for name in ['quiet-ready-development-03.json','timing-admission-development-03.json']:
        keep(BASE/name)
    write(output,dict(status='frozen_from_consumed_development',selected=best,numerical_primary=numerical_best,all_candidates=all_points,evidence=evidence,
      population='All254 consumed development requests; repeated timing on20 consumed native requests',
      final_selection_pending='Calibrate reuse costs and freeze native execution before any newly reserved blind capture',
      source_sha256=sha(__file__),selection_rule_sha256=sha(DOCS/'selection-rule.json'),reservation_sha256=sha(DOCS/'scene-reservation.json'),
      no_more_fits=True,promotion=False))
    print('Frozen learned candidates:',*[r['point']['identifier']for r in best],flush=True)
if __name__=='__main__':run()
