"""Real consumed-native qualification of the exact frozen cost-admission rule."""
from v6_support import ROOT,BASE,DOCS,read,write,sha
def run():
    out=BASE/'adaptive-calibrated-qualification-plan-01.json'
    if out.exists():raise ValueError('Calibrated qualification already reserved')
    cal=read(DOCS/'reuse-calibration.json');point=cal['point'];rows=read(ROOT/'docs/plans/pressure-methods-v5/scene-reservation.json')['rows']
    row=next(r for r in rows if r['population']=='rollout-development'and r['scene']['counts']==[64]*3)
    args=['runs/pressure-methods-v6/native-capture-06.py','--trajectory',row['trajectory_id'],'--reservation','docs/plans/pressure-methods-v5/scene-reservation.json',
      '--entry',point['entry'],'--method',point['method'],'--variant',point['variant'],'--local-execution',point['local_execution'],'--optimized',
      '--reuse','4','--retention','coefficient','--adaptive-reuse','--reuse-calibration','docs/plans/pressure-methods-v6/reuse-calibration.json',
      '--build','runs/pressure-methods-v4/streamed-build-01','--output','runs/pressure-methods-v6/adaptive-calibrated-qualification-01','--cpu','21']
    if point['channels_last']:args+=['--execution','large-channels-last']
    write(out,dict(commands=[dict(id='calibrated-native-adaptive',runtime='torch',timeout=900,arguments=args)],calibration_sha256=sha(DOCS/'reuse-calibration.json'),automatic_retry=False))
    print('Reserved actual32-step native qualification of frozen calibrated admission',flush=True)
if __name__=='__main__':run()
