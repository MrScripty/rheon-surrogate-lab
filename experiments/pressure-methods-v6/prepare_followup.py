"""Reserve full-field long trajectories and repeated bare native timing after freeze."""
import copy
from v6_support import ROOT,BASE,DOCS,read,write,sha
def run():
    if(BASE/'long-rollout-plan-01.json').exists():raise ValueError('Follow-up workload already reserved')
    selection=read(DOCS/'selection.json');primary=selection['primary'];reservation=read(DOCS/'scene-reservation.json')
    baseline=next(p for p in read(DOCS/'calibration-points-03.json')['points']if p['identifier']=='v5-optimized')
    numerical=selection['numerical_primary']
    spectral=dict(numerical,identifier='spectral',method='spectral',fixed_jacobi=False)
    methods=[dict(label='selected-fresh',point=primary,reuse=0,adaptive=False),dict(label='selected-fixed4',point=primary,reuse=4,adaptive=False),
      dict(label='selected-adaptive4',point=primary,reuse=4,adaptive=True),dict(label='baseline-v5',point=baseline,reuse=0,adaptive=False),
      dict(label='numerical',point=numerical,reuse=0,adaptive=False),dict(label='spectral',point=spectral,reuse=0,adaptive=False)]
    def arguments(row,method,folder,reservation_path,states=False):
        p=method['point'];args=['runs/pressure-methods-v6/native-capture-06.py','--trajectory',row['trajectory_id'],'--reservation',reservation_path,
          '--entry',p['entry'],'--method',p['method'],'--variant',p['variant'],'--local-execution',p['local_execution'],'--optimized',
          '--reuse',str(method['reuse']),'--retention','coefficient','--cpu','23','--timeout','3600',
          '--build','runs/pressure-methods-v4/streamed-build-01','--output',str(folder.relative_to(ROOT))]
        if p['channels_last']:args+=['--execution','large-channels-last']
        if p['numerical']:args+=['--numerical']
        if p.get('fixed_jacobi'):args+=['--fixed-jacobi']
        if method['adaptive']:args+=['--adaptive-reuse','--reuse-calibration','docs/plans/pressure-methods-v6/reuse-calibration.json']
        if states:args+=['--states']
        return args
    long=[];drift=[]
    for row in reservation['rows']:
        if row['population']!='long-blind':continue
        for method in [methods[-1],methods[-2],methods[0],methods[2]]:
            name=row['trajectory_id']+'-'+method['label'];folder=BASE/'long-rollouts'/name
            long.append(dict(id=name,runtime='torch',timeout=3900,arguments=arguments(row,method,folder,'docs/plans/pressure-methods-v6/scene-reservation.json',True)))
            if method['label']!='spectral':
                drift.append(dict(id=name,runtime='torch',timeout=600,arguments=['experiments/pressure-methods-v6/drift.py',
                  '--reference',str((BASE/'long-rollouts'/(row['trajectory_id']+'-spectral')).relative_to(ROOT)),
                  '--candidate',str(folder.relative_to(ROOT)),'--output',str((BASE/'long-drift'/(name+'.json')).relative_to(ROOT))]))
    write(BASE/'long-rollout-plan-01.json',dict(commands=long,selection_sha256=sha(DOCS/'selection.json'),automatic_retry=False))
    write(BASE/'long-drift-plan-01.json',dict(commands=drift,selection_sha256=sha(DOCS/'selection.json'),automatic_retry=False))
    rows=[]
    for counts in [[64]*3,[128]*3,[256]*3]:
        original=next(r for r in reservation['rows']if r['scene']['counts']==counts and r['scene']['cells_per_metre']==16 and r['population']in ['blind','large-blind'])
        row=copy.deepcopy(original);row['source_trajectory']=row['trajectory_id'];row['trajectory_id']='v6-final-timing-'+'x'.join(map(str,counts));row['population']='blind-timing';row['snapshots']=[]
        row['scene']['steps']=16 if counts==[64]*3 else 4;rows.append(row)
    write(DOCS/'timing-reservation.json',dict(status='reserved_after_development_selection_before_new_blind_results',rows=rows,methods=methods,
      selection_sha256=sha(DOCS/'selection.json'),scope='Three serial balanced repeats per method/grid;16 steps at64³ and bounded4-step prefixes at128³/256³. No labels, state exports or GUI.'))
    commands=[]
    for row in rows:
        for repeat in range(3):
            order=methods if repeat==0 else list(reversed(methods))if repeat==1 else methods[3:]+methods[:3]
            for method in order:
                name=row['trajectory_id']+'-'+method['label']+f'-r{repeat}';folder=BASE/'quiet-native'/name
                commands.append(dict(id=name,runtime='torch',timeout=3900,arguments=arguments(row,method,folder,'docs/plans/pressure-methods-v6/timing-reservation.json')))
    write(BASE/'final-native-timing-plan-01.json',dict(commands=commands,selection_sha256=sha(DOCS/'selection.json'),requires_fresh_readiness=True,automatic_retry=False))
    print('Reserved eight256-step full-state episodes, six drift comparisons, and54 bare native timing episodes',flush=True)
if __name__=='__main__':run()
