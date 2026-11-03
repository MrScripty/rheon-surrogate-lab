"""Reserve real development cost profiles for the frozen primary learned point."""
import copy
from v6_support import ROOT,BASE,DOCS,read,write,sha
def run():
    out=BASE/'quiet-reuse-cost-plan-01.json'
    if out.exists():raise ValueError('Reuse cost plan already reserved')
    pre=read(DOCS/'preselection.json');point=pre['selected'][0]['point'];old=read(ROOT/'docs/plans/pressure-methods-v5/scene-reservation.json')
    scenes=[r for r in old['rows']if(r['population']=='rollout-development'and r['scene']['counts']in [[32]*3,[64]*3])or(r['population']=='development'and r['scene']['counts']==[128]*3)]
    if len(scenes)!=3:raise ValueError('Three consumed32³/64³/128³ scenes required')
    rows=[];commands=[]
    for source in scenes:
        row=copy.deepcopy(source);row['source_trajectory']=row['trajectory_id'];row['trajectory_id']='v6-cost-'+'x'.join(map(str,row['scene']['counts']));row['population']='consumed-development-cost';row['snapshots']=[];rows.append(row)
        for label,capacity,retention in [('fresh',0,'coefficient'),('pressure4',4,'contribution'),('coefficient4',4,'coefficient')]:
            args=['runs/pressure-methods-v6/native-capture-03.py','--trajectory',row['trajectory_id'],'--reservation','docs/plans/pressure-methods-v6/reuse-cost-reservation.json',
              '--method',point['method'],'--entry',point['entry'],'--variant',point['variant'],'--local-execution',point['local_execution'],
              '--reuse',str(capacity),'--retention',retention,'--optimized','--cpu','23','--build','runs/pressure-methods-v4/streamed-build-01',
              '--output',str((BASE/'quiet-reuse-cost'/(row['trajectory_id']+'-'+label)).relative_to(ROOT))]
            if point['channels_last']:args+=['--execution','large-channels-last']
            commands.append(dict(id=row['trajectory_id']+'-'+label,runtime='torch',arguments=args,timeout=1800))
    write(DOCS/'reuse-cost-reservation.json',dict(status='reserved_from_consumed_development',rows=rows,point=point,preselection_sha256=sha(DOCS/'preselection.json'),
      scope='Instrumented development cost calibration: one episode per grid/method,32 steps at32³/64³ and8 at128³. Final repeated native timing is separate.'))
    write(out,dict(commands=commands,automatic_retry=False,source_sha256=sha(__file__),preselection_sha256=sha(DOCS/'preselection.json')))
    print('Reserved nine instrumented consumed-native cost profiles',flush=True)
if __name__=='__main__':run()
