"""Capture independent references and fresh/recycled numerical controls."""
from v5_support import ROOT,BASE,DOCS,read,write
def plan():
    commands=[]
    for row in read(DOCS/'scene-reservation.json')['rows']:
        if row['population']!='rollout-development':continue
        for name,method,reuse in [('spectral','spectral',0),('numerical-fresh','gcr',0),('numerical-recycle4','gcr',4)]:
            output=BASE/'rollouts'/(row['trajectory_id']+'-'+name)
            arguments=['runs/pressure-methods-v5/native-capture-01.py','--trajectory',row['trajectory_id'],'--method',method,'--reuse',str(reuse),'--numerical',
                '--build','runs/pressure-methods-v4/streamed-build-01','--output',str(output.relative_to(ROOT)),
                '--entry','runs/pressure-methods-v3/entries/v3-native.json','--cpu','18','--optimized']
            if name=='spectral':arguments.append('--states')
            commands.append(dict(id=row['trajectory_id']+'-'+name,runtime='torch',arguments=arguments,timeout=900))
    write(BASE/'rollout-controls-plan-01.json',dict(commands=commands))
if __name__=='__main__':plan()
