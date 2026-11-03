"""Complete the declared fixed-zero/fixed-four/adaptive drift contrast without reselection."""
import copy,time
from v6_support import BASE,DOCS,read,write,sha

def run():
    output=BASE/'long-fixed-rollout-plan-01.json'
    if output.exists():raise ValueError('Fresh supplemental fixed-reuse plan required')
    original=read(BASE/'long-rollout-plan-01.json');drift=read(BASE/'long-drift-plan-01.json');commands=[];curves=[]
    for row in original['commands']:
        if not row['id'].endswith('-selected-fresh'):continue
        new=copy.deepcopy(row);new['id']=new['id'].replace('-selected-fresh','-selected-fixed4')
        args=new['arguments'];args[args.index('--reuse')+1]='4';args[args.index('--output')+1]=args[args.index('--output')+1].replace('-selected-fresh','-selected-fixed4');commands.append(new)
    for row in drift['commands']:
        if row['id'].endswith('-selected-fresh'):
            new=copy.deepcopy(row);new['id']=new['id'].replace('-selected-fresh','-selected-fixed4');new['arguments']=[a.replace('-selected-fresh','-selected-fixed4')for a in new['arguments']];curves.append(new)
    if len(commands)!=2 or len(curves)!=2:raise ValueError('Exactly two fixed-four trajectories required')
    write(output,dict(commands=commands,selection_sha256=sha(DOCS/'selection.json'),automatic_retry=False,registered_unix=time.time(),reason='Complete the existing explicit fixed-zero/fixed-four/adaptive drift requirement. No model selection, fits, scenes or acceptance gates change.'))
    write(BASE/'long-fixed-drift-plan-01.json',dict(commands=curves,selection_sha256=sha(DOCS/'selection.json'),automatic_retry=False))
    write(BASE/'long-fixed-intent-01.json',dict(prerequisites=['long-rollout-batch-01'],wait_seconds=14400,batches=[dict(specification='long-fixed-rollout-plan-01.json',output='long-fixed-rollout-batch-01',timeout=8000),dict(specification='long-fixed-drift-plan-01.json',output='long-fixed-drift-batch-01',timeout=1400)]))
    print('Reserved two fixed-four long attempts and matched full-field drift; original frozen plans preserved',flush=True)
if __name__=='__main__':run()
