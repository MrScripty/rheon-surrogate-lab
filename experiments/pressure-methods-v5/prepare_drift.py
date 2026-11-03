"""Verify full-state candidate drift against matched spectral reference tracks."""
from v5_support import ROOT,BASE,DOCS,read,write
def plan():
    commands=[]
    for row in read(DOCS/'scene-reservation.json')['rows']:
        if row['population']!='rollout-development':continue
        for policy in ['tiny-recycle0','tiny-recycle2','tiny-recycle4','tiny-recycle4-tail']:
            name=row['trajectory_id']+'-'+policy
            commands.append(dict(id=name,runtime='torch',arguments=['experiments/pressure-methods-v5/drift.py','--trajectory',row['trajectory_id'],'--policy',policy,
                '--output',str((BASE/'drift'/(name+'.json')).relative_to(ROOT))],timeout=300))
    (BASE/'drift').mkdir(exist_ok=True);write(BASE/'drift-plan-03.json',dict(commands=commands))
if __name__=='__main__':plan()
