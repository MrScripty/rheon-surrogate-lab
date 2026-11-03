"""Finite comparison of independently advanced blind trajectories."""
from v5_support import ROOT,BASE,DOCS,read,write
def plan():
    commands=[];(BASE/'drift-blind').mkdir(exist_ok=True)
    for row in read(DOCS/'scene-reservation.json')['rows']:
        if row['population']!='rollout-blind':continue
        for policy in ['tiny-fresh','tiny-recycled']:
            name=row['trajectory_id']+'-'+policy
            commands.append(dict(id=name,runtime='torch',timeout=300,arguments=['experiments/pressure-methods-v5/drift_blind.py','--trajectory',row['trajectory_id'],'--policy',policy,
                '--output',str((BASE/'drift-blind'/(name+'.json')).relative_to(ROOT))]))
    write(BASE/'blind-drift-plan-03.json',dict(commands=commands))
if __name__=='__main__':plan()

