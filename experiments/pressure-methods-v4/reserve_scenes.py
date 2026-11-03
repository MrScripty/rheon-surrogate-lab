"""Reserve whole-scene splits before new native fields or candidate selection."""
import random
from v4_support import BASE,DOCS,write,sha

def run():
    path=DOCS/'scene-reservation.json'
    if path.exists():raise ValueError('Fresh scene reservation required')
    generator=random.Random(2026101904);rows=[]
    def profile():
        return dict(seed=generator.randrange(2**32),center=[generator.uniform(.15,.85)for _ in range(3)],
            acceleration=generator.uniform(.04,.3),radius=generator.uniform(.12,.43),swirl=generator.uniform(-.95,.95),
            fluctuation=generator.uniform(.05,.75),frequency=generator.uniform(.2,3.5),sgs=False)
    def add(name,population,counts,cpm,steps,snapshots,forcing):
        scene=dict(forcing,counts=counts,cells_per_metre=cpm,steps=steps,warmup_steps=0)
        rows.append(dict(trajectory_id=name,population=population,scene=scene,snapshots=snapshots))
    for i,(counts,cpm)in enumerate([([32]*3,8),([33,25,17],16),([64]*3,16),([61,45,29],8)]*2):
        add(f'v4-diverse-train-{i:02d}','diverse-train',counts,cpm,16,[0,15],profile())
    for i,(counts,cpm,steps)in enumerate([([32]*3,16,120),([64]*3,16,120),([33,25,17],8,120),([64]*3,8,120)]):
        add(f'v4-onpolicy-train-{i:02d}','onpolicy-train',counts,cpm,steps,[0,15,59,119],profile())
    for i,(counts,cpm,steps)in enumerate([([16]*3,16,600),([32]*3,16,300),([64]*3,16,120),([128]*3,16,32)]):
        add(f'v4-rollout-development-{i:02d}','rollout-development',counts,cpm,steps,[0,steps-1],profile())
    for group in ['development','blind']:
        for i,(counts,cpm)in enumerate([([32]*3,8),([37,29,19],16),([64]*3,16),([73,49,33],8),([128]*3,16)]):
            add(f'v4-{group}-{i:02d}',group,counts,cpm,16,[0,15],profile())
    add('v4-early256-development','early256-development',[256]*3,32,1,[0],profile())
    write(path,dict(status='reserved_before_capture_and_v4_fits',seed=2026101904,rows=rows,source_sha256=sha(__file__),
        blind_gate='Capture only after all candidate selection and training is frozen',
        splits='Whole forcing scenes; no development, diagnostic256 or blind data may enter fitting',
        physical_context='Cubic cells, rho1000kg/m3, dt1/60s, unchanged sealed operator and host gates',
        drift_acceptance='Diagnostic curves only; no application drift threshold selected'))
    (BASE/'scenes').mkdir(parents=True,exist_ok=True)
    for row in rows:write(BASE/'scenes'/(row['trajectory_id']+'.json'),row['scene'])
    print('Reserved',len(rows),'whole scenes across six distinct populations',flush=True)

if __name__=='__main__':run()
