"""Freeze serial common-population and native paired timing commands."""
from v4_support import ROOT,BASE,DOCS,read,write,sha

def prepare():
    selection=read(DOCS/'selection.json')
    if selection['status']!='frozen_before_blind_capture':raise ValueError('Frozen selection required')
    if (BASE/'quiet-plan-01.json').exists():raise ValueError('Fresh timing plan required')
    entries=BASE/'timing-entries';entries.mkdir()
    original=next(r for r in read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')['models']if r['id']=='C53')
    write(entries/'C53.json',original)
    native=read(ROOT/'runs/pressure-methods-v3/entries/v3-native.json');write(entries/'v3-native.json',native)
    rows=[('C53-fine',entries/'C53.json','fine'),('C53-num6',entries/'C53.json','numerical'),
        ('v2-num6',BASE/'entries/v2-num6.json','numerical')]
    rows += [('v3-native-'+policy,entries/'v3-native.json',policy)for policy in ['learned','large-channels-last','vcycle-neural','vcycle-only','gcr-multilevel','spectral']]
    rows += [(arm+'-learned',BASE/'entries'/(arm+'.json'),'learned')for arm in ['v4-student75-distill','v4-student75-control','v4-student50-distill','v4-student50-control','v4-replay','v4-gcr','v4-diverse','v4-onpolicy']]
    rows += [(arm+'-gcr-learned',BASE/'entries'/(arm+'.json'),'gcr-learned')for arm in ['v4-replay','v4-gcr']]
    commands=[]
    for identifier,entry,policy in rows:
        commands.append(dict(id=identifier,timeout=300,arguments=['experiments/pressure-methods-v4/quiet.py',
            '--identifier',identifier,'--entry',str(entry.relative_to(ROOT)),'--policy',policy,'--population','historical128',
            '--output',str((BASE/'quiet'/(identifier+'.json')).relative_to(ROOT))]))
    write(BASE/'quiet-plan-01.json',dict(commands=commands,selection_sha256=sha(DOCS/'selection.json'),source_sha256=sha(__file__)))
    reserved=read(DOCS/'scene-reservation.json');timing=[]
    for shape in [[64]*3,[128]*3,[73,49,33]]:
        source=next(r for r in reserved['rows']if r['population']=='blind'and r['scene']['counts']==shape)
        identity='v4-timing-'+'x'.join(map(str,shape))
        timing.append(dict(source,trajectory_id=identity,population='blind-timing',scene=dict(source['scene'],steps=4),snapshots=[0,3],source_trajectory=source['trajectory_id']))
    reservation=DOCS/'timing-reservation.json';write(reservation,dict(status='frozen_matched_prefix_of_reserved_blind_scenes',rows=timing,
        source_reservation_sha256=sha(DOCS/'scene-reservation.json'),selection_sha256=sha(DOCS/'selection.json')))
    methods=[]
    for r in selection['selected']:
        path=entries/(r['identifier']+'.json');write(path,r['entry'])
        methods.append(dict(identifier=r['identifier'],entry=path,policy=r['policy'],execution='eager'))
    for identifier,policy in [('v3-native-baseline','learned'),('spectral','spectral'),('numerical-gcr-multilevel','gcr-multilevel')]:
        methods.append(dict(identifier=identifier,entry=entries/'v3-native.json',policy=policy,execution='eager'))
    methods.insert(4,dict(identifier='v3-native-layout',entry=entries/'v3-native.json',policy='learned',execution='large-channels-last'))
    coupled=[]
    for row in timing:
        for rep in range(3):
            order=methods if rep==0 else list(reversed(methods))if rep==1 else methods[3:]+methods[:3]
            for method in order:
                label=row['trajectory_id']+'-'+method['identifier']+f'-r{rep}'
                coupled.append(dict(id=label,timeout=900,arguments=[str((BASE/'candidate-capture-02.py').relative_to(ROOT)),
                    '--trajectory',row['trajectory_id'],'--reservation',str(reservation.relative_to(ROOT)),
                    '--entry',str(method['entry'].relative_to(ROOT)),'--policy',method['policy'],'--execution',method['execution'],
                    '--build',str((BASE/'streamed-build-01').relative_to(ROOT)),'--output',str((BASE/'coupled-quiet'/label).relative_to(ROOT))]))
    write(BASE/'coupled-quiet-plan-01.json',dict(commands=coupled,selection_sha256=sha(DOCS/'selection.json'),source_sha256=sha(__file__),
        timing='Three serial per-method repeats on matched four-step prefixes. Ordering forward, reverse, rotated; baseline/layout appear in AB, BA, AB order.'))
    print('Prepared',len(commands),'historical policy points and',len(coupled),'native timing episodes',flush=True)

if __name__=='__main__':prepare()
