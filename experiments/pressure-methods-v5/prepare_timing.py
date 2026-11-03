"""Freeze timing workloads before blind outcomes; admission still requires quiet."""
import copy
from v5_support import ROOT,BASE,DOCS,read,write,sha
def prepare():
    if(BASE/'all-timing-plan-03.json').exists():raise ValueError('Fresh timing reservation required')
    selection=read(DOCS/'selection.json');arm=selection['selected_model'];entry=f'runs/pressure-methods-v5/entries/{arm}.json';capacity=selection['reuse']['capacity']
    points=[]
    def pressure(id,model,method='gcr',numeric=False,opt=False,layout=False,initial=False):
        version=5 if model.startswith('v5-')else 3
        points.append(dict(identifier=id,entry=f'runs/pressure-methods-v{version}/entries/{model}.json',method=method,numerical=numeric,optimized=opt,channels_last=layout,initial=initial))
    for model in ['v5-raw','v5-multilevel','v5-spectrum']:pressure(model+'-gcr',model)
    pressure('selected-optimized',arm,opt=True);pressure('selected-layout',arm,opt=True,layout=True)
    pressure('v3-legacy','v3-native',method='legacy');pressure('v3-common','v3-native',method='vcycle')
    pressure('numerical-original',arm,numeric=True);pressure('numerical-optimized',arm,numeric=True,opt=True)
    pressure('spectral',arm,method='spectral',numeric=True);pressure('initial-control','v5-raw',initial=True)
    commands=[]
    for point in points:
        for population in ['historical128','native-blind']:
            args=['experiments/pressure-methods-v5/time_pressure.py','--entry',point['entry'],'--identifier',point['identifier'],'--method',point['method'],'--population',population,
                '--output',str((BASE/'quiet-pressure'/(point['identifier']+'-'+population+'.json')).relative_to(ROOT))]
            for key,flag in [('numerical','--numerical'),('optimized','--optimized'),('channels_last','--channels-last'),('initial','--initial-control')]:
                if point[key]:args.append(flag)
            commands.append(dict(id=point['identifier']+'-'+population,runtime='torch',arguments=args,timeout=900))
    rows=[]
    for row in read(DOCS/'scene-reservation.json')['rows']:
        if row['population']!='blind'or max(row['scene']['counts'])<64:continue
        row=copy.deepcopy(row);row['source_trajectory']=row['trajectory_id'];row['trajectory_id']='v5-timing-'+'x'.join(map(str,row['scene']['counts']));row['population']='blind-timing';row['scene']['steps']=4 if max(row['scene']['counts'])==128 else 32;row['snapshots']=[];rows.append(row)
    write(DOCS/'timing-reservation.json',dict(status='frozen_before_blind_results',rows=rows,selection_sha256=sha(DOCS/'selection.json'),
        note='32-step64/rectangular prefixes expose repeated timestep work; the128³ comparison is a bounded four-step prefix.',source_sha256=sha(__file__)))
    methods=[dict(identifier='legacy-v3',legacy=True,method='vcycle',numerical=False,reuse=0,opt=False,layout=False),
        dict(identifier='selected-original',method='gcr',numerical=False,reuse=0,opt=False,layout=False),
        dict(identifier='selected-optimized',method='gcr',numerical=False,reuse=0,opt=selection['optimized'],layout=selection['channels_last']),
        dict(identifier='selected-recycled',method='gcr',numerical=False,reuse=capacity,opt=selection['optimized'],layout=selection['channels_last']),
        dict(identifier='numerical-optimized',method='gcr',numerical=True,reuse=0,opt=True,layout=False),
        dict(identifier='numerical-recycled',method='gcr',numerical=True,reuse=4,opt=True,layout=False),
        dict(identifier='spectral',method='spectral',numerical=True,reuse=0,opt=False,layout=False)]
    for row in rows:
        trial_methods=list(methods)
        if row['scene']['counts']==[128]*3:trial_methods.append(dict(methods[2],identifier='selected-duplicate-audit',audit=True))
        for repeat in range(3):
            order=trial_methods if repeat==0 else list(reversed(trial_methods))if repeat==1 else trial_methods[3:]+trial_methods[:3]
            for point in order:
                name=row['trajectory_id']+'-'+point['identifier']+f'-r{repeat}';out=BASE/'quiet-native'/name
                if point.get('legacy'):
                    args=['runs/pressure-methods-v4/candidate-capture-02.py','--trajectory',row['trajectory_id'],'--policy','vcycle-neural','--entry','runs/pressure-methods-v3/entries/v3-native.json']
                else:
                    args=['runs/pressure-methods-v5/native-capture-03.py','--trajectory',row['trajectory_id'],'--method',point['method'],'--entry',entry,'--reuse',str(point['reuse'])]
                    if point['numerical']:args.append('--numerical')
                    if point['opt']:args.append('--optimized')
                    if point['layout']:args.extend(['--execution','large-channels-last'])
                    if point.get('audit'):args.append('--audit-callback')
                args+=['--reservation','docs/plans/pressure-methods-v5/timing-reservation.json','--build','runs/pressure-methods-v4/streamed-build-01',
                    '--output',str(out.relative_to(ROOT)),'--cpu','23']
                commands.append(dict(id=name,runtime='torch',arguments=args,timeout=1800))
    write(BASE/'all-timing-plan-03.json',dict(commands=commands,pressure_points=points,native_methods=methods,selection_sha256=sha(DOCS/'selection.json'),
        source_sha256=sha(__file__),requires_fresh_readiness=True))
    print('Reserved',len(commands),'serial timing tasks (22 pressure populations,66 native episodes)',flush=True)
if __name__=='__main__':prepare()
