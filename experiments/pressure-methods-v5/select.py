"""Freeze model, basis retention and execution before all new blind material."""
import statistics
from v5_support import ROOT,HERE,BASE,DOCS,read,write,sha
def select():
    if(DOCS/'selection.json').exists():raise ValueError('Blind selection already frozen')
    model=read(DOCS/'model-selection.json');arm=model['selected']['arm'];entry=f'runs/pressure-methods-v5/entries/{arm}.json'
    matrix=read(DOCS/'development-matrix.json');evidence={}
    for row in matrix['points']:
        p=BASE/'evaluation-final'/(row['identifier']+'.json');d=read(p)
        if d['status']!='completed'or len(d['rows'])!=220:raise ValueError('Complete common development matrix required')
        evidence[str(p.relative_to(ROOT))]=sha(p)
    p=BASE/'evaluation-final/v5-initial-gcr.json';d=read(p)
    if not d['initialization_only']or len(d['rows'])!=220:raise ValueError('Untrained control required')
    evidence[str(p.relative_to(ROOT))]=sha(p)
    for name in ['controls-evaluation-batch-02','trained-evaluation-batch-02','rollout-model-batch-02','composition-batch-03','drift-batch-03']:
        p=BASE/name/'execution.json';d=read(p)
        if d['status']!='completed'or not d['owned_child_reaped']:raise ValueError('Complete development dependency required: '+name)
        evidence[str(p.relative_to(ROOT))]=sha(p)
    reuse=[]
    scenes=[r for r in read(DOCS/'scene-reservation.json')['rows']if r['population']=='rollout-development']
    for capacity in [0,2,4]:
        ds=[]
        for row in scenes:
            p=BASE/'rollouts-final'/(row['trajectory_id']+f'-tiny-recycle{capacity}')/'receipt.json';d=read(p);ds.append(d);evidence[str(p.relative_to(ROOT))]=sha(p)
        advanced=sum(sum(f['advanced']for f in d['frames'])for d in ds);cycles=sum(sum(c['work']['coarse_calls']for c in d['callbacks'])for d in ds)
        cache=max(c['work']['cache_bytes']for d in ds for c in d['callbacks'])
        reuse.append(dict(capacity=capacity,advanced=advanced,requested=96,coarse_calls=cycles,max_cache_bytes=cache,score=[advanced,-cycles,-cache]))
    selected=max(reuse,key=lambda r:tuple(r['score']))
    screen=read(BASE/'execution-selected-03.json');executions=[]
    for name in ['eager','direct-fine','direct-fine-channels-last']:
        group=[next(x for x in r['policy']if x['execution']==name)for r in screen['rows']]
        executions.append(dict(name=name,qualified=sum(x['work']['qualified']for x in group),sum_exploratory_seconds=sum(x['seconds']for x in group)))
    layout=all(read(BASE/'coupled-composition'/n/'receipt.json')['status']=='completed'for n in ['tiny-layout-fresh','tiny-layout-recycle4'])
    usable=[r for r in executions if r['qualified']==6 and(layout or r['name']!='direct-fine-channels-last')]
    execution=min(usable,key=lambda r:r['sum_exploratory_seconds'])['name']
    optimized=execution.startswith('direct-fine');channels_last=execution.endswith('channels-last')
    points=[]
    def add(id,model_id=arm,method='gcr',numeric=False,opt=False,reuse=0,layout=False,retention='contribution'):
        version=5 if model_id.startswith('v5-')else 3
        points.append(dict(identifier=id,entry=f'runs/pressure-methods-v{version}/entries/{model_id}.json',method=method,numerical=numeric,optimized=opt,reuse=reuse,channels_last=layout,retention=retention))
    for model_id in ['v5-raw','v5-multilevel','v5-spectrum']:add(model_id+'-gcr',model_id)
    add('v3-native-vcycle','v3-native',method='vcycle')
    add('numerical-gcr',numeric=True);add('spectral',method='spectral',numeric=True)
    add('selected-optimized',opt=True);add('selected-layout',opt=True,layout=True)
    add('selected-recycled',opt=optimized,reuse=selected['capacity'],layout=channels_last)
    add('selected-tail4',opt=optimized,reuse=4,layout=channels_last,retention='tail')
    add('numerical-optimized',numeric=True,opt=True);add('numerical-recycled',numeric=True,opt=True,reuse=4)
    add('v3-native-optimized','v3-native',method='vcycle',opt=True)
    for p in [BASE/'execution-selected-03.json',BASE/'sequence-stress-03.json',BASE/'returned-pressure-green-03.json']:
        evidence[str(p.relative_to(ROOT))]=sha(p)
    result=dict(status='frozen_before_blind_capture',selected_model=arm,entry=read(ROOT/entry),reuse=selected,reuse_candidates=reuse,
        selected_execution=execution,optimized=optimized,channels_last=channels_last,execution_candidates=executions,
        note='Execution selection uses the qualified development screen; final serial timing tests the choice without refitting or reselection.',
        blind_points=points,initialization_control=True,all_development_evaluations=evidence,
        model_selection_sha256=sha(DOCS/'model-selection.json'),selection_rule_sha256=sha(DOCS/'selection-rule.json'),
        reservation_sha256=sha(DOCS/'scene-reservation.json'),source_sha256=sha(__file__),promotion=False)
    write(DOCS/'selection.json',result)
    capture=[]
    for row in read(DOCS/'scene-reservation.json')['rows']:
        if row['population']!='blind':continue
        capture.append(dict(id=row['trajectory_id'],runtime='torch',timeout=600,arguments=['runs/pressure-methods-v5/native-capture-03.py',
            '--trajectory',row['trajectory_id'],'--method','spectral','--numerical','--entry',entry,'--record-requests','--cpu','18',
            '--build','runs/pressure-methods-v4/streamed-build-01','--output',str((BASE/'native-data'/(row['trajectory_id']+'-spectral')).relative_to(ROOT))]))
    write(BASE/'blind-capture-plan-03.json',dict(commands=capture))
    groups=[[],[]]
    for i,row in enumerate(points):
        args=['experiments/pressure-methods-v5/evaluate_v3.py','--entry',row['entry'],'--identifier',row['identifier'],'--method',row['method'],
            '--reuse',str(row['reuse']),'--retention',row['retention'],'--population','blind','--cpu',str(20+i%2),
            '--output',str((BASE/'blind-evaluation'/(row['identifier']+'.json')).relative_to(ROOT))]
        for k,f in [('numerical','--numerical'),('optimized','--optimized'),('channels_last','--channels-last')]:
            if row[k]:args.append(f)
        groups[i%2].append(dict(id=row['identifier'],runtime='torch',timeout=900,arguments=args))
    groups[0].append(dict(id='initial-control',runtime='torch',timeout=900,arguments=['experiments/pressure-methods-v5/evaluate_initial_v3.py','--population','blind','--cpu','20',
        '--output',str((BASE/'blind-evaluation/v5-initial-gcr.json').relative_to(ROOT))]))
    for i,commands in enumerate(groups):write(BASE/f'blind-evaluation-plan-03-{i}.json',dict(commands=commands))
    rollouts=[]
    for row in read(DOCS/'scene-reservation.json')['rows']:
        if row['population']!='rollout-blind':continue
        for name,method,numeric,capacity,retention,state_fields in [
            ('spectral','spectral',True,0,'contribution',True),('tiny-fresh','gcr',False,0,'contribution',True),
            ('tiny-recycled','gcr',False,selected['capacity'],'contribution',True),('tiny-tail4','gcr',False,4,'tail',False),
            ('numerical-fresh','gcr',True,0,'contribution',False),('numerical-recycled','gcr',True,4,'contribution',False)]:
            args=['runs/pressure-methods-v5/native-capture-03.py','--trajectory',row['trajectory_id'],'--method',method,'--reuse',str(capacity),'--retention',retention,
                '--entry',entry,'--cpu','19','--build','runs/pressure-methods-v4/streamed-build-01','--output',str((BASE/'rollouts-blind'/(row['trajectory_id']+'-'+name)).relative_to(ROOT))]
            if numeric:args.append('--numerical')
            if method!='spectral'and(optimized or numeric):args.append('--optimized')
            if channels_last and not numeric:args.extend(['--execution','large-channels-last'])
            if state_fields:args.append('--states')
            rollouts.append(dict(id=row['trajectory_id']+'-'+name,runtime='torch',timeout=900,arguments=args))
    write(BASE/'blind-rollout-plan-03.json',dict(commands=rollouts))
    print('Frozen blind selection:',arm,'capacity',selected['capacity'],'execution',execution,flush=True)
if __name__=='__main__':select()
