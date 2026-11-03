"""Freeze common development comparisons before looking at candidate outcomes."""
from v5_support import ROOT,BASE,DOCS,write,sha
def plan():
    points=[]
    for arm in ['v5-raw','v5-multilevel','v5-spectrum','v3-native']:
        version=5 if arm.startswith('v5-')else 3
        for method in ['vcycle','gcr']:
            points.append(dict(identifier=arm+'-'+method,entry=f'runs/pressure-methods-v{version}/entries/{arm}.json',method=method,numerical=False,optimized=False,channels_last=False))
    for method in ['vcycle','gcr','spectral']:
        points.append(dict(identifier='numerical-'+method,entry='runs/pressure-methods-v3/entries/v3-native.json',method=method,numerical=True,optimized=False,channels_last=False))
    for name,entry,method,numerical,layout in [
        ('v5-raw-gcr-optimized','v5-raw','gcr',False,False),('v5-raw-gcr-layout','v5-raw','gcr',False,True),
        ('v3-native-vcycle-optimized','v3-native','vcycle',False,False),('numerical-gcr-optimized','v3-native','gcr',True,False)]:
        version=5 if entry.startswith('v5-')else 3
        points.append(dict(identifier=name,entry=f'runs/pressure-methods-v{version}/entries/{entry}.json',method=method,numerical=numerical,optimized=True,channels_last=layout))
    write(DOCS/'development-matrix.json',dict(status='reserved_comparisons',points=points,prior_requests=192,new_native_requests=10,new_spectral_requests=18,source_sha256=sha(__file__)))
    for group in ['controls','trained']:
        commands=[]
        for point in points:
            if (point['identifier'].startswith(('v5-multilevel','v5-spectrum')))!=(group=='trained'):continue
            args=['experiments/pressure-methods-v5/evaluate.py','--entry',point['entry'],'--identifier',point['identifier'],'--method',point['method'],
                '--output',f"runs/pressure-methods-v5/evaluation/{point['identifier']}.json",'--old','--cpu','20'if group=='controls'else'21']
            for key,flag in [('numerical','--numerical'),('optimized','--optimized'),('channels_last','--channels-last')]:
                if point[key]:args.append(flag)
            commands.append(dict(id=point['identifier'],runtime='torch',arguments=args,timeout=900))
        write(BASE/(group+'-evaluation-plan-01.json'),dict(commands=commands))
if __name__=='__main__':plan()
