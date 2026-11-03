"""Freeze learned points and the development-only reuse rule before blind capture."""
import copy,time
from v6_support import ROOT,HERE,BASE,DOCS,read,write,sha
def run():
    output=DOCS/'selection.json'
    if output.exists():raise ValueError('Selection is already frozen')
    pre=read(DOCS/'preselection.json');cal=read(DOCS/'reuse-calibration.json');primary=pre['selected'][0]['point']
    if cal['preselection_sha256']!=sha(DOCS/'preselection.json')or cal['point']!=primary:raise ValueError('Calibrated solver differs from primary frozen point')
    qualification=read(BASE/'adaptive-calibrated-qualification-01/receipt.json')
    if qualification['status']!='completed'or qualification['reuse_calibration_sha256']!=sha(DOCS/'reuse-calibration.json')or not qualification['native_process']['reaped']or not all(f['advanced']and f['quality']['qualified']for f in qualification['frames']):raise ValueError('Real native qualification of calibrated admission required')
    if cal['solver_source_sha256']!=sha(HERE/'solver_v2.py')or cal['adaptive_source_sha256']!=sha(HERE/'adaptive.py'):raise ValueError('Qualified predictor source changed')
    for name,expected in pre['evidence'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Preselection evidence changed')
    for name,expected in cal['evidence'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Cost evidence changed')
    points=read(DOCS/'calibration-points-03.json')['points'];by_id={p['identifier']:p for p in points};blind=[];seen=set()
    for point in [*[by_id[a]for a in read(DOCS/'campaign.json')['fits']],*[r['point']for r in pre['selected']],by_id['v5-optimized'],by_id['numerical'],by_id['fixed-jacobi'],by_id['spectral']]:
        key=tuple(point[k]for k in ['entry','variant','local_execution','channels_last','numerical','method'])+(point.get('fixed_jacobi',False),)
        if key not in seen:seen.add(key);blind.append(point)
    sources={str((HERE/n).relative_to(ROOT)):sha(HERE/n)for n in ['v6_support.py','solver.py','solver_v2.py','recycling.py','recycling_v2.py','adaptive.py','native_local.py','local_native.cpp','local_native_stream.cpp','fixed_control.py']}
    sources[str((BASE/'native-capture-06.py').relative_to(ROOT))]=sha(BASE/'native-capture-06.py')
    for folder in ['local-native-build-01','local-native-stream-build-01']:
        for name in ['build.json','local_native.so']:
            p=BASE/folder/name;sources[str(p.relative_to(ROOT))]=sha(p)
    sources[str((DOCS/'reuse-calibration.json').relative_to(ROOT))]=sha(DOCS/'reuse-calibration.json')
    runtime_manifest=ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json'
    sources[str(runtime_manifest.relative_to(ROOT))]=sha(runtime_manifest)
    for point in blind:
        entry_path=ROOT/point['entry'];sources[point['entry']]=sha(entry_path);entry=read(entry_path)
        for name,expected in entry['identities'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Restored model material changed before freeze')
            sources[name]=expected
        project=ROOT/entry['project']
        for name,expected in read(project/'source-witness.json')['sources'].items():
            if sha(project/name)!=expected:raise ValueError('Captured predictor changed before freeze')
            sources[str((project/name).relative_to(ROOT))]=expected
    write(output,dict(status='frozen_before_blind_capture',frozen_unix=time.time(),selected=pre['selected'],primary=primary,numerical_primary=pre['numerical_primary']['point'],blind_points=blind,
      reuse_calibration_sha256=sha(DOCS/'reuse-calibration.json'),preselection_sha256=sha(DOCS/'preselection.json'),
      reservation_sha256=sha(DOCS/'scene-reservation.json'),frozen_sources=sources,source_sha256=sha(__file__),no_more_fits=True,promotion=False))
    capture=[]
    for row in read(DOCS/'scene-reservation.json')['rows']:
        if row['population']not in ['blind','large-blind']:continue
        args=['runs/pressure-methods-v6/native-capture-06.py','--trajectory',row['trajectory_id'],'--reservation','docs/plans/pressure-methods-v6/scene-reservation.json',
          '--method','spectral','--numerical','--entry',primary['entry'],'--record-requests','--cpu','21','--build','runs/pressure-methods-v4/streamed-build-01',
          '--output',str((BASE/'native-data'/(row['trajectory_id']+'-spectral')).relative_to(ROOT))]
        capture.append(dict(id=row['trajectory_id'],runtime='torch',arguments=args,timeout=1800))
    write(BASE/'blind-capture-plan-01.json',dict(commands=capture,selection_sha256=sha(output),automatic_retry=False))
    groups=[[],[]]
    for i,point in enumerate(blind):
        args=['experiments/pressure-methods-v6/evaluate_blind.py','--identifier',point['identifier'],'--cpu',str(21+i%2),
          '--output',str((BASE/'blind-evaluation'/(point['identifier']+'.json')).relative_to(ROOT))]
        groups[i%2].append(dict(id=point['identifier'],runtime='torch',arguments=args,timeout=3600))
    for i,commands in enumerate(groups):write(BASE/f'blind-evaluation-plan-01-{i}.json',dict(commands=commands,selection_sha256=sha(output),automatic_retry=False))
    print('Frozen learned points and calibrated reuse; reserved seven-scene blind capture and all-fit evaluation',flush=True)
if __name__=='__main__':run()
