"""Rebuild typed Project closure after explicit experiment-source changes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid
from v4_support import ROOT,BASE,read,write,sha
from common import environment

def package(output,arm,extra_sources=()):
    prior=ROOT/'runs/pressure-methods-v3/v3-native-project-07'
    old=read(prior/'source-witness.json');project=read(output/'project.json')
    project.update(id=uuid.uuid4().hex,name='Pressure methods v4 · '+arm)
    config=read(output/'config.json');initial=next(r for r in project['data']if r['id']=='initial_weights')
    slots={r['id']:ROOT/r['path']for r in project['data']}
    for item in project['data']:
        if sha(slots[item['id']])!=item['sha256']:raise ValueError('Project input changed: '+item['id'])
    viewslots=['dataset','dataset_extra',*sorted(k for k in slots if k.startswith('view_'))]
    runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    probe="""import sys,json;sys.path.insert(0,sys.argv[1]);import model,data,policy
model.configure('cpu');model.select(json.load(open(sys.argv[1]+'/config.json')))
packet,parents=data.load(tuple(open(p,'rb').read()for p in sys.argv[3:]),json.load(open(sys.argv[2])))
norm=model.fit_normalization(parents,data.fit_normalization)
print(json.dumps(dict(architecture=model.ARCHITECTURE,packet_sha256=policy.sha(policy.canonical(packet)),normalization_sha256=policy.sha(policy.canonical(norm)))))
"""
    observed=subprocess.run([runtime['python'],'-B','-c',probe,str(output),str(slots['splits']),*[str(slots[k])for k in viewslots]],
        cwd=ROOT,env=environment(runtime),capture_output=True,text=True,timeout=180)
    if observed.returncode:raise RuntimeError('Qualified Project data/architecture probe failed: '+observed.stderr[-3000:])
    selected=json.loads(observed.stdout);names=set(old['sources'])|set(extra_sources)
    witness=dict(old,architecture=selected['architecture'],experiment_sha256=sha(output/'config.json'),
        packet_experiment_sha256=selected['packet_sha256'],normalization_sha256=selected['normalization_sha256'],
        sources={n:sha(output/n)for n in sorted(names)},
        inputs={k:dict(bytes=p.stat().st_size,sha256=sha(p))for k,p in slots.items()if k not in ['runtime_manifest','pressure_probe']})
    write(output/'source-witness.json',witness)
    prefix=output.relative_to(ROOT).as_posix()+'/'
    sources=[dict(path=prefix+n,sha256=sha(output/n))for n in sorted(names|{'source-witness.json'})]
    sources.extend(dict(path=prefix+'tuldok_extension_sdk/'+p.name,sha256=sha(p))for p in sorted((output/'tuldok_extension_sdk').glob('*.py')))
    namespace='rheon.pressure.v4.'+arm+'.'+hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:12]
    definitions={role:read(output/(role+'-contribution.json'))for role in ['reader','task','model','trainer','evaluator','view']}
    old_ids={d['id']:namespace+':'+role for role,d in definitions.items()};refs=[]
    for role,value in definitions.items():
        value['id']=namespace+':'+role
        if 'implementation'in value:value['implementation']['source']=prefix+Path(value['implementation']['source']).name
        for dependency in value.get('requires',[]):
            dependency['id']=old_ids.get(dependency['id'],dependency['id'])
            resolved=next(r for r in refs if r['id']==dependency['id']);dependency.update({k:resolved[k]for k in ['version','sha256']})
        path=output/(role+'-contribution.json');write(path,value)
        refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size,id=value['id'],version=value['version'],kind=value['kind']))
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1'))
    from typed_pressure_project import policy as field_policy
    pressure_probe=read(ROOT/'runs/dependency-migration-v1/receipts/typed-pressure-preparation-01/pressure-probe.json')
    manifest=read(slots['splits']);fit_members=[]
    for index,parent in enumerate(manifest['parents']):
        if parent['split']!='train':raise ValueError('Evaluation parent in fitting view')
        for slot,shard in zip(viewslots,manifest['shards']):
            if any(key.startswith(f'p{index:04d}_')for key in shard['keys']):
                fit_members.append(dict(input=slot,member_id=parent['trajectory_id'],sha256=sha(slots[slot]),split='train'))
    for mode in ['train','restore']:
        recipe=read(output/(mode+'-recipe.json'));recipe.update(id=arm+'-'+mode,name=arm+' '+mode,sources=sources,entrypoint=prefix+'evidence.py')
        recipe['arguments']=[dict(source=prefix+Path(r['source']).name)if 'source'in r else r for r in recipe['arguments']]
        if mode=='train':
            recipe['extension_selection']={'selections':{r['kind']:{k:r[k]for k in ['id','version','sha256']}for r in refs if r['kind']!='reader'}}
            original=recipe['model_evidence'];bundle=dict(definitions=definitions,architecture=witness['architecture'],
                physical_context=original['model_specification']['physical_context'],preprocessing_policy=original['preprocessing_policy'],output_policy=original['output_policy'])
            evidence=field_policy(recipe,bundle,pressure_probe,fit_members);evidence['profile']['capture']['interval']=config['training']['updates'];evidence['field_outputs'][0]['clock']=config['training']['updates'];recipe['model_evidence']=evidence
        write(output/(mode+'-recipe.json'),recipe)
    provider=read(output/'provider.json');provider.update(package={'id':namespace,'version':'1.0.0'},label=arm,contributions=refs,
        sources=[dict(r,bytes=(ROOT/r['path']).stat().st_size)for r in sources]);write(output/'provider.json',provider)
    project['recipes']=[dict(id=m,label=m,path=prefix+m+'-recipe.json',sha256=sha(output/(m+'-recipe.json')))for m in ['train','restore']]
    project['extensions']=[dict(id='pressure',label=arm,path=prefix+'provider.json',sha256=sha(output/'provider.json'))];write(output/'project.json',project)
    settings=read(ROOT/'runs/pressure-methods-v3/v3-native-experiment-07.json')
    settings.update(arms=[dict(id=arm,model_id=arm,steps=2,teacher_weight=0.)],
        learning_rate=config['training']['learning_rate'],schedule=config['training']['schedule'],
        training_seed=config['training']['seed'],train_initialization='v3-native')
    experiment=BASE/(arm+'-experiment-01.json');write(experiment,settings)
    policy=read(output/'train-recipe.json')['model_evidence'];assembly=read(output/'assembly.json')
    assembly.update(project=dict(id=project['id'],path=prefix+'project.json',sha256=sha(output/'project.json')),sources=sources,
        configuration_sha256=sha(experiment),initialization_weights_sha256=initial['sha256'],model_specification=policy['model_specification'],
        preprocessing_policy=policy['preprocessing_policy'],output_policy=policy['output_policy'])
    write(output/'assembly.json',assembly)
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/Tuldok'))
    import projects,recipe_contract,extension_contract
    from dependency_config import canonical
    projects.manifest(project);extension_contract.declaration(canonical(provider),'extension')
    for mode in ['train','restore']:recipe_contract.decode_definition(canonical(read(output/(mode+'-recipe.json'))))
    print('Captured typed v4 Project',arm,flush=True)
