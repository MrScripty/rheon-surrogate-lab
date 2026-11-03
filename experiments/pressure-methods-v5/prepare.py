"""Capture three matched tiny-smoother Projects with ordinary managed custody."""
import argparse
import json
import re
import shutil
import subprocess
from v5_support import ROOT,HERE,BASE,DOCS,read,write,sha,environment
from package_project import package

def prepare(arm,generation):
    output=BASE/(arm+f'-project-{generation:02d}')
    if output.exists():raise ValueError('Fresh typed Project required')
    prior=ROOT/'runs/pressure-methods-v3/v3-native-project-07'
    for name,expected in read(prior/'source-witness.json')['sources'].items():
        if sha(prior/name)!=expected:raise ValueError('Completed control source changed')
    shutil.copytree(prior,output,ignore=shutil.ignore_patterns('__pycache__'))
    for source,target in [('tiny_model.py','model.py'),('tiny_objective.py','fine_tuning.py'),('conditioned_data.py','data.py'),('training_reconstruct.py','native_reconstruct.py')]:shutil.copyfile(HERE/source,output/target)
    shutil.copyfile(ROOT/'experiments/pressure-methods-v4/multilevel.py',output/'training_multilevel.py')
    campaign=read(DOCS/'campaign.json');config=read(output/'config.json')
    config['id']=arm;config['training'].update(seed=campaign['shared_seed'],learning_rate=campaign['learning_rate'],schedule=campaign['schedule'])
    write(output/'config.json',config)
    objective=dict(schema='rheon-multilevel-smoother-objective-v1',arm=arm,coarse_training_cycles=0 if arm=='v5-raw'else 1,
        initial_weights_sha256='',initial_inventory={},training_local_corrections=1,relative_equation_target=1e-6)
    write(output/'fine-tuning.json',objective)
    seed=BASE/'tiny-initialization-01';runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    if not seed.exists():
        seed.mkdir()
        observed=subprocess.run([runtime['python'],'-B',str(HERE/'seed_probe.py'),str(output),str(seed/'weights.npz')],env=environment(runtime),cwd=ROOT,capture_output=True,text=True,timeout=90)
        if observed.returncode:raise RuntimeError('Seed preparation failed: '+observed.stderr[-3000:])
        inventory=json.loads(observed.stdout);write(seed/'initialization.json',dict(weights_sha256=sha(seed/'weights.npz'),inventory=inventory,
            parameters=sum(s['bytes']//4 for s in inventory.values()),source_sha256=sha(HERE/'tiny_model.py'),seed=campaign['shared_seed']))
    initial=read(seed/'initialization.json')
    if initial['parameters']!=648 or sha(seed/'weights.npz')!=initial['weights_sha256']:raise ValueError('Shared tiny initialization changed')
    objective.update(initial_weights_sha256=initial['weights_sha256'],initial_inventory=initial['inventory']);write(output/'fine-tuning.json',objective)
    (output/'head_optimizer.py').write_text('"""One complete disjoint tiny-model parameter group."""\nimport math\ndef groups(network,learning_rate):\n    if not math.isfinite(learning_rate) or learning_rate<=0:raise ValueError("Positive finite rate required")\n    return [dict(params=list(network.parameters()),lr=learning_rate,lr_multiplier=1.)]\n')
    view=BASE/'spectrum-training-view-01'if arm=='v5-spectrum'else ROOT/'runs/pressure-methods-v3/derived-training-view-01'
    manifest=read(view/'manifest.json');extras=[f'view_{i:02d}'for i in range(2,len(manifest['shards']))];slots=['dataset','dataset_extra',*extras]
    trainer=(output/'train.py').read_text();trainer,count=re.subn(r'^VIEW_SLOTS = .*$',f'VIEW_SLOTS = {extras!r}',trainer,count=1,flags=re.MULTILINE)
    if count!=1:raise ValueError('View slots source seam changed')
    (output/'train.py').write_text(trainer)
    project=read(output/'project.json');project['data']=[r for r in project['data']if r['id']not in ['dataset','dataset_extra','splits','initial_weights']and not r['id'].startswith('view_')]
    for slot,shard in zip(slots,manifest['shards']):
        p=view/shard['path'];project['data'].append(dict(id=slot,label=arm+' '+slot,path=str(p.relative_to(ROOT)),sha256=sha(p)))
    for slot,p in [('splits',view/'manifest.json'),('initial_weights',seed/'weights.npz')]:project['data'].append(dict(id=slot,label=arm+' '+slot,path=str(p.relative_to(ROOT)),sha256=sha(p)))
    write(output/'project.json',project)
    for mode in ['train','restore']:
        recipe=read(output/(mode+'-recipe.json'));recipe['inputs']={k:v for k,v in recipe['inputs'].items()if not k.startswith('view_')}
        recipe['inputs'].update({k:dict(max_bytes=40*1024**2)for k in extras})
        arguments=recipe['arguments'];start=next(i for i,r in enumerate(arguments)if r.get('literal')=='--view-shards');end=start+1
        while end<len(arguments)and arguments[end].get('input','').startswith('view_'):end+=1
        recipe['arguments']=arguments[:start]+[dict(literal='--view-shards'),*[dict(input=k)for k in extras]]+arguments[end:]
        if mode=='train':
            recipe['model_evidence']['output_policy'].update(correction=dict(activation='linear',steps=1),output_scale='laplacian-rhs-rms',
                reconstruction='Float32 raw times fixed train-only Laplacian RHS RMS; cell-zero gauge; add supplied base once.')
        write(output/(mode+'-recipe.json'),recipe)
    package(output,arm,['training_multilevel.py'])
    print('Captured matched tiny-smoother Project',arm,flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['v5-raw','v5-multilevel','v5-spectrum'],required=True);parser.add_argument('--generation',type=int,default=2);args=parser.parse_args();prepare(args.arm,args.generation)
