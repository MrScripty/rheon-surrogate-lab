"""Freeze native-size and reduced-width Projects within ordinary managed limits."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha,environment

def change(text,before,after):
    if text.count(before)!=1: raise ValueError('Training source seam changed: '+before[:100])
    return text.replace(before,after)

def prepare(arm):
    base=ROOT/'runs/pressure-methods-v3'; output=base/(arm+'-project-07')
    if output.exists(): raise ValueError('Fresh Project required')
    reduced=arm in ['v3-c37half','v3-c44half']
    prior=ROOT/'runs/pressure-methods-v2'/('v2-num6-project-02' if reduced else 'v2-headshape-project-02')
    old_witness=read(prior/'source-witness.json')
    for name,expected in old_witness['sources'].items():
        if sha(prior/name)!=expected: raise ValueError('Prior qualified source changed')
    shutil.copytree(prior,output,ignore=shutil.ignore_patterns('__pycache__'))
    helper=output/'original_train.py'
    old_helper_sha=sha(helper)
    helper.write_text(helper.read_text().replace('4 * 1024**3','12 * 1024**3').replace('exceeds 4 GiB','exceeds 12 GiB'))
    pinned=output/'frozen_gn.py'
    pinned.write_text(change(pinned.read_text(),old_helper_sha,sha(helper)))
    view=base/('replay-training-view-01' if arm=='v3-replay' else 'derived-training-view-01')
    manifest=read(view/'manifest.json'); extra_ids=[f'view_{i:02d}' for i in range(2,len(manifest['shards']))]
    initialization=base/('C37-half-initialization-01' if arm=='v3-c37half' else 'C44-half-initialization-01' if arm=='v3-c44half' else 'head-continuation-initialization-01')
    initial=read(initialization/'initialization.json')
    if initial['status']!='qualified_initialization' or sha(initialization/'weights.npz')!=initial['weights_sha256']:
        raise ValueError('Qualified exact initialization required')
    config=read(output/'config.json');config['id']=arm;config['training']['seed']=2026101902
    config['resources']['max_output_bytes']=64*1024**2
    write(output/'config.json',config)
    objective=read(output/'fine-tuning.json');objective.update(arm=arm,initial_weights_sha256=initial['weights_sha256'],initial_inventory=initial['inventory'])
    write(output/'fine-tuning.json',objective)
    # Explicit new per-Project CPU/CUDA field ceiling. Original adapters stay immutable.
    for name in ['policy.py','portable_policy.py']:
        text=(output/name).read_text().replace('MAX_CELLS=262144','MAX_CELLS=2097152').replace('MAX_CELLS = 262144','MAX_CELLS = 2097152')
        if name=='portable_policy.py': text=change(text,"resources['max_output_bytes']<=32*MIB","resources['max_output_bytes']<=64*MIB")
        (output/name).write_text(text)
    reader=(ROOT/'experiments/pressure-methods-v1/view.py').read_text()
    reader=reader.replace('rheon-pressure-methods-learning-view-v1','rheon-pressure-methods-learning-view-v3').replace('262144','2097152')
    reader=reader.replace('len(raw_shards) != 2 or len(manifest[\'shards\']) != 2','not 2 <= len(raw_shards) <= 18 or len(raw_shards) != len(manifest[\'shards\'])')
    reader=reader.replace('1 <= n <= 64','1 <= n <= 128').replace('64*1024**2','768*1024**2').replace('64 MiB total','768 MiB total').replace('32*1024**2','40*1024**2')
    derived=bool(manifest.get('derived_fields'))
    if derived:
        reader=change(reader,'    if set(arrays) != set(inventory):','    import native_reconstruct\n    arrays = native_reconstruct.complete(arrays,manifest)\n    if set(arrays) != set(inventory):')
        shutil.copyfile(HERE/'native_reconstruct.py',output/'native_reconstruct.py')
        shutil.copyfile(ROOT/'experiments/pressure-methods-v1/numerics.py',output/'reference_spectral.py')
    (output/'methods_view.py').write_text(reader)
    data=(output/'data.py').read_text().replace('rheon-pressure-methods-learning-view-v1','rheon-pressure-methods-learning-view-v3').replace('max_cells=262144','max_cells=2097152')
    (output/'data.py').write_text(data)
    shutil.copyfile(HERE/'order.py',output/'mixed_order.py')
    if reduced:
        shutil.copyfile(HERE/'half_model.py',output/'model.py')
        slicing=(HERE/'frozen_matrix.py').read_text();start=slicing.index('def half_width(');end=slicing.index('\ndef learned_branch(',start)
        (output/'channel_slice.py').write_text('import copy\nimport hashlib\n'+slicing[start:end])
        coarse=(output/'coarse.py').read_text().replace('> 64','> 128').replace('> 262144','> 2097152')
        (output/'coarse.py').write_text(coarse)
    else:
        text=(output/'model.py').read_text().replace("fine_initialization='Original C53 bytes preserved'","fine_initialization='Exact completed v2-headshape CPU restore; matched continuation control'")
        (output/'model.py').write_text(text)
    trainer=(output/'train.py').read_text()
    start=trainer.index('def input_slots(config):');end=trainer.index('\ndef outputs(',start)
    trainer=trainer[:start]+f"VIEW_SLOTS = {extra_ids!r}\n\ndef input_slots(config):\n    return {{**DATA,'initial_weights':'initial.npz',**{{k:k+'.npz' for k in VIEW_SLOTS}}}}\n\n"+trainer[end:]
    trainer=trainer.replace("'predictions.npz': ('predictions', 16*MIB)","'predictions.npz': ('predictions', 8*MIB)")
    trainer=change(trainer,'def predictions_on_device(network, parents, normalization, population):',
        'def publication_arrays(predicted):\n    """Sampled diagnostic export only; full-field hashes remain authoritative."""\n    return {key:value[tuple(slice(None,None,max(1,math.ceil(n/32))) for n in value.shape)] for key,value in predicted.items()}\n\n\ndef predictions_on_device(network, parents, normalization, population):')
    trainer=change(trainer,'np.savez(stream, **predicted)','np.savez(stream, **publication_arrays(predicted))')
    trainer=change(trainer,"helpers.read_npz(raw_trained['trained_predictions_npz'], prediction_inventory, 16*MIB)",
        "helpers.read_npz(raw_trained['trained_predictions_npz'], data.inventory(publication_arrays(predicted)), 8*MIB)")
    trainer=change(trainer,"'per_case_inference_s': clocks, 'outputs': artifact_specs, 'originating_training_run': origin,",
        "'per_case_inference_s': clocks, 'outputs': artifact_specs, 'originating_training_run': origin,\n               'prediction_export_inventory': data.inventory(publication_arrays(predicted)),\n               'prediction_export_policy': 'NPZ samples at most32cells/axis; full prediction_inventory hashes every full-resolution value and is checked exactly on fresh CPU restore',")
    trainer=trainer.replace("else 32*MIB) for slot in input_slots(config)","else 40*MIB) for slot in input_slots(config)")
    trainer=change(trainer,"data.load((inputs['dataset'], inputs['dataset_extra']), manifest)","data.load(tuple(inputs[k] for k in ['dataset','dataset_extra',*VIEW_SLOTS]), manifest)")
    trainer=change(trainer,"    for slot in set(input_slots(config))-set(DATA):\n        parser.add_argument('--'+slot.replace('_', '-'), dest=slot, type=Path, required=True)","    parser.add_argument('--initial-weights',type=Path,required=True)\n    if VIEW_SLOTS: parser.add_argument('--view-shards',type=Path,nargs=len(VIEW_SLOTS),required=True)")
    trainer=change(trainer,'    args = parser.parse_args()','    args = parser.parse_args()\n    for slot,path in zip(VIEW_SLOTS,getattr(args,\'view_shards\',[])): setattr(args,slot,path)')
    (output/'train.py').write_text(trainer)
    project=read(output/'project.json');project.update(id=uuid.uuid4().hex,name='Pressure methods v3 · '+arm)
    disabled={'direction_dataset','direction_manifest','current_dataset','current_manifest'}
    project['data']=[r for r in project['data'] if r['id'] not in disabled]
    replacements=dict(dataset=view/manifest['shards'][0]['path'],dataset_extra=view/manifest['shards'][1]['path'],splits=view/'manifest.json',initial_weights=initialization/'weights.npz')
    for slot,shard in zip(extra_ids,manifest['shards'][2:]):
        replacements[slot]=view/shard['path'];project['data'].append(dict(id=slot,label=slot,path='',sha256=''))
    for row in project['data']:
        if row['id'] in replacements:
            p=replacements[row['id']];row.update(path=str(p.relative_to(ROOT)),sha256=sha(p),label=arm+' '+row['id'])
    write(output/'project.json',project)
    slots={r['id']:ROOT/r['path']for r in project['data']}
    runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    probe="""import sys,json;sys.path.insert(0,sys.argv[1]);import model,data,policy
model.configure('cpu');model.select(json.load(open(sys.argv[1]+'/config.json')))
packet,parents=data.load(tuple(open(p,'rb').read()for p in sys.argv[3:]),json.load(open(sys.argv[2])))
norm=model.fit_normalization(parents,data.fit_normalization)
print(json.dumps(dict(architecture=model.ARCHITECTURE,packet_sha256=policy.sha(policy.canonical(packet)),normalization_sha256=policy.sha(policy.canonical(norm)))))
"""
    raw=subprocess.run([runtime['python'],'-B','-c',probe,str(output),str(slots['splits']),*[str(slots[k])for k in ['dataset','dataset_extra',*extra_ids]]],env=environment(runtime),capture_output=True,text=True,timeout=180)
    if raw.returncode: raise RuntimeError('Training view/architecture probe failed: '+raw.stderr[-2500:])
    selected=json.loads(raw.stdout)
    names=set(old_witness['sources'])|({'channel_slice.py'}if reduced else set())|({'native_reconstruct.py','reference_spectral.py'}if derived else set())
    witness=dict(old_witness,architecture=selected['architecture'],experiment_sha256=sha(output/'config.json'),
        packet_experiment_sha256=selected['packet_sha256'],normalization_sha256=selected['normalization_sha256'],
        sources={n:sha(output/n)for n in sorted(names)},inputs={k:dict(bytes=p.stat().st_size,sha256=sha(p))for k,p in slots.items()if k not in ['runtime_manifest','pressure_probe']})
    write(output/'source-witness.json',witness)
    prefix=output.relative_to(ROOT).as_posix()+'/'
    sources=[dict(path=prefix+n,sha256=sha(output/n))for n in sorted(names|{'source-witness.json'})]
    sources.extend(dict(path=prefix+'tuldok_extension_sdk/'+p.name,sha256=sha(p))for p in sorted((output/'tuldok_extension_sdk').glob('*.py')))
    namespace='rheon.pressure.v3.'+arm+'.'+hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:12]
    definitions={role:read(output/(role+'-contribution.json'))for role in ['reader','task','model','trainer','evaluator','view']}
    old_ids={v['id']:namespace+':'+role for role,v in definitions.items()}; refs=[]
    for role,value in definitions.items():
        value['id']=namespace+':'+role
        if 'implementation'in value:value['implementation']['source']=prefix+Path(value['implementation']['source']).name
        for dependency in value.get('requires',[]):
            dependency['id']=old_ids.get(dependency['id'],dependency['id'])
            resolved=next(r for r in refs if r['id']==dependency['id'])
            dependency.update({k:resolved[k]for k in ['version','sha256']})
        path=output/(role+'-contribution.json');write(path,value)
        refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size,id=value['id'],version=value['version'],kind=value['kind']))
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1'))
    from typed_pressure_project import policy as field_policy
    pressure_probe=read(ROOT/'runs/dependency-migration-v1/receipts/typed-pressure-preparation-01/pressure-probe.json')
    fit_members=[]
    for index,parent in enumerate(manifest['parents']):
        for slot,shard in zip(['dataset','dataset_extra',*extra_ids],manifest['shards']):
            if any(key.startswith(f'p{index:04d}_')for key in shard['keys']):
                fit_members.append(dict(input=slot,member_id=parent['trajectory_id'],sha256=sha(slots[slot]),split='train'))
    for mode in ['train','restore']:
        recipe=read(prior/(mode+'-recipe.json'));recipe.update(id=arm+'-'+mode,name=arm+' '+mode,sources=sources,entrypoint=prefix+'evidence.py')
        recipe['inputs']={k:v for k,v in recipe['inputs'].items()if k not in disabled}
        for slot in ['dataset','dataset_extra',*extra_ids]:recipe['inputs'][slot]={'max_bytes':40*1024**2}
        arguments=recipe['arguments'];filtered=[];index=0
        while index<len(arguments):
            item=arguments[index]
            if item.get('literal') in {'--'+k.replace('_','-')for k in disabled}:index+=2;continue
            if 'source'in item:item=dict(source=prefix+Path(item['source']).name)
            filtered.append(item);index+=1
        if extra_ids:filtered +=[{'literal':'--view-shards'},*[{'input':k}for k in extra_ids]]
        recipe['arguments']=filtered;recipe['resources'].update(max_output_bytes=64*1024**2,memory_bytes=16*1024**3)
        for row in recipe['outputs']:
            if row['path']=='predictions.npz':row['max_bytes']=8*1024**2
        if mode=='restore':recipe['inputs']['trained_predictions_npz']['max_bytes']=8*1024**2
        else:
            recipe['extension_selection']={'selections':{r['kind']:{k:r[k]for k in ['id','version','sha256']}for r in refs if r['kind']!='reader'}}
            original=recipe['model_evidence'];bundle=dict(definitions=definitions,architecture=witness['architecture'],physical_context=original['model_specification']['physical_context'],preprocessing_policy=original['preprocessing_policy'],output_policy=original['output_policy'])
            evidence=field_policy(recipe,bundle,pressure_probe,fit_members);evidence['profile']['capture']['interval']=2400;evidence['field_outputs'][0]['clock']=2400;recipe['model_evidence']=evidence
        write(output/(mode+'-recipe.json'),recipe)
    provider=read(output/'provider.json');provider.update(package={'id':namespace,'version':'1.0.0'},label=arm,contributions=refs,sources=[dict(r,bytes=(ROOT/r['path']).stat().st_size)for r in sources]);write(output/'provider.json',provider)
    provider['resources']['output_bytes']=64*1024**2
    write(output/'provider.json',provider)
    project['recipes']=[dict(id=m,label=m,path=prefix+m+'-recipe.json',sha256=sha(output/(m+'-recipe.json')))for m in ['train','restore']]
    project['extensions']=[dict(id='pressure',label=arm,path=prefix+'provider.json',sha256=sha(output/'provider.json'))];write(output/'project.json',project)
    policy=read(output/'train-recipe.json')['model_evidence'];assembly=read(prior/'assembly.json')
    experiment=base/(arm+'-experiment-07.json');settings=read(ROOT/'runs/pressure-methods-v2/v2-headshape-experiment-02.json');settings.update(arms=[dict(id=arm,model_id=arm,steps=2,teacher_weight=0.)],training_seed=2026101902);write(experiment,settings)
    assembly.update(project=dict(id=project['id'],path=prefix+'project.json',sha256=sha(output/'project.json')),sources=sources,configuration_sha256=sha(experiment),initialization_weights_sha256=initial['weights_sha256'],model_specification=policy['model_specification'],preprocessing_policy=policy['preprocessing_policy'],output_policy=policy['output_policy'])
    write(output/'assembly.json',assembly)
    write(output/'domain-assembly.json',dict(status='prepared_not_admitted',cohort=3 if reduced else 5,arm=arm,fitting_parents=[p['trajectory_id']for p in manifest['parents']],source_sha256=sha(__file__),optimizer_updates_executed=0))
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/Tuldok'))
    import projects,recipe_contract,extension_contract
    from dependency_config import canonical
    projects.manifest(project);extension_contract.declaration(canonical(provider),'extension')
    for mode in ['train','restore']:recipe_contract.decode_definition(canonical(read(output/(mode+'-recipe.json'))))
    print('Prepared ordinary managed Project',arm,'shards',len(manifest['shards']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['v3-replay','v3-native','v3-c37half','v3-c44half'],required=True)
    prepare(parser.parse_args().arm)
