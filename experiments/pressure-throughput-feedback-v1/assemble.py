"""Freeze matched finite CUDA studies in current Tuldok Projects, without execution.

Inputs and the earlier consumed probe are explicit. The captured trainer owns
physics and optimization; current Tuldok decoders and SDK own their contracts.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'dependency-migration-v1'))
from dependency_config import canonical, decode, digest, require
from typed_pressure_project import policy as pressure_policy


def write(path, value):
    path.write_bytes(canonical(value))


def assemble(root, template, declarations, probe_path, tuldok, runtime, configuration, output, current_bank=None,
             model_overlay=None, gradient_diagnostic=None, diagnostic_updates=None, full_fit_observer=None):
    root, template, tuldok = [x.resolve(strict=True) for x in (root, template, tuldok)]
    runtime=runtime.resolve(strict=True)
    output = output.absolute()
    require(output.is_relative_to(root) and not output.exists(), 'fresh rooted destination required')
    require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=HERE.parents[1]),
            'commit qualified trainer source before capture')
    config = decode(configuration.read_bytes())
    require(config['schema']in('rheon-water-residual-experiment-v10','rheon-water-residual-experiment-v11'), 'explicit band-feedback configuration required')
    bundle = decode(declarations.read_bytes())
    probe = decode(probe_path.read_bytes())
    require(digest(Path(probe['path']))==probe['sha256'] and probe['consumption'].startswith('already consumed'),
            'previously consumed exact probe required')
    original_project = decode((template/'project.json').read_bytes())
    witness = decode((template/'source-witness.json').read_bytes())
    # Measure the full original material before reusing it. No historical bytes change.
    for name, identity in witness['sources'].items():
        require(digest(template/name)==identity, 'template source changed: '+name)
    for row in original_project['data']:
        require(digest(root/row['path'])==row['sha256'], 'template data changed')
    require(digest(runtime)==witness['runtime_manifest_sha256'], 'runtime differs from qualified template')
    output.mkdir(parents=True)
    prefix=output.relative_to(root).as_posix()+'/'
    for name in witness['sources']:
        shutil.copyfile(template/name,output/name)
    for name in ('model.py','policy.py','train.py','band_feedback.py','gradient_clipping.py'):
        shutil.copyfile(HERE.parent/'water-conditioning-v1/training'/name,output/name)
    if model_overlay is not None:
        overlay = model_overlay.resolve(strict=True)
        require(overlay.is_dir() and {p.name for p in overlay.iterdir()} == {'model.py', 'coarse.py'},
                'explicit two-file model overlay required')
        require(all(p.is_file() and not p.is_symlink() and p.stat().st_size <= 1024**2
                    for p in overlay.iterdir()), 'bounded regular model overlay sources required')
        (output/'model.py').rename(output/'base_model.py')
        for name in ('model.py', 'coarse.py'):
            shutil.copyfile(overlay/name, output/name)
    require((gradient_diagnostic is None) == (diagnostic_updates is None),
            'diagnostic sources and explicit update cap must be selected together')
    require(full_fit_observer is None or gradient_diagnostic is None,
            'select full-fit observer or capped diagnostic, not both')
    training_entry = 'evidence.py'
    if gradient_diagnostic is not None:
        diagnostic = gradient_diagnostic.resolve(strict=True)
        names = {p.name for p in diagnostic.iterdir()}
        historical = {'gradient_screen.py', 'gradient-reference.json'}
        repaired = historical | {'stability_screen.py', 'capture_policy.py', 'baseline_train.py'}
        require(names in (historical, repaired), 'explicit diagnostic or repaired stability sources required')
        require(diagnostic.is_dir() and {p.name for p in diagnostic.iterdir()} == names
                and all(p.is_file() and not p.is_symlink() and p.stat().st_size <= 1024**2
                        for p in diagnostic.iterdir())
                and type(diagnostic_updates) is int and 1 <= diagnostic_updates <= 1500,
                'bounded explicit gradient diagnostic sources and cap required')
        for name in names:
            shutil.copyfile(diagnostic/name, output/name)
        training_entry = 'stability_screen.py' if names == repaired else 'gradient_screen.py'
    if full_fit_observer is not None:
        observer = full_fit_observer.resolve(strict=True)
        names = {'full_fit.py','gradient_screen.py','stability_screen.py','capture_policy.py',
                 'baseline_train.py','gradient-reference.json'}
        require(observer.is_dir() and {p.name for p in observer.iterdir()} == names
                and all(p.is_file() and not p.is_symlink() and p.stat().st_size <= 1024**2
                        for p in observer.iterdir()), 'bounded qualified full-fit observation sources required')
        for name in names:
            shutil.copyfile(observer/name, output/name)
        training_entry = 'full_fit.py'
    shutil.copyfile(tuldok/'recipe_process.py',output/'custody-validator.py')
    witness['custody_validator_sha256']=digest(output/'custody-validator.py')
    shutil.copyfile(HERE/'training_diagnostic.py',output/'training_diagnostic.py')
    shutil.copyfile(HERE/'evidence.py',output/'evidence.py')
    shutil.copyfile(runtime,output/'runtime-manifest.json')
    current_rows=[]
    if config['schema']=='rheon-water-residual-experiment-v11':
        require(current_bank is not None,'explicit current-policy bank required')
        current_bank=current_bank.resolve(strict=True)
        bank=decode((current_bank/'manifest.json').read_bytes())
        require(digest(current_bank/'manifest.json')==config['current_feedback']['manifest_sha256'],
                'selected current-policy manifest differs')
        for slot,path in [('current_manifest',current_bank/'manifest.json'),
                          ('current_dataset',current_bank/(config['current_feedback']['stage']+'.npz'))]:
            path=path.resolve(strict=True)
            require(path.is_relative_to(root),'current input outside artifact root')
            if slot=='current_dataset':
                selected=bank['stages'][config['current_feedback']['stage']]
                require(path.stat().st_size==selected['bytes']and digest(path)==selected['sha256'],
                        'current payload differs from qualified bank')
            witness['inputs'][slot]={'sha256':digest(path),'bytes':path.stat().st_size}
            current_rows.append(dict(id=slot,label='Current-policy '+slot,path=path.relative_to(root).as_posix(),sha256=digest(path)))
    else:
        require(current_bank is None,'no unselected current-policy input')
    write(output/'config.json',config)
    # Use the selected Pumas interpreter; no optimizer is constructed by this probe.
    runtime_doc=decode(runtime.read_bytes())
    bootstrap="import sys,json;sys.path[:0]=[sys.argv[1],sys.argv[2]];import model,policy;model.select(policy.experiment(json.load(open(sys.argv[1]+'/config.json'))));print(json.dumps(model.ARCHITECTURE,sort_keys=True))"
    result=subprocess.run([runtime_doc['python'],'-B','-c',bootstrap,str(output),runtime_doc['site_packages']],
        env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
        capture_output=True,timeout=60,check=True)
    architecture=decode(result.stdout)
    witness.update(architecture=architecture,experiment_sha256=digest(output/'config.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE.parents[1],text=True).strip())
    witness['sources']={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='runtime-manifest.json'}
    write(output/'source-witness.json',witness)
    (output/'tuldok_extension_sdk').mkdir()
    for name in ('__init__.py','evidence.py'):
        shutil.copyfile(tuldok/'tuldok_extension_sdk'/name,output/'tuldok_extension_sdk'/name)
    sources=[{'path':prefix+p.relative_to(output).as_posix(),'sha256':digest(p)}
        for p in sorted(output.rglob('*')) if p.is_file() and p.name!='runtime-manifest.json' and '__pycache__' not in p.parts]
    # Independent provider namespaces avoid replacing a captured historical extension.
    namespace='rheon.pressure.'+config['id'][:24]+'.'+hashlib.sha256(canonical(sources)).hexdigest()[:12]
    definitions=copy.deepcopy(bundle['definitions']);refs=[]
    old_ids={v['id']:namespace+':'+role for role,v in definitions.items()}
    for role,value in definitions.items():
        value['id']=namespace+':'+role
        if 'implementation' in value:
            value['implementation']['source']=prefix+Path(value['implementation']['source']).name
        if role in ('trainer','evaluator'):
            value['implementation']['source']=prefix+(training_entry if role=='trainer' else 'evidence.py')
        for dep in value.get('requires',[]):dep['id']=old_ids.get(dep['id'],dep['id'])
    for role in ('reader','task','model','trainer','evaluator','view'):
        value=definitions[role]
        if role=='trainer':
            reader=next(r for r in refs if r['kind']=='reader');value['requires']=[{k:reader[k] for k in ('id','version','sha256')}]
        file=output/(role+'-contribution.json');write(file,value)
        refs.append(dict(path=prefix+file.name,sha256=digest(file),bytes=file.stat().st_size,
                         id=value['id'],version=value['version'],kind=value['kind']))
    selections={r['kind']:{k:r[k] for k in ('id','version','sha256')} for r in refs if r['kind']!='reader'}
    updates=config['training']['updates'];recipes={}
    for mode in ('train','restore'):
        recipe=copy.deepcopy(bundle['recipes'][mode]);recipe.update(id=config['id']+'-'+mode,name=config['id']+' '+mode,
            entrypoint=prefix+'evidence.py',sources=sources,resources={**config['resources'],'cpu_cores':2 if mode=='train' else 1,
            'memory_bytes':8*1024**3 if mode=='train' else 4*1024**3})
        if mode=='restore':recipe['resources']['wall_seconds']=600
        if current_rows:
            for slot in ('current_dataset','current_manifest'):
                recipe['inputs'][slot]={'max_bytes':32*1024**2 if slot=='current_dataset'else 4*1024**2}
                recipe['arguments'] += [{'literal':'--'+slot.replace('_','-')},{'input':slot}]
        recipe['inputs']['runtime_manifest']={'max_bytes':32*1024**2}
        recipe['outputs']=[r for r in recipe['outputs'] if not r['path'].startswith('checkpoint-') or r['path']=='checkpoint-000.npz']
        recipe['outputs'].append(dict(path=f'checkpoint-{updates:03d}.npz',role='checkpoint',max_bytes=2*1024**2,required=True))
        for arg in recipe['arguments']:
            if 'source' in arg:
                name=Path(arg['source']).name
                arg.clear();arg.update({'input':'runtime_manifest'} if name=='runtime-manifest.json' else {'source':prefix+name})
        if mode=='restore':
            recipe['inputs'].pop('trained_checkpoint_001_npz')
            recipe['inputs'][f'trained_checkpoint_{updates:03d}_npz']={'max_bytes':2*1024**2}
            for arg in recipe['arguments']:
                if arg.get('literal')=='--trained-checkpoint-001-npz':arg['literal']=f'--trained-checkpoint-{updates:03d}-npz'
                if arg.get('input')=='trained_checkpoint_001_npz':arg['input']=f'trained_checkpoint_{updates:03d}_npz'
        else:
            recipe['entrypoint'] = prefix + training_entry
            if gradient_diagnostic is not None:
                recipe['arguments'] += [{'literal':'--gradient-limit'}, {'literal':str(diagnostic_updates)},
                    {'literal':'--gradient-reference'}, {'source':prefix+'gradient-reference.json'}]
            if full_fit_observer is not None:
                recipe['arguments'] += [{'literal':'--fit-reference'}, {'source':prefix+'gradient-reference.json'}]
            recipe.update(schema_version=3,format='external_model_evidence_recipe_v2',extension_selection={'selections':selections})
            recipe['inputs']['pressure_probe']={'max_bytes':1024*1024}
            recipe['arguments'] += [{'literal':'--pressure-probe'},{'input':'pressure_probe'},
                {'literal':'--model-evidence-context'},{'model_evidence_context':True}]
            for row in recipe['outputs']:
                if row['path']=='profile.json':row['path']='training-profile.json'
            names=['execution-context.json','model-evidence.json','model-definition.json','profile.json','probe.json',
                'metric-definitions.json','checkpoints.json','observations.json','field-index.json','fitted-transform.json',
                'output-policy.json',f'checkpoint-{updates}.json','pressure-diagnostic.json',
                'probe-residual.npy','probe-base.npy','pressure-prediction.npy','pressure-reference.npy']
            recipe['outputs'] += [dict(path=n,role='model_context' if n=='execution-context.json' else 'model_evidence',
                max_bytes=65536 if n=='execution-context.json' else 8*1024**2,required=True) for n in names]
            data={r['id']:r for r in original_project['data']};splits=decode((root/data['splits']['path']).read_bytes())
            fit=[dict(input=slot,sha256=data[slot]['sha256'],member_id=p['entry']['trajectory_id'],split='train')
                 for p in splits['parents'] for slot in ('dataset','dataset_extra')]
            policy=pressure_policy(recipe,{**bundle,'definitions':definitions,'architecture':architecture},probe,fit)
            policy['profile']['capture']['interval']=updates;policy['field_outputs'][0]['clock']=updates
            recipe['model_evidence']=policy
        write(output/(mode+'-recipe.json'),recipe);recipes[mode]=recipe
    provider={'format':'tuldok_extension_v1','package':{'id':namespace,'version':'1.0.0'},'label':config['id'],
        'license':{'id':'private-lab-authored','notices':[]},'sources':[dict(r,bytes=(root/r['path']).stat().st_size) for r in sources],
        'host_contracts':[{'id':'tuldok:extension','version':'1.0.0'},{'id':'tuldok:project','version':'2.0.0'}],
        'runtime':{'profile':'managed_torch_python_v1','snapshot_sha256':digest(runtime)},'effects':['filesystem'],
        'resources':{'wall_seconds':config['resources']['wall_seconds'],'source_bytes':8*1024**2,'output_bytes':config['resources']['max_output_bytes']},'contributions':refs}
    write(output/'provider.json',provider)
    project=copy.deepcopy(original_project);project.update(id=uuid.uuid4().hex,name=config['id'])
    if gradient_diagnostic is not None:
        project['name'] += (' clipping stability (cap ' if training_entry == 'stability_screen.py' else ' gradient diagnostic (cap ')+str(diagnostic_updates)+')'
    if full_fit_observer is not None:
        project['name'] += ' full5000 clipping-qualified fit'
    project['data'] += current_rows+[dict(id='runtime_manifest',label='Pumas runtime',path=runtime.relative_to(root).as_posix(),sha256=digest(runtime)),
        dict(id='pressure_probe',label='Consumed development probe',path=Path(probe['path']).relative_to(root).as_posix(),sha256=probe['sha256'])]
    project['recipes']=[dict(id=mode,label=mode,path=prefix+mode+'-recipe.json',sha256=digest(output/(mode+'-recipe.json'))) for mode in recipes]
    project['extensions']=[dict(id='pressure',label=config['id'],path=prefix+'provider.json',sha256=digest(output/'provider.json'))]
    write(output/'project.json',project)
    sys.path.insert(0,str(tuldok));import projects,recipe_contract,extension_contract
    projects.manifest(project)
    extension_contract.declaration(canonical(provider),'extension')
    for value in definitions.values():extension_contract.declaration(canonical(value),'contribution')
    for recipe in recipes.values():recipe_contract.decode_definition(canonical(recipe))
    receipt=dict(format='rheon_pressure_training_assembly_v1',project=dict(id=project['id'],path=prefix+'project.json',sha256=digest(output/'project.json')),
        runtime_manifest_sha256=digest(runtime),configuration_sha256=digest(configuration),updates=updates,band_feedback=config['band_feedback'],
        model_specification=recipes['train']['model_evidence']['model_specification'],preprocessing_policy=bundle['preprocessing_policy'],output_policy=bundle['output_policy'],
        limits=dict(attempt_slots=1 if gradient_diagnostic is not None or full_fit_observer is not None else 2,
                    admitted_wall_seconds=2*config['resources']['wall_seconds'],
                    retained_bytes=536870912,peak_staging_bytes=1073741824),
        sources=sources,template=str(template),optimizer_updates_executed=0,model_promotion=False)
    write(output/'assembly.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','template','declarations','probe','tuldok','runtime','configuration','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--current-bank',type=Path)
    p.add_argument('--model-overlay',type=Path)
    p.add_argument('--gradient-diagnostic',type=Path)
    p.add_argument('--diagnostic-updates',type=int)
    p.add_argument('--full-fit-observer',type=Path)
    a=p.parse_args();r=assemble(a.root,a.template,a.declarations,a.probe,a.tuldok,a.runtime,a.configuration,a.output,a.current_bank,a.model_overlay,a.gradient_diagnostic,a.diagnostic_updates,a.full_fit_observer)
    print(json.dumps({'project':r['project'],'updates':r['updates']}))

if __name__=='__main__':main()
