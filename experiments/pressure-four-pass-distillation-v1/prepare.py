"""Freeze visible Tuldok fine-tuning projects over measured historical sources."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from common import ROOT, HERE, environment, read, sha, write


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Captured trainer seam changed: ' + before[:80])
    return text.replace(before, after, 1)


def prepare(config_path, arm_id, destination):
    config = read(config_path)
    material = read(ROOT / config['baseline_material'])
    selected = next(m for m in material['models'] if m['id'] == config['train_initialization'])
    for path, identity in selected['identities'].items():
        if sha(ROOT / path) != identity:
            raise ValueError('Baseline identity changed: ' + path)
    template = ROOT / selected['project']
    artifacts = ROOT / selected['restore']
    witness = read(template / 'source-witness.json')
    project = read(template / 'project.json')
    for name, identity in witness['sources'].items():
        if sha(template / name) != identity:
            raise ValueError('Captured template source changed: ' + name)
    for row in project['data']:
        if sha(ROOT / row['path']) != row['sha256']:
            raise ValueError('Captured template data changed: ' + row['id'])
    arm = next(a for a in config['arms'] if a['id'] == arm_id)
    if destination.exists() or not destination.is_relative_to(ROOT):
        raise ValueError('Fresh rooted project required')
    shutil.copytree(template, destination, ignore=shutil.ignore_patterns('__pycache__'))
    prefix = destination.relative_to(ROOT).as_posix() + '/'
    qualified = ROOT / 'runs/pressure-coarse-full-fit-v1/project-01'
    qualified_witness = read(qualified / 'source-witness.json')
    for name in ['train.py', 'gradient_clipping.py']:
        if sha(qualified / name) != qualified_witness['sources'][name]:
            raise ValueError('Qualified clipping/trainer source changed')
        shutil.copyfile(qualified / name, destination / name)
    shutil.copyfile(template / 'model.py', destination / 'baseline_model.py')
    for name in ['model.py', 'fine_tuning.py']:
        shutil.copyfile(HERE / name, destination / name)
    history = read(artifacts / 'history.json')
    baseline_config = read(template / 'config.json')
    inventory = history['checkpoints'][f"checkpoint-{baseline_config['training']['updates']:03d}.npz"]['inventory']
    objective = dict(schema='rheon-fine-tuning-objective-v1', arm=arm_id,
                     initial_weights_sha256=sha(artifacts / 'weights.npz'), initial_inventory=inventory,
                     training_passes=arm['steps'], teacher_passes=config['teacher_passes'],
                     teacher_weight=arm['teacher_weight'], early_weight=config['early_feedback_weight'],
                     relative_equation_target=config['target_relative_residual'], teacher_cache_bytes=config['teacher_cache_bytes'])
    write(destination / 'fine-tuning.json', objective)
    captured = copy.deepcopy(baseline_config)
    captured['id'] = 'fourpass-' + arm_id
    captured['training'].update(updates=config['updates'], checkpoints=[0, config['updates']],
                                seed=config['training_seed'], sampling_seed=config['sampling_seed'],
                                learning_rate=config['learning_rate'], schedule=config['schedule'])
    captured['resources']['wall_seconds'] = config['wall_seconds']
    write(destination / 'config.json', captured)
    trainer = (destination / 'train.py').read_text()
    trainer = replace_once(trainer, "return {**DATA,**direction_feedback.SLOTS,'current_dataset':'current.npz','current_manifest':'current.json'}",
                           "return {**DATA,**direction_feedback.SLOTS,'current_dataset':'current.npz','current_manifest':'current.json','initial_weights':'initial.npz'}")
    trainer = replace_once(trainer, 'network = model.PressureCorrectionNet().to(device); initial_inventory = helpers.state_inventory(network)',
                           'network = model.PressureCorrectionNet().to(device)\n    import fine_tuning\n    fine_tuning.initialize(network, args.initial_weights, helpers)\n    initial_inventory = helpers.state_inventory(network)')
    trainer = replace_once(trainer, "write(output/'receipt.json', receipt)",
                           "receipt['fine_tuning_statistics'] = dict(fine_tuning.STATS)\n    write(output/'receipt.json', receipt)")
    (destination / 'train.py').write_text(trainer)
    runtime = read(ROOT / config['runtime'])
    bootstrap = "import sys,json;sys.path.insert(0,sys.argv[1]);import model;model.select(json.load(open(sys.argv[1]+'/config.json')));print(json.dumps(model.ARCHITECTURE,sort_keys=True))"
    result = subprocess.run([runtime['python'], '-B', '-c', bootstrap, str(destination)],
                            env=environment(runtime), capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Architecture probe failed: ' + result.stderr[-2000:])
    architecture = json.loads(result.stdout)
    names = set(witness['sources']) | {'baseline_model.py', 'fine_tuning.py', 'fine-tuning.json', 'gradient_clipping.py'}
    witness.update(architecture=architecture, experiment_sha256=sha(destination / 'config.json'),
                   sources={name:sha(destination/name) for name in sorted(names)})
    witness['inputs']['initial_weights'] = dict(bytes=(artifacts/'weights.npz').stat().st_size, sha256=sha(artifacts/'weights.npz'))
    write(destination / 'source-witness.json', witness)
    source_names = names | {'source-witness.json'}
    sources = [dict(path=prefix+name, sha256=sha(destination/name)) for name in sorted(source_names)]
    sources += [dict(path=prefix+'tuldok_extension_sdk/'+p.name, sha256=sha(p))
                for p in sorted((destination/'tuldok_extension_sdk').glob('*.py'))]
    namespace = 'rheon.pressure.fourpass.' + arm_id + '.' + hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:12]
    definitions = {role:read(destination/(role+'-contribution.json')) for role in ['reader','task','model','trainer','evaluator','view']}
    old_ids = {value['id']:namespace+':'+role for role,value in definitions.items()}
    refs = []
    for role, value in definitions.items():
        value['id'] = namespace+':'+role
        if 'implementation' in value:
            value['implementation']['source'] = prefix + Path(value['implementation']['source']).name
        for dependency in value.get('requires', []):
            dependency['id'] = old_ids.get(dependency['id'], dependency['id'])
        if role == 'trainer':
            reader = next(r for r in refs if r['kind']=='reader')
            value['requires'] = [{k:reader[k] for k in ['id','version','sha256']}]
        path = destination/(role+'-contribution.json')
        write(path,value)
        refs.append(dict(path=prefix+path.name, sha256=sha(path), bytes=path.stat().st_size,
                         id=value['id'], version=value['version'], kind=value['kind']))
    selection = {r['kind']:{k:r[k] for k in ['id','version','sha256']} for r in refs if r['kind']!='reader'}
    sys.path.insert(0,str(ROOT/config['source_lab']/'experiments/dependency-migration-v1'))
    from typed_pressure_project import policy as field_policy
    probe = read(ROOT/'runs/dependency-migration-v1/receipts/typed-pressure-preparation-01/pressure-probe.json')
    original_policy = read(template/'train-recipe.json')['model_evidence']
    bundle = dict(definitions=definitions, architecture=architecture, physical_context=original_policy['model_specification']['physical_context'],
                  preprocessing_policy=original_policy['preprocessing_policy'], output_policy=original_policy['output_policy'])
    for mode in ['train','restore']:
        recipe = read(template/(mode+'-recipe.json'))
        recipe.update(id=captured['id']+'-'+mode, name=captured['id']+' '+mode, entrypoint=prefix+'evidence.py', sources=sources)
        recipe['inputs']['initial_weights'] = {'max_bytes':2*1024**2}
        recipe['arguments'] += [{'literal':'--initial-weights'},{'input':'initial_weights'}]
        for argument in recipe['arguments']:
            if 'source' in argument:
                argument['source'] = prefix + Path(argument['source']).name
            for key in ['literal','input']:
                if key in argument and isinstance(argument[key],str):
                    argument[key] = argument[key].replace('checkpoint-5000','checkpoint-'+str(config['updates'])).replace('checkpoint_5000','checkpoint_'+str(config['updates']))
        if mode == 'restore':
            recipe['inputs']['trained_checkpoint_'+str(config['updates'])+'_npz'] = recipe['inputs'].pop('trained_checkpoint_5000_npz')
        for row in recipe['outputs']:
            row['path'] = row['path'].replace('checkpoint-5000','checkpoint-'+str(config['updates']))
        if mode == 'train':
            recipe['extension_selection'] = {'selections':selection}
            policy = field_policy(recipe,bundle,probe,original_policy['fit_members'])
            policy['profile']['capture']['interval'] = config['updates']
            policy['field_outputs'][0]['clock'] = config['updates']
            recipe['model_evidence'] = policy
        write(destination/(mode+'-recipe.json'),recipe)
    provider = read(template/'provider.json')
    provider.update(package={'id':namespace,'version':'1.0.0'},label=captured['id'],contributions=refs,
                    sources=[dict(row,bytes=(ROOT/row['path']).stat().st_size)for row in sources])
    write(destination/'provider.json',provider)
    project.update(id=uuid.uuid4().hex,name='Pressure four-pass · '+arm_id+' · C53 fine-tune')
    project['data'].append(dict(id='initial_weights',label='Verified C53 initialization',path=selected['restore']+'/weights.npz',sha256=sha(artifacts/'weights.npz')))
    project['recipes'] = [dict(id=mode,label=mode,path=prefix+mode+'-recipe.json',sha256=sha(destination/(mode+'-recipe.json')))for mode in ['train','restore']]
    project['extensions'] = [dict(id='pressure',label=captured['id'],path=prefix+'provider.json',sha256=sha(destination/'provider.json'))]
    write(destination/'project.json',project)
    tuldok = ROOT/'runs/dependency-migration-v1/Tuldok'
    sys.path.insert(0,str(tuldok))
    import projects, recipe_contract, extension_contract
    from dependency_config import canonical
    projects.manifest(project)
    extension_contract.declaration(canonical(provider),'extension')
    for value in definitions.values():
        extension_contract.declaration(canonical(value),'contribution')
    for mode in ['train','restore']:
        recipe_contract.decode_definition(canonical(read(destination/(mode+'-recipe.json'))))
    policy = read(destination/'train-recipe.json')['model_evidence']
    assembly = dict(format='rheon_pressure_training_assembly_v1', project=dict(id=project['id'],path=prefix+'project.json',sha256=sha(destination/'project.json')),
                    updates=config['updates'], band_feedback=captured['band_feedback'], sources=sources,
                    model_specification=policy['model_specification'],preprocessing_policy=policy['preprocessing_policy'],output_policy=policy['output_policy'],
                    limits=dict(attempt_slots=1,admitted_wall_seconds=2*config['wall_seconds'],retained_bytes=536870912,peak_staging_bytes=1073741824),
                    optimizer_updates_executed=0,model_promotion=False,configuration_sha256=sha(config_path),
                    runtime_manifest_sha256=sha(ROOT/config['runtime']),initialization_weights_sha256=selected['weights_sha256'])
    write(destination/'assembly.json',assembly)
    print(json.dumps(dict(status='prepared',arm=arm_id,project=assembly['project'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--arm',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.config.resolve(),args.arm,args.output.resolve())
