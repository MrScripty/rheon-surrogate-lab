"""Capture C107 continuations as new typed CUDA Projects with CPU restores."""
import argparse
from pathlib import Path
import shutil
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha
from package_pressure import package

def prepare(arm,seed=2026101907,generation=1,project_generation=1):
    registration=read(DOCS/'registration.json');spec=registration['track_a']
    if arm not in spec['arms']:raise ValueError('Registered continuation arm required')
    identifier=arm if generation==1 else arm+f'-s{generation}'
    output=BASE/(identifier+f'-project-{project_generation:02d}')
    if output.exists():raise ValueError('Fresh captured Project required')
    prior=ROOT/'runs/pressure-methods-v4/v4-student50-control-project-01'
    for name,digest in read(prior/'source-witness.json')['sources'].items():
        if sha(prior/name)!=digest:raise ValueError('C107 source changed')
    entry=read(ROOT/'runs/pressure-methods-v4/entries/v4-student50-control.json')
    origin=ROOT/entry['restore'];weights=origin/'weights.npz'
    if sha(weights)!=entry['weights_sha256']:raise ValueError('C107 weights changed')
    ranking_path=BASE/'pressure-profile-02/train-ranking.json'
    ranking=read(ranking_path)
    if ranking['status']!='frozen_from_training_only' or ranking['development_or_reserved_requests_used']:
        raise ValueError('Qualified train-only ranking required')
    shutil.copytree(prior,output,ignore=shutil.ignore_patterns('__pycache__'))
    config=read(output/'config.json')
    config['id']=identifier
    config['training'].update(updates=1200,checkpoints=[0,600,1200],seed=seed,sampling_seed=2026101908+generation-1,
        learning_rate=3e-5,schedule=[dict(update=901,learning_rate=1e-5)])
    write(output/'config.json',config)
    objective=read(output/'fine-tuning.json')
    objective.update(arm=identifier,initial_weights_sha256=sha(weights),
        initial_inventory=read(origin/'history.json')['checkpoints']['checkpoint-2400.npz']['inventory'])
    write(output/'fine-tuning.json',objective)
    multiplier=spec['head_multipliers'][arm]
    optimizer=(output/'head_optimizer.py').read_text()
    if optimizer.count('learning_rate*(10/3)')!=1 or optimizer.count('lr_multiplier=(10/3)')!=1:
        raise ValueError('Captured optimizer seam changed')
    optimizer=optimizer.replace('learning_rate*(10/3)',f'learning_rate*{multiplier!r}').replace('lr_multiplier=(10/3)',f'lr_multiplier={multiplier!r}')
    (output/'head_optimizer.py').write_text(optimizer)
    shutil.copyfile(HERE/'fit_sampling.py',output/'mixed_order.py')
    write(output/'fit-sampling.json',dict(emphasize_tail=arm in ('a-tail','a-combined'),
        ranking_sha256=sha(ranking_path),ranking_records=ranking['records'],training_only=True,
        registration_sha256=sha(DOCS/'registration.json')))
    document=read(output/'project.json')
    for item in document['data']:
        if item['id']=='initial_weights':item.update(path=str(weights.relative_to(ROOT)),sha256=sha(weights))
    write(output/'project.json',document)
    for mode in ('train','restore'):
        recipe=read(output/(mode+'-recipe.json'))
        outputs=[]
        for row in recipe['outputs']:
            if row['path']=='checkpoint-2400.npz':
                outputs.extend([dict(row,path=f'checkpoint-{u:03d}.npz')for u in (600,1200)])
            elif row['path']=='checkpoint-2400.json':outputs.append(dict(row,path='checkpoint-1200.json'))
            else:outputs.append(row)
        recipe['outputs']=outputs
        if mode=='restore':
            old=recipe['inputs'].pop('trained_checkpoint_2400_npz')
            recipe['inputs'].update({f'trained_checkpoint_{u:03d}_npz':old for u in (600,1200)})
            recipe['arguments']=[r for r in recipe['arguments']if r not in ({'literal':'--trained-checkpoint-2400-npz'},{'input':'trained_checkpoint_2400_npz'})]
            for u in (600,1200):
                recipe['arguments'].extend([{'literal':f'--trained-checkpoint-{u:03d}-npz'},{'input':f'trained_checkpoint_{u:03d}_npz'}])
        write(output/(mode+'-recipe.json'),recipe)
    package(output,identifier,['fit-sampling.json'],experiment_path=BASE/(identifier+f'-experiment-{project_generation:02d}.json'))
    write(BASE/(identifier+'-preparation.json'),dict(status='captured',arm=arm,identifier=identifier,
        project=str(output.relative_to(ROOT)),initial_weights_sha256=sha(weights),head_multiplier=multiplier,
        emphasize_tail=arm in ('a-tail','a-combined'),seed=seed,sampling_seed=config['training']['sampling_seed'],
        optimizer_updates=0,source_sha256=sha(__file__),registration_sha256=sha(DOCS/'registration.json')))
    print('Captured C107 continuation',identifier,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',required=True);parser.add_argument('--seed',type=int,default=2026101907)
    parser.add_argument('--generation',type=int,default=1)
    parser.add_argument('--project-generation',type=int,default=1)
    args=parser.parse_args();prepare(args.arm,args.seed,args.generation,args.project_generation)
