"""Matched full-width continuation with one separately qualified data expansion."""
import argparse
import re
import shutil
from v4_support import ROOT,HERE,BASE,read,write,sha
from package_project import package

def prepare(arm):
    output=BASE/(arm+'-project-01')
    if output.exists():raise ValueError('Fresh data-expansion Project required')
    control=BASE/'v4-replay-project-01';witness=read(control/'source-witness.json')
    for name,expected in witness['sources'].items():
        if sha(control/name)!=expected:raise ValueError('Exact replay control source changed')
    view=BASE/('diverse-training-view-01'if arm=='v4-diverse'else'onpolicy-training-view-01')
    manifest=read(view/'manifest.json')
    if not manifest['exact_full_decoded_population_verified']or manifest['retained_v3_parents']!=34:
        raise ValueError('Lossless original-bank retention required')
    shutil.copytree(control,output,ignore=shutil.ignore_patterns('__pycache__'))
    config=read(output/'config.json');config['id']=arm;write(output/'config.json',config)
    objective=read(output/'fine-tuning.json');objective['arm']=arm;write(output/'fine-tuning.json',objective)
    # New, bounded reader version: four on-policy snapshots per parent. Earlier
    # captured readers and all original array values stay immutable.
    for name in ['data.py','methods_view.py']:
        text=(output/name).read_text().replace('rheon-pressure-methods-learning-view-v3','rheon-pressure-methods-learning-view-v4')
        if name=='methods_view.py':text=text.replace("not 1 <= len(parent['frames']) <= 2","not 1 <= len(parent['frames']) <= 4")
        (output/name).write_text(text)
    shutil.copyfile(HERE/'reconstruct.py',output/'native_reconstruct.py')
    extra=[f'view_{i:02d}'for i in range(2,len(manifest['shards']))];slots=['dataset','dataset_extra',*extra]
    trainer=(output/'train.py').read_text()
    trainer,count=re.subn(r'^VIEW_SLOTS = .*$',f'VIEW_SLOTS = {extra!r}',trainer,count=1,flags=re.MULTILINE)
    if count!=1:raise ValueError('Captured view-slot seam changed')
    (output/'train.py').write_text(trainer)
    project=read(output/'project.json');project['data']=[r for r in project['data']if r['id']not in ['dataset','dataset_extra','splits']and not r['id'].startswith('view_')]
    for slot,shard in zip(slots,manifest['shards']):
        path=view/shard['path'];project['data'].append(dict(id=slot,label=arm+' '+slot,path=str(path.relative_to(ROOT)),sha256=sha(path)))
    project['data'].append(dict(id='splits',label=arm+' train-only manifest',path=str((view/'manifest.json').relative_to(ROOT)),sha256=sha(view/'manifest.json')))
    write(output/'project.json',project)
    for mode in ['train','restore']:
        recipe=read(output/(mode+'-recipe.json'));recipe['inputs']={k:v for k,v in recipe['inputs'].items()if not k.startswith('view_')}
        recipe['inputs'].update({k:{'max_bytes':40*1024**2}for k in extra})
        arguments=recipe['arguments'];start=next(i for i,r in enumerate(arguments)if r.get('literal')=='--view-shards')
        end=start+1
        while end<len(arguments)and arguments[end].get('input','').startswith('view_'):end+=1
        recipe['arguments']=arguments[:start]+[{'literal':'--view-shards'},*[{'input':k}for k in extra]]+arguments[end:]
        write(output/(mode+'-recipe.json'),recipe)
    package(output,arm)
    print('Prepared isolated data expansion',arm,'new parents',manifest['added_v4_parents'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=['v4-diverse','v4-onpolicy'],required=True)
    prepare(parser.parse_args().arm)
