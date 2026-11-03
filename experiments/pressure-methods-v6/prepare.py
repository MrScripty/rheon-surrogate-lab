"""Capture four finite managed fits without changing any completed Project."""
import argparse,json,shutil,subprocess
from pathlib import Path
from v6_support import ROOT,HERE,BASE,DOCS,read,write,sha,environment
from package_project import package
def prepare(arm):
    output=BASE/(arm+'-project-01')
    if output.exists():raise ValueError('Fresh Project required')
    prior=ROOT/'runs/pressure-methods-v5/v5-multilevel-project-02'
    for name,expected in read(prior/'source-witness.json')['sources'].items():
        if sha(prior/name)!=expected:raise ValueError('Completed Project changed')
    shutil.copytree(prior,output,ignore=shutil.ignore_patterns('__pycache__'))
    for source,target in [('local_model.py','model.py'),('solver_objective.py','fine_tuning.py')]:shutil.copyfile(HERE/source,output/target)
    campaign=read(DOCS/'campaign.json');config=read(output/'config.json')
    config['id']=arm;config['training'].update(seed=campaign['training_seed'],learning_rate=campaign['learning_rate'],schedule=campaign['schedule'])
    write(output/'config.json',config)
    arch='stencil'if arm=='v6-stencil'else'separable'if arm=='v6-separable'else'tiny'
    objective=dict(schema='rheon-multilevel-smoother-objective-v1',arm=arm,architecture=arch,parameters={'tiny':648,'stencil':27,'separable':9}[arch],
        coarse_training_cycles=0,solver_training_iterations=3 if arm=='v6-unrolled'else 1,
        training_local_corrections=3 if arm=='v6-unrolled'else 1,relative_equation_target=1e-6,initial_weights_sha256='',initial_inventory={})
    write(output/'fine-tuning.json',objective)
    runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    seed=BASE/('shared-tiny-initialization-01'if arch=='tiny'else arm+'-initialization-01')
    if not seed.exists():
        seed.mkdir()
        if arch=='tiny':
            origin=ROOT/'runs/pressure-methods-v5/v5-multilevel-study-01/restore'
            shutil.copyfile(origin/'weights.npz',seed/'weights.npz')
            inventory=read(origin/'history.json')['checkpoints']['checkpoint-2400.npz']['inventory']
            source=dict(origin=str(origin.relative_to(ROOT)),origin_weights_sha256=sha(origin/'weights.npz'))
        else:
            probe="""import sys,json,torch;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import model,data
model.configure('cpu');model.select(json.load(open(sys.argv[1]+'/config.json')));torch.manual_seed(2026101910)
n=model.PressureCorrectionNet();h=data.helpers();h.save_weights(n,Path(sys.argv[2]));print(json.dumps(h.state_inventory(n)))
"""
            result=subprocess.run([runtime['python'],'-B','-c',probe,str(output),str(seed/'weights.npz')],cwd=ROOT,env=environment(runtime),capture_output=True,text=True,timeout=90)
            if result.returncode:raise RuntimeError('Initialization failed: '+result.stderr[-3000:])
            inventory=json.loads(result.stdout);source=dict(seed=2026101910)
        write(seed/'initialization.json',dict(parameters=objective['parameters'],inventory=inventory,weights_sha256=sha(seed/'weights.npz'),**source))
    initial=read(seed/'initialization.json');objective.update(initial_weights_sha256=initial['weights_sha256'],initial_inventory=initial['inventory'])
    write(output/'fine-tuning.json',objective)
    project=read(output/'project.json')
    for item in project['data']:
        if item['id']=='initial_weights':item.update(path=str((seed/'weights.npz').relative_to(ROOT)),sha256=initial['weights_sha256'])
    write(output/'project.json',project)
    package(output,arm,['training_multilevel.py'])
    experiment=BASE/(arm+'-experiment-01.json');spec=read(experiment)
    spec['train_initialization']='same-v5-multilevel-checkpoint-2400'if arch=='tiny'else'seeded-'+arch
    write(experiment,spec);assembly=read(output/'assembly.json');assembly['configuration_sha256']=sha(experiment);write(output/'assembly.json',assembly)
    print('Prepared',arm,objective['parameters'],'parameters,',objective['solver_training_iterations'],'differentiable GCR iterations',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['v6-replay','v6-unrolled','v6-stencil','v6-separable']);prepare(p.parse_args().arm)
