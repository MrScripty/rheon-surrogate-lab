"""Capture a new execution after the time-limited attempt; never alter its material."""
import shutil
from v6_support import ROOT,HERE,BASE,DOCS,read,write,sha
from package_project import package
def run():
    old=BASE/'v6-unrolled-project-01';out=BASE/'v6-unrolled-project-02'
    if out.exists():raise ValueError('Fresh adjoint Project required')
    for name,expected in read(old/'source-witness.json')['sources'].items():
        if sha(old/name)!=expected:raise ValueError('Original captured training material changed')
    shutil.copytree(old,out,ignore=shutil.ignore_patterns('__pycache__'))
    objective=(out/'fine_tuning.py').read_text()
    if objective.count('import training_multilevel as mg')!=1:raise ValueError('Frozen-cycle call seam changed')
    (out/'fine_tuning.py').write_text(objective.replace('import training_multilevel as mg','import adjoint_multilevel as mg'))
    shutil.copyfile(HERE/'adjoint_multilevel.py',out/'adjoint_multilevel.py')
    fast=(ROOT/'experiments/pressure-methods-v5/fast_multilevel.py').read_text().replace('import multilevel as original','import training_multilevel as original')
    (out/'training_fast_multilevel.py').write_text(fast)
    experiment=BASE/'v6-unrolled-experiment-02.json'
    package(out,'v6-unrolled',['training_multilevel.py','training_fast_multilevel.py','adjoint_multilevel.py'],experiment_path=experiment)
    # Preserve the same initialization metadata and finite scientific settings;
    # the amended execution uses the existing1800-second profile ceiling.
    settings=read(BASE/'v6-unrolled-experiment-01.json');write(experiment,settings)
    assembly=read(out/'assembly.json');assembly['configuration_sha256']=sha(experiment);write(out/'assembly.json',assembly)
    for name in ['config.json','fine-tuning.json']:
        if sha(old/name)!=sha(out/name):raise ValueError('Scientific configuration changed')
    before=read(old/'source-witness.json');after=read(out/'source-witness.json')
    if before['normalization_sha256']!=after['normalization_sha256']or before['packet_experiment_sha256']!=after['packet_experiment_sha256']:raise ValueError('Actual fitting input or normalization changed')
    write(BASE/'adjoint-execution-amendment-01.json',dict(status='captured_before_new_admission',old_project=str(old.relative_to(ROOT)),new_project=str(out.relative_to(ROOT)),
      old_project_sha256=sha(old/'project.json'),new_project_sha256=sha(out/'project.json'),same_config_sha256=sha(out/'config.json'),same_objective_specification_sha256=sha(out/'fine-tuning.json'),
      same_initial_weights_sha256=read(out/'fine-tuning.json')['initial_weights_sha256'],same_packet_sha256=after['packet_experiment_sha256'],same_normalization_sha256=after['normalization_sha256'],
      optimizer_updates=2400,solver_training_iterations=3,wall_seconds=1800,global_runtime_limits_changed=False,
      change='Qualified direct fine operator plus exact symmetric-cycle adjoint, including output-gauge transpose. No gradient is dropped; the first attempt remains separate.',source_sha256=sha(__file__)))
    print('Captured generation02: same scientific settings and1800-second cap; exact frozen-cycle adjoint',flush=True)
if __name__=='__main__':run()
