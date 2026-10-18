"""Prepare capacity or curriculum Projects through the typed assembly owner."""
import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT/'experiments/pressure-four-pass-distillation-v1'
sys.path.insert(0,str(PREVIOUS))
from common import read, sha, write
from prepare import prepare, replace_once


def single_trainer(text):
    first = '        analytical_samples=None;analytical_evidence=None\n'
    last = "        profile_report = {'schema':'rheon-cuda-training-profile-v2'"
    if text.count(first) != 1 or text.count(last) != 1:
        raise ValueError('Captured bank-construction seam changed')
    start,end = text.index(first),text.index(last)
    text = text[:start]+'''        import single_case
        single_case.bind(cache)
        analytical_samples=None;analytical_evidence=None
        supplemental_samples=None;supplemental_evidence=None
        direction_samples=None;direction_evidence=None
        current_samples=None;current_evidence=None
        feedback=None;supplemental=None;direction=None
        current=config['current_feedback']
'''+text[end:]
    text = replace_once(text,'loss, gradient = one_update(network, optimizer, parents, indices, normalization, cache, strategy, prepared,',
                        'loss, gradient = single_case.one_update(network, optimizer, parents, indices, normalization, cache, strategy, prepared,')
    text = replace_once(text,"                optimizer_work['enabled']=False\n                verify_cuda_update",
                        "                optimizer_work['enabled']=False\n                single_case.observe(network,normalization,update)\n                verify_cuda_update")
    text = replace_once(text,"**({'current_feedback':current_evidence,'optimizer_work':optimizer_work}if current_evidence is not None else {})",
                        "**{'optimizer_work':optimizer_work}")
    return text


def capacity_gate(path, config):
    if path is None:
        raise ValueError('Restored CPU capacity decision required')
    decision=read(path)
    if decision['status']!='capacity_screen_completed' or not decision['capacity_gate_passed']:
        raise ValueError('The capacity gate did not pass')
    evidence=decision['capacity_evidence']
    for name,identity in evidence.items():
        if sha(ROOT/name)!=identity:
            raise ValueError('Capacity evidence changed: '+name)
    fitted=[read(ROOT/name)for name in evidence if name.endswith('-fitted.json')]
    baseline=next(r for r in fitted if r['model']=='C53')
    candidate=next(r for r in fitted if r['model']!='C53')
    if any(r['fitting_case']!=config['fitting_case'] or r['cuda_initialized']for r in fitted):
        raise ValueError('Same cubic case and restored CPU inference required')
    previous={r['passes']:r['relative_equation_residual']for r in baseline['measurements']}
    values={r['passes']:r['relative_equation_residual']for r in candidate['measurements']}
    if not (values[4]<=1e-6 and values[8]<=1e-6 and all(values[n]<=previous[n]for n in [1,2])):
        raise ValueError('Capacity measurements do not pass the declared gate')


def build(config_path, arm_id, output, capacity_decision=None):
    config = read(config_path)
    arm = next(a for a in config['arms'] if a['id']==arm_id)
    if arm_id!='single-mode-fit':
        capacity_gate(capacity_decision,config)
    selected = {**config,**{k:arm[k] for k in ['updates','learning_rate','schedule','training_seed','sampling_seed','early_feedback_weight']},'arms':[arm]}
    frozen = output.parent/(arm_id+'-experiment.json')
    if output.exists() or (frozen.exists() and read(frozen) != selected):
        raise ValueError('Fresh project and unchanged frozen arm configuration required')
    if not frozen.exists():
        write(frozen,selected)
    if arm_id == 'single-mode-fit':
        prepare(frozen,arm_id,output,source_overrides={'fine_tuning.py':HERE/'fine_tuning.py',
            'original_fine_tuning.py':PREVIOUS/'fine_tuning.py','single_case.py':HERE/'single_case.py'},
            objective_overrides={'fitting_case':config['fitting_case']},trainer_transform=single_trainer)
    else:
        material=read(ROOT/config['baseline_material'])
        control=next(m for m in material['models']if m['id']=='C60')
        project=ROOT/control['project']
        for name,identity in read(project/'source-witness.json')['sources'].items():
            if sha(project/name)!=identity:
                raise ValueError('Historical control source changed: '+name)
        descriptor=output.parent/'mode-curriculum.json'
        if descriptor.exists() and read(descriptor)!=config['curriculum']:
            raise ValueError('Immutable curriculum specification required')
        if not descriptor.exists():
            write(descriptor,config['curriculum'])
        prepare(frozen,arm_id,output,source_overrides={'fine_tuning.py':project/'fine_tuning.py',
            'original_fine_tuning.py':project/'original_fine_tuning.py',
            'analytical.py':HERE/'analytical.py','original_analytical.py':project/'analytical.py',
            'mode-curriculum.json':descriptor},
            config_overrides={'supplemental_feedback':copy.deepcopy(read(project/'config.json')['supplemental_feedback'])},
            objective_overrides={'low_mode_feedback':dict(weight=0.,max_mode=3,signal_floor_fraction=.1)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--arm',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--capacity-decision',type=Path)
    args=parser.parse_args()
    build(args.config.resolve(),args.arm,args.output.resolve(),args.capacity_decision)
