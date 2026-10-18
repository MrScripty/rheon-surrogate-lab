"""Prepare capacity or curriculum Projects through the typed assembly owner."""
import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT/'experiments/pressure-four-pass-distillation-v1'
sys.path.insert(0,str(PREVIOUS))
from common import read, write
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


def build(config_path, arm_id, output):
    config = read(config_path)
    arm = next(a for a in config['arms'] if a['id']==arm_id)
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
        raise ValueError('Curriculum preparation requires a passed capacity gate')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--arm',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.config.resolve(),args.arm,args.output.resolve())
