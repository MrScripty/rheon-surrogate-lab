"""Prepare a bounded fitting bank through the existing typed Project assembly."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PREVIOUS=ROOT/'experiments/pressure-four-pass-distillation-v1'
sys.path.insert(0,str(PREVIOUS))
from common import read,sha,write
from prepare import prepare,replace_once


def transform(text):
    spec=importlib.util.spec_from_file_location('single_fit_preparation',ROOT/'experiments/pressure-single-mode-fit-v1/prepare.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    text=owner.single_trainer(text)
    text=replace_once(text,'import single_case','import fit_bank')
    text=replace_once(text,'single_case.bind(cache)',"fit_bank.bind(cache,config['training']['updates'])")
    text=replace_once(text,'single_case.one_update(','fit_bank.one_update(')
    return replace_once(text,'single_case.observe(','fit_bank.observe(')


def selected(config, arm):
    grids=[config['grids'][i]for i in arm['grid_indices']]
    modes=[config['modes_zyx'][i]for i in arm['mode_indices']]
    cases=len(grids)*len(modes);updates=cases*config['exposures_per_case']
    schedule=[dict(update=cases*exposures+1,learning_rate=rate)for exposures,rate in [(100,1e-5),(150,1e-6),(175,1e-7),(190,1e-8)]]
    return {**config,'arms':[arm],'updates':updates,'schedule':schedule,
        'fitting_bank':dict(grids=grids,modes_zyx=modes,sampling_seed=config['bank_sampling_seed'],
            cache_max_bytes=config['bank_cache_max_bytes'],amplitude_reference='original-train-correction-rms')}


def build(config_path,arm_id,output,decision_path=None):
    config=read(config_path);arm=next(a for a in config['arms']if a['id']==arm_id)
    if arm_id!='combined-fundamentals':
        if decision_path is None:
            raise ValueError('Completed combined-bank decision required before subset experiments')
        decision=read(decision_path)
        if decision['status']!='capacity_screen_completed' or decision['capacity_gate_passed']:
            raise ValueError('Subset experiments follow a failed combined fitting gate only')
        for name,identity in decision['capacity_evidence'].items():
            if sha(ROOT/name)!=identity:
                raise ValueError('Combined fitting evidence changed')
    frozen=output.parent/(arm_id+'-experiment.json');settings=selected(config,arm)
    if output.exists() or (frozen.exists()and read(frozen)!=settings):
        raise ValueError('Fresh Project and immutable experiment settings required')
    if not frozen.exists():write(frozen,settings)
    prepare(frozen,arm_id,output,source_overrides={'fine_tuning.py':HERE/'fine_tuning.py',
        'original_fine_tuning.py':PREVIOUS/'fine_tuning.py','fit_bank.py':HERE/'fit_bank.py'},
        objective_overrides={'fitting_bank':settings['fitting_bank']},trainer_transform=transform)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--arm',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--decision',type=Path)
    args=parser.parse_args();build(args.config.resolve(),args.arm,args.output.resolve(),args.decision)
