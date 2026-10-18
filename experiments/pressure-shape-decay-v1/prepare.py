"""Freeze a schedule-only bank experiment using the qualified Project owner."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
BANK=ROOT/'experiments/pressure-fundamental-bank-fit-v1'
PREVIOUS=ROOT/'experiments/pressure-four-pass-distillation-v1'
sys.path.insert(0,str(PREVIOUS))
from common import read,sha,write
from prepare import prepare


def owner():
    spec=importlib.util.spec_from_file_location('bank_preparation',BANK/'prepare.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def selected(config,arm):
    settings=owner().selected(config,arm)
    phases=config['lr_phases']
    if sum(p['exposures']for p in phases)!=config['exposures_per_case']or any(type(p['exposures'])is not int or p['exposures']<=0 or not 0<p['learning_rate']<=1e-3 for p in phases):
        raise ValueError('Positive bounded LR phases with exact per-case exposure required')
    count=len(settings['fitting_bank']['grids'])*len(settings['fitting_bank']['modes_zyx'])
    schedule=[];seen=phases[0]['exposures']
    for phase in phases[1:]:
        schedule.append(dict(update=count*seen+1,learning_rate=phase['learning_rate']));seen+=phase['exposures']
    return {**settings,'learning_rate':phases[0]['learning_rate'],'schedule':schedule}


def build(config_path,arm_id,output,decision_path=None):
    config=read(config_path);arm=next(a for a in config['arms']if a['id']==arm_id)
    if arm_id=='joint-later-decay':
        if decision_path is None:raise ValueError('Completed passing C67 decision required')
        decision=read(decision_path)
        if decision['status']!='capacity_screen_completed'or not decision['capacity_gate_passed']or decision['candidate']!='C67':
            raise ValueError('Completed passing C67 decision required')
        for name,identity in decision['capacity_evidence'].items():
            if sha(ROOT/name)!=identity:raise ValueError('C67 gate evidence changed')
    settings=selected(config,arm);frozen=output.parent/(arm_id+'-experiment.json')
    if output.exists()or (frozen.exists()and read(frozen)!=settings):raise ValueError('Fresh immutable arm settings required')
    if not frozen.exists():write(frozen,settings)
    prepare(frozen,arm_id,output,source_overrides={'fine_tuning.py':BANK/'fine_tuning.py',
        'original_fine_tuning.py':PREVIOUS/'fine_tuning.py','fit_bank.py':BANK/'fit_bank.py'},
        objective_overrides={'fitting_bank':settings['fitting_bank']},trainer_transform=owner().transform)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--arm',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--decision',type=Path)
    args=parser.parse_args();build(args.config.resolve(),args.arm,args.output.resolve(),args.decision)
