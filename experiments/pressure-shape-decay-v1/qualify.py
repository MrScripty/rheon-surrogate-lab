"""Publish qualification only after bank and schedule-only control checks pass."""
import argparse
import copy
import importlib.util
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def check(config_path,project,output):
    started=time.perf_counter();config=read(config_path)
    if output.exists():raise ValueError('Fresh complete qualification required')
    arm=config['arms'][0];control_arm='shape-isolation'if arm['id']=='shape-later-decay'else'combined-fundamentals'
    control=ROOT/config['control_projects']/(control_arm+'-project-01')
    left=read(project/'config.json');right=read(control/'config.json')
    left=copy.deepcopy(left);right=copy.deepcopy(right)
    actual_schedule=left['training'].pop('schedule');right['training'].pop('schedule')
    left.pop('id');right.pop('id')
    if left!=right or actual_schedule!=config['schedule']:
        raise ValueError('Only captured schedule and experiment identity may differ from control')
    objective=read(project/'fine-tuning.json');previous=read(control/'fine-tuning.json')
    objective.pop('arm');previous.pop('arm')
    if objective!=previous:raise ValueError('Control objective differs')
    witness=read(project/'source-witness.json');old=read(control/'source-witness.json')
    comparable=lambda w:{k:v for k,v in w['sources'].items()if k not in ['fine-tuning.json','config.json']}
    if comparable(witness)!=comparable(old)or witness['normalization_sha256']!=old['normalization_sha256']:
        raise ValueError('Control model, trainer, bank, operator sources or normalization differ')
    prior=read(ROOT/config['control_projects']/(control_arm+'-study-01')/'receipt.json')
    if read(project/'assembly.json')['runtime_manifest_sha256']!=read(control/'assembly.json')['runtime_manifest_sha256']:
        raise ValueError('Managed runtime differs')
    spec=importlib.util.spec_from_file_location('bank_qualification',ROOT/'experiments/pressure-fundamental-bank-fit-v1/qualify.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    bank_output=output.with_suffix('.bank.json');owner.check(project,bank_output)
    bank=read(bank_output)
    if bank['fundamental_bank']['sampling_order_sha256']!=prior['training_receipt']['fine_tuning_statistics']['fundamental_bank']['sampling_order_sha256']:
        raise ValueError('Balanced training sample order differs from retained control')
    bank.update(matched_schedule=dict(control='C66'if control_arm=='shape-isolation'else'C64',
        unchanged_model_trainer_operator_bank_sources=True,unchanged_objective=True,unchanged_initialization=True,
        unchanged_normalization=True,unchanged_runtime=True,unchanged_case_order=True,
        only_learning_rate_schedule_changed=True,schedule=actual_schedule,control_project=str(control.relative_to(ROOT))),
        schedule_qualification_source_sha256=sha(__file__),schedule_qualification_elapsed_s=time.perf_counter()-started)
    write(output,bank);print('Qualified matched later schedule',arm['model_id'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','project','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();check(args.config.resolve(),args.project.resolve(),args.output.resolve())
