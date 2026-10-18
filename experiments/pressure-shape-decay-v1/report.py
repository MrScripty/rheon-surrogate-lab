"""Apply the existing CPU gate and retain a matched schedule comparison."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def compare(config_path,projects,cpu,controller,qualification,output):
    if output.exists():raise ValueError('Fresh immutable comparison directory required')
    config=read(config_path);arm=config['arms'][0];identifier=arm['model_id']
    control_arm='shape-isolation'if identifier=='C67'else'combined-fundamentals'
    control_id='C66'if identifier=='C67'else'C64'
    control_path=ROOT/config['control_study']/control_arm/'results.json';control=read(control_path)
    spec=importlib.util.spec_from_file_location('capacity_report',ROOT/'experiments/pressure-fundamental-bank-fit-v1/report.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    raw=projects/(arm['id']+'-full-report-01')
    owner.report(config_path,projects,cpu,controller,raw)
    candidate=read(raw/'results.json');training=candidate['training'][0];prior=control['training'][0]
    study=projects/(arm['id']+'-study-01');history=read(study/'train/history.json')
    proof=read(qualification);execution=read(controller)
    if execution['qualification_sha256']!=sha(qualification)or not proof['matched_schedule']['only_learning_rate_schedule_changed']:
        raise ValueError('Matched qualification changed after admission')
    bank=training['fundamental_bank'];previous=prior['fundamental_bank']
    if (training['optimizer_work']!=prior['optimizer_work']or training['updates']!=prior['updates']
            or bank['sampling_order_sha256']!=previous['sampling_order_sha256']
            or bank['cases']!=previous['cases']or bank['amplitude_rms_pa']!=previous['amplitude_rms_pa']):
        raise ValueError('Actual optimizer work, sampling or fitting population differs')
    for row in history['trace']:
        rate=config['learning_rate']
        for change in config['schedule']:
            if change['update']<=row['update']:rate=change['learning_rate']
        if row['learning_rate']!=rate:raise ValueError('Observed CUDA learning rate differs')
    count=len(bank['cases']);prefix=100*count
    earlier=[r for r in previous['trace']if r['update']<=prefix]
    current=[r for r in bank['trace']if r['update']<=prefix]
    if earlier!=current:
        raise ValueError('Identical pre-schedule-change CUDA trajectory did not replay exactly')
    decision={**candidate['decision'],'candidate':identifier,'control':control_id,
        'next_action':('Qualify the conditional 24-case C68 test.'if identifier=='C67'and candidate['decision']['capacity_gate_passed']else
            'Stop this sequence; retain C53 and diagnose optimization before another run.'if identifier=='C67'else
            'Retain C53 and use this result to plan a separate retention experiment.')}
    decision['capacity_evidence'].update({str(qualification.relative_to(ROOT)):sha(qualification)})
    matched=dict(control=control_id,control_result_source=str(control_path.relative_to(ROOT)),control_result_sha256=sha(control_path),
        actual_optimizer_work_exact=True,actual_case_order_exact=True,unchanged_calibrated_amplitude=True,
        exact_shared_cuda_prefix_updates=prefix,exact_shared_cuda_prefix_observations=len(current),
        observed_lr_trace_matches=True,control_fitted=control['fitted'][control_id],
        control_real_summary=control['snapshots'][control_id]['summary'],
        control_manufactured_summary=control['diagnostics'][control_id]['summary'])
    # The full row receipt is kept once. The offline aggregate projects summaries.
    result={**candidate,'schema':'rheon-shape-decay-arm-results-v1','decision':decision,'matched_comparison':matched,
        'evidence_sources':{str(p.relative_to(ROOT)):sha(p)for p in [qualification,raw/'results.json',Path(__file__).resolve()]}}
    output.mkdir(parents=True)
    compact=json.dumps(result,separators=(',',':'),allow_nan=False)+'\n'
    (output/'result.json').write_text(compact)
    write(output/'decision.json',decision)
    print('Recorded',identifier,decision['capacity_gate_passed'],'exact common prefix',prefix,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','projects','cpu','controller','qualification','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();compare(args.config.resolve(),args.projects.resolve(),args.cpu.resolve(),args.controller.resolve(),args.qualification.resolve(),args.output.resolve())
