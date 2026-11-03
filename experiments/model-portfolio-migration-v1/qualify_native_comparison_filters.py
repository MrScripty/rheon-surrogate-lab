"""Four retained native reads with independent original-capture verification.

Sequential one-shot requests, no automatic retry. Model inference and reference
generation are outside this workflow. Leave the user controller/project open.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / 'runs/dependency-migration-v1/app-env/bin/python'
EXPERIMENTS = ROOT / 'experiments/model-portfolio-migration-v1'
AUTHORING = ROOT / 'runs/model-portfolio-migration-v1/native-portfolio-comparison-authoring-01'


def save(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def run(controller, output, report):
    if output.exists() or report.exists():
        raise ValueError('Fresh qualification outputs required; reconcile earlier requests explicitly')
    output.mkdir(parents=True)
    state = dict(schema='rheon.native-comparison-filter-qualification.v1',status='running',
        created_unix=time.time(),reads=[],new_neural_forwards=0,new_reference_generation=False,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),user_UI_left_open=True)
    save(report,state)
    try:
        for name in ('density8','domain16','missing-timings','unknown-device'):
            item=dict(name=name,state='dispatching_once')
            state['reads'].append(item);save(report,state)
            with (output/(name+'.owner.log')).open('x') as log:
                subprocess.run([str(PYTHON),'-B',str(EXPERIMENTS/'native_project_owner.py'),
                    '--controller',str(controller),'--operation','project_files.measurement_compare',
                    '--body',str(AUTHORING/(name+'.body.json')),'--output',str(output/name),
                    '--timeout-seconds','300'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=370)
            original_report=output/(name+'.original-verification.json')
            with (output/(name+'.verification.log')).open('x') as log:
                subprocess.run([str(PYTHON),'-B',str(EXPERIMENTS/'verify_native_comparison.py'),
                    '--request',str(output/(name+'.request.json')),'--response',str(output/(name+'.response.json')),
                    '--report',str(original_report)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
            result=json.loads(original_report.read_bytes())
            item.update(state='passed_original_capture_verification',report=str(original_report.relative_to(ROOT)),
                metric=result['metric'],filters=result['filters'],observations=result['observations'],
                measured=result['measured'],unavailable=result['explicit_unavailable'])
            save(report,state)
            print(json.dumps(dict(name=name,state=item['state'],observations=item['observations'])),flush=True)
        state.update(status='passed_four_native_filters_and_original_capture_oracles',finished_unix=time.time())
    except BaseException as error:
        state.update(status='stopped_for_explicit_reconciliation',error_type=type(error).__name__,
            error=str(error),next_action='Inspect last retained request and live controller; no automatic replay')
        raise
    finally:
        save(report,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    run(args.controller.resolve(),args.output.resolve(),args.report.resolve())
