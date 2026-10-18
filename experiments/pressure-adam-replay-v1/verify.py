"""Verify retained scientific scope, provenance, byte parity and report custody."""
import argparse
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def verify(directory,output):
    if output.exists():raise ValueError('Fresh verification receipt required')
    value=read(directory/'results.json')
    assert value['status']=='completed' and not value['model_promoted'] and value['baseline_retained']=='C53'
    assert value['final_weights_exact'] and value['final_quality_exact']
    assert value['reference_weights_sha256']==value['replay_weights_sha256']
    assert len(value['final_pressure_comparison'])==32
    assert all(r['reference']==r['replay'] for r in value['final_pressure_comparison'])
    assert all(value['artifact_parity'].values()) and len(value['artifact_parity'])==10
    assert all(r['state']=='completed' and r['custody_phase']=='drained' for r in value['custody'])
    assert not value['cuda_initialized'] and value['optimizer_updates']==0
    assert [r['update'] for r in value['observations']]==[1,10,100,150,200]
    for row,capture in zip(value['observations'],value['metadata']['records'],strict=True):
        assert len(row['cases'])==8 and row['adam_reconstruction_passed']
        assert row['adam_reconstruction_max_abs']<=value['adam_absolute_tolerance']
        assert row['gradient_precision_check_passed']==(row['cpu_mean_vs_captured_cuda_gradient_relative_l2']<=value['gradient_relative_tolerance'])
        for case in row['cases']:
            i=case['case'];before,after=capture['before']['losses'][i],capture['after']['losses'][i]
            assert case['gpu_loss_before']==before and case['gpu_loss_after']==after
            assert case['actual_loss_change']==after-before and math.isfinite(case['predicted_first_order_loss_change'])
    assert value['all_gradient_precision_checks_passed']==all(r['gradient_precision_check_passed'] for r in value['observations'])
    work=value['managed']['optimizer_work']
    assert work['network_forwards']==6400 and work['forward_cells']==7424000 and work['diagnostic_network_forwards']==960
    assert value['managed']['retained_training_bytes']<=64*1024**2
    assert value['managed']['execution']['model_dtype']=='float32' and not value['managed']['execution']['amp']
    for name,identity in value['source_receipts'].items():assert sha(ROOT/name)==identity,name
    for name in ('qualification-execution','analysis-execution'):
        receipt=read(directory/'reports'/(name+'.json'));assert receipt['reaped'] and receipt['returncode']==0
    assert read(directory/'reports/controller.json')['owned_controller_reaped']
    browser=read(directory/'reports/browser.json')
    assert browser['status']=='passed' and browser['browser_reaped'] and browser['console_errors']==[]
    assert browser['html_sha256']==sha(directory/'index.html')
    for name in ('plan.md','commands.md','issues.md','execution-ledger.md','next-steps.md'):assert (directory/name).is_file()
    write(output,dict(status='passed',actual_pressure_replay_exact=True,parameter_replay_exact=True,
        all_adam_steps_reconstructed=True,gradient_precision_limited_updates=[r['update'] for r in value['observations'] if not r['gradient_precision_check_passed']],
        scientific_scope='Eight consumed train cases; pipeline/byte/value consistency only, not independent physics conformance or generalization.',
        all_owned_processes_reaped=True,results_sha256=sha(directory/'results.json'),html_sha256=sha(directory/'index.html'),source_sha256=sha(__file__)))
    print('Verified exact replay, custody, declared work and retained precision limits',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.directory.resolve(),a.output.resolve())
