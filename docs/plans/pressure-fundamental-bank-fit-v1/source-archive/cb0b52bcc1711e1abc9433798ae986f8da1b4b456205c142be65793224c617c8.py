"""Retain exact execution evidence and apply the predeclared capacity gate."""
import argparse
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def report(config_path,projects,cpu,controller_path,output):
    config=read(config_path)
    arm=config['arms'][0]
    if len(config['arms'])!=1:
        raise ValueError('One completed bank arm per report')
    identifier=arm['model_id']
    controller=read(controller_path);execution=read(cpu/'execution.json')
    if (controller['status']!='completed' or not controller['owned_controller_reaped']
            or execution['status']!='completed' or not execution['all_owned_children_reaped']):
        raise ValueError('Completed and reaped managed and CPU owners required')
    study=projects/(arm['id']+'-study-01')
    receipt=read(study/'receipt.json')
    trained,restored=receipt['training_receipt'],receipt['restore_receipt']
    intent=read(study/'intent.json');assembly_path=Path(intent['assembly'])
    assembly=read(assembly_path);project=ROOT/assembly['project']['path']
    history=read(study/'train/history.json');objective=read(project.parent/'fine-tuning.json')
    if (sha(assembly_path)!=receipt['assembly_sha256'] or sha(project)!=assembly['project']['sha256']
            or receipt['status']!='completed' or not all(receipt['artifact_parity'].values())
            or trained['optimizer_updates']!=config['updates'] or trained['execution']['device']!='cuda:0'
            or trained['execution']['model_dtype']!='float32' or trained['execution']['amp'] or trained['execution']['tf32']
            or not trained['execution']['parameters_gradients_adam_moments_cuda']
            or restored['optimizer_updates']!=0 or restored['execution']['device']!='cpu' or restored['execution']['cuda_initialized']
            or history['checkpoints']['checkpoint-000.npz']['inventory']!=objective['initial_inventory']):
        raise ValueError('Actual CUDA initialization/execution/fresh CPU restore differs')
    custody=[]
    for phase in ['training','restore']:
        terminal=read(study/('terminal-'+receipt[phase+'_run_id']+'.json'))
        if terminal['state']!='completed' or terminal['custody_phase']!='drained':
            raise ValueError('Both remote Runs must complete and drain')
        custody.append(dict(phase=phase,run_id=receipt[phase+'_run_id'],state=terminal['state'],
            custody_phase=terminal['custody_phase'],artifacts=len(terminal['artifacts']),
            terminal_sha256=sha(study/('terminal-'+receipt[phase+'_run_id']+'.json'))))
    if sha(project.parent/'fit_bank.py')!=sha(Path(__file__).with_name('fit_bank.py')):
        raise ValueError('Captured fitting source differs from recorded experiment source')
    stats=trained['fine_tuning_statistics']
    bank=stats['fundamental_bank'];cases=bank['cases'];count=len(cases)
    expected_work=200*arm['steps']*sum(math.prod(c['counts_xyz'])for c in cases)
    if (bank['optimizer_examples']!=config['updates'] or bank['expected_exposures']!=[200]*count
            or bank['optimizer_exposures']!=[200]*count or stats['teacher_forwards']!=0
            or history['optimizer_work']['network_forwards']!=config['updates']*arm['steps']
            or history['optimizer_work']['forward_cells']!=expected_work):
        raise ValueError('Actual balanced fitting-bank optimizer exposure differs')
    snapshots={i:read(cpu/i/'snapshots.json')for i in ['C53',identifier]}
    fitted={i:read(cpu/(i+'-fitted.json'))for i in ['C53',identifier]}
    diagnostics={identifier:read(cpu/(identifier+'-diagnosis.json')),
        'C53':read(ROOT/'docs/plans/pressure-low-mode-tail-v1/C53-diagnosis.json')}
    for group in [snapshots,fitted,diagnostics]:
        for i,row in group.items():
            if row['status']!='passed' or row['cuda_initialized']:
                raise ValueError('Finite CPU-only measurements required')
            expected=receipt['weights_sha256']if i==identifier else objective['initial_weights_sha256']
            if row['weights_sha256']!=expected:
                raise ValueError('CPU measurement belongs to different weights')
    expected_cases=[dict(**g,mode_zyx=m)for g in config['fitting_bank']['grids']for m in config['fitting_bank']['modes_zyx']]
    if (cases!=expected_cases or history['analytical_feedback']is not None
            or any(name in history for name in ['supplemental_feedback','direction_feedback','current_feedback'])
            or [r['update']for r in bank['trace']]!=list(range(10*count,config['updates']+1,10*count))
            or any(r['cases']!=cases or not math.isclose(r['amplitude_rms_pa'],bank['amplitude_rms_pa'],rel_tol=1e-12)for r in fitted.values())):
        raise ValueError('Exact cases, scalar calibration and balanced optimizer exposure required')
    historical=read(ROOT/'runs/pressure-low-mode-tail-v1/cpu-01/C53/snapshots.json')
    key=lambda r:(r['trajectory_id'],r['native_frame'],r['passes'])
    old={key(r):r for r in historical['rows']}
    if len(old)!=512 or any(any(r[f]!=old[key(r)][f]for f in ['relative_residual','same_request_pressure_rms_pa',
        'same_request_mean_free_pressure_relative_l2'])for r in snapshots['C53']['rows']):
        raise ValueError('Historical C53 quality did not replay exactly')
    baseline={(r['case'],r['passes']):r['relative_equation_residual']for r in fitted['C53']['measurements']}
    candidate={(r['case'],r['passes']):r['relative_equation_residual']for r in fitted[identifier]['measurements']}
    expected_keys={(i,n)for i in range(count)for n in [1,2,4,8]}
    if set(baseline)!=expected_keys or set(candidate)!=expected_keys:
        raise ValueError('Every fitted case and pass count required')
    checks=dict(four_pass_all_fitted=all(candidate[i,4]<=1e-6 for i in range(count)),
        eight_pass_all_fitted=all(candidate[i,8]<=1e-6 for i in range(count)),
        one_pass_no_case_worse=all(candidate[i,1]<=baseline[i,1]for i in range(count)),
        two_pass_no_case_worse=all(candidate[i,2]<=baseline[i,2]for i in range(count)))
    capacity=all(checks.values())
    decision=dict(status='capacity_screen_completed',baseline_retained='C53',capacity_gate_passed=capacity,
        capacity_checks=checks,model_promoted=False,original_untouched_holdouts_consumed=0,
        next_action=('Propose a separate mixed-loss retention test; skip the conditional subset fits.'if capacity else
            'Run the predeclared axis and shape isolation fits.'if arm['id']=='combined-fundamentals'else
            'Compare the completed axis and shape fitting screens before choosing further work.'),
        interpretation='A bounded pure-fitting screen is not deployment qualification or proof of architectural impossibility.')
    decision['capacity_evidence']={str(p.relative_to(ROOT)):sha(p)for p in [cpu/'C53-fitted.json',
        cpu/(identifier+'-fitted.json'),study/'receipt.json',config_path]}
    training=dict(id=identifier,arm=arm,source=str((study/'receipt.json').relative_to(ROOT)),sha256=sha(study/'receipt.json'),
        project_id=receipt['project_id'],training_run_id=receipt['training_run_id'],restore_run_id=receipt['restore_run_id'],
        weights_sha256=receipt['weights_sha256'],initialization_sha256=objective['initial_weights_sha256'],
        updates=trained['optimizer_updates'],trainer_elapsed_s=trained['elapsed_s'],
        training_loop_s=history['training_finished_unix']-history['training_started_unix'],
        managed_workflow_elapsed_s=receipt['elapsed_s'],peak_cuda_allocated_bytes=trained['execution']['peak_allocated_bytes'],
        peak_host_rss_bytes=trained['runtime']['peak_rss_bytes'],execution=trained['execution'],
        runtime=trained['runtime'],restore_runtime=restored['runtime'],managed_custody=custody,
        optimizer_work=history['optimizer_work'],fundamental_bank=bank,
        inherited_real_frame_trace=history['trace'],artifact_parity=receipt['artifact_parity'],closure=trained['closure'])
    sources=[config_path,assembly_path,project,project.parent/'config.json',project.parent/'fine-tuning.json',
        project.parent/'source-witness.json',controller_path,cpu/'execution.json',cpu/'material.json',study/'receipt.json',
        study/'readback-profile.json',Path(__file__).resolve()]
    sources.extend(sorted(Path(__file__).parent.glob('*.py')))
    result=dict(schema='rheon-fundamental-bank-fit-results-v1',status='completed',training=[training],
        snapshots=snapshots,fitted=fitted,diagnostics=diagnostics,decision=decision,cpu_execution=execution,
        historical_baseline_quality_exact_rows=512,readback=read(study/'readback-profile.json'),
        sources={str(p.relative_to(ROOT)):sha(p)for p in sources})
    output.mkdir(parents=True,exist_ok=False)
    for name,value in [('capacity-decision.json',decision),('capacity-results.json',result)]:
        if (output/name).exists():
            raise ValueError('Capacity evidence is immutable; use a fresh report destination')
        write(output/name,value)
    write(output/'decision.json',decision);write(output/'results.json',result)
    print(decision,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','projects','cpu','controller','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    report(args.config.resolve(),args.projects.resolve(),args.cpu.resolve(),args.controller.resolve(),args.output.resolve())
