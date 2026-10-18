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
    if len(config['arms'])!=1 or arm['id']!='single-mode-fit':
        raise ValueError('Single-mode capacity report only')
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
    for phase in ['training','restore']:
        terminal=read(study/('terminal-'+receipt[phase+'_run_id']+'.json'))
        if terminal['state']!='completed' or terminal['custody_phase']!='drained':
            raise ValueError('Both remote Runs must complete and drain')
    stats=trained['fine_tuning_statistics']
    if (stats['single_case']['optimizer_examples']!=config['updates'] or stats['teacher_forwards']!=0
            or history['optimizer_work']['network_forwards']!=config['updates']*arm['steps']
            or history['optimizer_work']['forward_cells']!=config['updates']*arm['steps']*math.prod(config['fitting_case']['counts_xyz'])):
        raise ValueError('Actual single-case optimizer exposure differs')
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
    if (stats['single_case']['case']!=config['fitting_case'] or history['analytical_feedback']is not None
            or any(name in history for name in ['supplemental_feedback','direction_feedback','current_feedback'])
            or [r['update']for r in stats['single_case']['trace']]!=list(range(100,config['updates']+1,100))
            or any(r['fitting_case']!=config['fitting_case']or not math.isclose(r['amplitude_rms_pa'],
                stats['single_case']['amplitude_rms_pa'],rel_tol=1e-12)for r in fitted.values())):
        raise ValueError('Exact case, scalar calibration and one-example exposure required')
    historical=read(ROOT/'runs/pressure-low-mode-tail-v1/cpu-01/C53/snapshots.json')
    key=lambda r:(r['trajectory_id'],r['native_frame'],r['passes'])
    old={key(r):r for r in historical['rows']}
    if len(old)!=512 or any(any(r[f]!=old[key(r)][f]for f in ['relative_residual','same_request_pressure_rms_pa',
        'same_request_mean_free_pressure_relative_l2'])for r in snapshots['C53']['rows']):
        raise ValueError('Historical C53 quality did not replay exactly')
    baseline={r['passes']:r['relative_equation_residual']for r in fitted['C53']['measurements']}
    candidate={r['passes']:r['relative_equation_residual']for r in fitted[identifier]['measurements']}
    checks=dict(four_pass_fitted_residual=candidate[4]<=1e-6,eight_pass_fitted_residual=candidate[8]<=1e-6,
        one_pass_no_worse=candidate[1]<=baseline[1],two_pass_no_worse=candidate[2]<=baseline[2])
    capacity=all(checks.values())
    decision=dict(status='capacity_screen_completed',baseline_retained='C53',capacity_gate_passed=capacity,
        capacity_checks=checks,frequency_curriculum_authorized=capacity,model_promoted=False,
        original_untouched_holdouts_consumed=0,
        next_action='Run the matched frequency-balanced curriculum from C53.'if capacity else
            'Do not start the curriculum or extend this fit. Diagnose optimization and loss interference on the same mode before a structural correction change.',
        interpretation='A passed fitting gate establishes this single-mode capacity only; regression guards determine deployment eligibility separately.')
    decision['capacity_evidence']={str(p.relative_to(ROOT)):sha(p)for p in [cpu/'C53-fitted.json',
        cpu/(identifier+'-fitted.json'),study/'receipt.json',config_path]}
    training=dict(id=identifier,arm=arm,source=str((study/'receipt.json').relative_to(ROOT)),sha256=sha(study/'receipt.json'),
        project_id=receipt['project_id'],training_run_id=receipt['training_run_id'],restore_run_id=receipt['restore_run_id'],
        weights_sha256=receipt['weights_sha256'],initialization_sha256=objective['initial_weights_sha256'],
        updates=trained['optimizer_updates'],trainer_elapsed_s=trained['elapsed_s'],
        training_loop_s=history['training_finished_unix']-history['training_started_unix'],
        managed_workflow_elapsed_s=receipt['elapsed_s'],peak_cuda_allocated_bytes=trained['execution']['peak_allocated_bytes'],
        peak_host_rss_bytes=trained['runtime']['peak_rss_bytes'],execution=trained['execution'],
        optimizer_work=history['optimizer_work'],single_case=stats['single_case'],
        inherited_real_frame_trace=history['trace'],artifact_parity=receipt['artifact_parity'],closure=trained['closure'])
    sources=[config_path,assembly_path,project,project.parent/'config.json',project.parent/'fine-tuning.json',
        project.parent/'source-witness.json',controller_path,cpu/'execution.json',cpu/'material.json',study/'receipt.json',
        study/'readback-profile.json',Path(__file__).resolve()]
    result=dict(schema='rheon-single-mode-fit-results-v1',status='completed',training=[training],
        snapshots=snapshots,fitted=fitted,diagnostics=diagnostics,decision=decision,cpu_execution=execution,
        historical_baseline_quality_exact_rows=512,readback=read(study/'readback-profile.json'),
        sources={str(p.relative_to(ROOT)):sha(p)for p in sources})
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
