"""Close the matched curriculum screen and retain both study stages."""
import argparse
import math
import statistics
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def report(config_path,projects,cpu,controller_path,qualification_path,output):
    capacity=read(output/'capacity-results.json')
    if not capacity['decision']['capacity_gate_passed']:
        raise ValueError('Capacity success required first')
    config=read(config_path);arm=config['arms'][0];identifier=arm['model_id']
    if len(config['arms'])!=1 or arm['id']!='frequency-curriculum':
        raise ValueError('One frequency-curriculum candidate required')
    controller=read(controller_path);execution=read(cpu/'execution.json');qualification=read(qualification_path)
    if (controller['status']!='completed' or not controller['owned_controller_reaped']
            or execution['status']!='completed' or not execution['all_owned_children_reaped']):
        raise ValueError('Completed and reaped owners required')
    study=projects/(arm['id']+'-study-01');receipt=read(study/'receipt.json')
    trained,restored=receipt['training_receipt'],receipt['restore_receipt']
    assembly_path=Path(read(study/'intent.json')['assembly']);assembly=read(assembly_path)
    project=ROOT/assembly['project']['path'];objective=read(project.parent/'fine-tuning.json')
    history=read(study/'train/history.json');bank=history['supplemental_feedback']
    material=read(cpu/'material.json');control=next(m for m in material['models']if m['id']=='C60')
    control_receipt_path=ROOT/control['training_receipt_source']
    if sha(control_receipt_path)!=control['training_receipt_sha256']:
        raise ValueError('Historical control receipt changed')
    control_receipt=read(control_receipt_path);control_history=read(control_receipt_path.parent/'train/history.json')
    if (sha(assembly_path)!=receipt['assembly_sha256'] or sha(project)!=assembly['project']['sha256']
            or qualification['project_sha256']!=sha(project) or qualification['source_witness_sha256']!=sha(project.parent/'source-witness.json')
            or qualification['status']!='passed' or qualification['optimizer_updates']!=0
            or receipt['status']!='completed' or not all(receipt['artifact_parity'].values())
            or trained['optimizer_updates']!=config['updates'] or trained['execution']['device']!='cuda:0'
            or trained['execution']['model_dtype']!='float32' or trained['execution']['amp'] or trained['execution']['tf32']
            or not trained['execution']['parameters_gradients_adam_moments_cuda']
            or restored['optimizer_updates']!=0 or restored['execution']['device']!='cpu' or restored['execution']['cuda_initialized']
            or history['checkpoints']['checkpoint-000.npz']['inventory']!=objective['initial_inventory']):
        raise ValueError('Qualified CUDA initialization/execution/fresh CPU restore differs')
    for phase in ['training','restore']:
        terminal=read(study/('terminal-'+receipt[phase+'_run_id']+'.json'))
        if terminal['state']!='completed' or terminal['custody_phase']!='drained':
            raise ValueError('Both managed Runs must complete and drain')
    for name in ['analytical_feedback','direction_feedback','current_feedback','execution_strategy','optimizer_work']:
        if history[name]!=control_history[name]:
            raise ValueError('Matched control exposure/work changed: '+name)
    for name in ['grid_exposures','sampling_order_sha256','resident_tensor_bytes','weight',
                 'combined_resident_tensor_bytes','primary_bank_description_sha256']:
        if bank[name]!=control_history['supplemental_feedback'][name]:
            raise ValueError('Matched supplemental setting changed: '+name)
    if (bank['actual_mode_category_exposures']!=qualification['hard_bank']['actual_mode_category_exposures']
            or not math.isclose(bank['amplitude_rms_pa'],qualification['hard_bank']['amplitude_rms_pa'],rel_tol=1e-12)):
        raise ValueError('Actual qualified spectral exposure/scalar calibration changed')
    closure=lambda r:{k:v for k,v in r['closure'].items()if k not in ['experiment_sha256','source_witness_sha256']}
    if closure(trained)!=closure(control_receipt['training_receipt']):
        raise ValueError('Training input/order/normalization/runtime closure changed')
    current={i:read(cpu/i/'snapshots.json')for i in ['C53',identifier]}
    key=lambda r:(r['trajectory_id'],r['native_frame'],r['passes'])
    old={key(r):r for r in capacity['snapshots']['C53']['rows']}
    if len(old)!=512 or any(r['relative_residual']!=old[key(r)]['relative_residual']for r in current['C53']['rows']):
        raise ValueError('C53 quality did not replay exactly')
    diagnostic=read(cpu/(identifier+'-diagnosis.json'))
    fitted=read(cpu/(identifier+'-fitted.json'))
    if any(r['cuda_initialized']or r['weights_sha256']!=receipt['weights_sha256']for r in [current[identifier],diagnostic,fitted]):
        raise ValueError('CPU measurements belong to different weights/device')
    baseline={r['passes']:r for r in current['C53']['summary']}
    summaries={r['passes']:r for r in current[identifier]['summary']}
    manufactured=sum(r['relative_equation_residual']<=1e-6 for r in diagnostic['rows']if r['passes']==8)
    checks=dict(four_pass_all_128=summaries[4]['equation_passes']==128,eight_pass_all_128=summaries[8]['equation_passes']==128,
        earlier_pass_worst_no_worse=all(summaries[n]['worst_equation_residual']<=baseline[n]['worst_equation_residual']for n in [1,2]),
        four_pass_worst_no_worse=summaries[4]['worst_equation_residual']<=baseline[4]['worst_equation_residual'],
        manufactured_eight_pass_at_least_230=manufactured>=230)
    passed=all(checks.values())
    decision={**capacity['decision'],'status':'study_completed','curriculum_completed':True,'curriculum_checks':checks,
        'eligible_for_independent_scenes':[identifier]if passed else [],'model_promoted':False,
        'next_action':'Qualify eight new independent scenes before any promotion.'if passed else
            'Keep C53. Do not lengthen this curriculum fit; diagnose the per-mode contraction and early-pass tradeoffs before the next matched ablation.'}
    training=dict(id=identifier,arm=arm,source=str((study/'receipt.json').relative_to(ROOT)),sha256=sha(study/'receipt.json'),
        project_id=receipt['project_id'],training_run_id=receipt['training_run_id'],restore_run_id=receipt['restore_run_id'],
        weights_sha256=receipt['weights_sha256'],initialization_sha256=objective['initial_weights_sha256'],updates=trained['optimizer_updates'],
        trainer_elapsed_s=trained['elapsed_s'],training_loop_s=history['training_finished_unix']-history['training_started_unix'],
        managed_workflow_elapsed_s=receipt['elapsed_s'],peak_cuda_allocated_bytes=trained['execution']['peak_allocated_bytes'],
        peak_host_rss_bytes=trained['runtime']['peak_rss_bytes'],execution=trained['execution'],optimizer_work=history['optimizer_work'],
        artifact_parity=receipt['artifact_parity'],closure=trained['closure'],bank=bank,trace=history['trace'],
        matched_control='C60',matched_qualification=qualification['matched_control'])
    previous=read(ROOT/'docs/plans/pressure-low-mode-tail-v1/results.json')
    control_modes=previous['diagnosis']['C60']
    mode_key=lambda r:(tuple(r['counts_xyz']),r['cell_size_m'],tuple(r['mode_zyx']),r['amplitude_multiplier'],r['passes'])
    control_rows={mode_key(r):r for r in control_modes['rows']}
    if set(control_rows)!={mode_key(r)for r in diagnostic['rows']}:
        raise ValueError('Matched manufactured regression membership changed')
    mode_comparisons=[]
    for passes in [4,8]:
        for mode in sorted({tuple(r['mode_zyx'])for r in diagnostic['rows']}):
            group=[r for r in diagnostic['rows']if r['passes']==passes and tuple(r['mode_zyx'])==mode]
            ratios=[r['relative_equation_residual']/control_rows[mode_key(r)]['relative_equation_residual']for r in group]
            mode_comparisons.append(dict(passes=passes,mode_zyx=list(mode),cases=len(group),
                median_paired_residual_ratio_to_C60=statistics.median(ratios),
                improved=sum(r<1 for r in ratios),worsened=sum(r>1 for r in ratios),
                worst_relative_residual=max(r['relative_equation_residual']for r in group),
                qualified=sum(r['relative_equation_residual']<=1e-6 for r in group)))
    result={**capacity,'schema':'rheon-single-mode-fit-results-v1','status':'completed',
        'training':capacity['training']+[training],
        'snapshots':{**capacity['snapshots'],'C53':current['C53'],'C60':previous['snapshots']['C60'],identifier:current[identifier]},
        'fitted':{**capacity['fitted'],identifier:fitted},
        'diagnostics':{**capacity['diagnostics'],'C60':previous['diagnosis']['C60'],identifier:diagnostic},
        'decision':decision,'mode_comparisons_to_C60':mode_comparisons,'curriculum_cpu_execution':execution,
        'curriculum_baseline_snapshot':current['C53'],'curriculum_readback':read(study/'readback-profile.json'),
        'curriculum_sources':{str(p.relative_to(ROOT)):sha(p)for p in [config_path,assembly_path,project,project.parent/'source-witness.json',
            study/'receipt.json',controller_path,qualification_path,cpu/'material.json',cpu/'execution.json',
            cpu/'C53/snapshots.json',cpu/identifier/'snapshots.json',cpu/(identifier+'-diagnosis.json'),
            cpu/(identifier+'-fitted.json'),Path(__file__).resolve()]}}
    write(output/'curriculum-decision.json',decision)
    write(output/'decision.json',decision);write(output/'results.json',result)
    print(decision,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','projects','cpu','controller','qualification','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    report(args.config.resolve(),args.projects.resolve(),args.cpu.resolve(),args.controller.resolve(),args.qualification.resolve(),args.output.resolve())
