"""Audit all24full development histories and explicit current-backend inheritance."""
import argparse,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(args):
    if args.output.exists():raise ValueError('Fresh coverage audit evidence required')
    reg=read(args.protocol);controller=read(args.controller)
    if controller['status']!='completed_registered_remaining_development_240histories' or not controller['all_accuracy_pass'] or len(controller['completed'])!=17:
        raise ValueError('Exactly17remaining terminal/pass cases required')
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Frozen batch source/evidence changed: '+p)
    bank=read(ROOT/reg['bank']) if 'bank' in reg else read(ROOT/'runs/full-ai-data-diversity-v1/verified-expanded-bank-02/development-bank.json')
    population={r['identifier']:r for r in bank['trajectories']}
    paths={r['identifier']:r['path'] for r in reg['inherited_completed_development']}
    for row in controller['completed']:
        if row['identifier'] in paths or digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Duplicate or changed completed case')
        paths[row['identifier']]=row['path']
    if len(paths)!=24 or set(paths)!=set(population):raise ValueError('Whole24development population required')
    proof=read(ROOT/reg['selected_execution_equivalence'])
    if proof['status']!='passed_all_four_GPU_own_history_byte_equivalence' or not proof['every_output_exact_to_native_qualified_baseline']:
        raise ValueError('Current backend inheritance proof required for original32cases')
    rows=[]
    for identifier,path in paths.items():
        q=read(ROOT/path);source=population[identifier];custody=q['native_process']
        if (q['status']!='completed_streamed_expanded_gpu_240step_scene' or not q['all_accuracy_pass'] or len(q['rows'])!=960
            or q['source_split']!='development' or q['counts_xyz']!=source['counts_xyz'] or q['cells_per_metre']!=source['cells_per_metre']
            or q['internal_substeps']!=source['reference_substeps'] or q['output_dt_s']!=1/60
            or not q['candidate_own_history_without_teacher_resets'] or not q['every_intermediate_physical_output_scored']
            or not custody['reaped'] or custody['returncode']!=0 or not custody['graceful_protocol_complete']):
            raise ValueError('Incomplete accuracy/clock/metadata/native custody: '+identifier)
        expected={(k,step) for k in [1,2,3,4] for step in range(1,241)}
        if {(r['K'],r['step']) for r in q['rows']}!=expected or not all(r['metrics']['all_accuracy_criteria_pass'] and all(v<=1e-5 for v in r['metrics']['relative'].values()) for r in q['rows']):
            raise ValueError('All960unique state criteria required: '+identifier)
        if len(q['reference_rows'])!=240 or {r['step'] for r in q['reference_rows']}!=set(range(1,241)):
            raise ValueError('Every native physical frame required')
        if not all(r['all_typed_bytes_match'] for r in q['old_checkpoint_proofs']):raise ValueError('Original archive lineage changed')
        if identifier in ['diversity-development-g024-cpm50','diversity-development-g024-cpm100']:
            inherited={ (r['K'],r['step']):r['output_sha256'] for r in proof['rows'] if r['identifier']==identifier }
            if set(inherited)!=expected or any(inherited[(r['K'],r['step'])]!=r['output_sha256'] for r in q['rows']):
                raise ValueError('Current backend inheritance not proved output-by-output')
            mode='native04plus_current_backend_exact960field_hash_proof'
        else:
            if 'experiments/full-ai-data-diversity-v1/debug/cube64_layout_transform_candidate_v1.py' not in q['source_sha256']:
                raise ValueError('Direct case must source-bind current repaired candidate')
            mode='direct_current_backend_uninterrupted_native_reference'
        reference={r['step']:r['actual_field_sha256'] for r in q['reference_rows']}
        rows.append(dict(identifier=identifier,counts_xyz=q['counts_xyz'],cells_per_metre=q['cells_per_metre'],internal_substeps=q['internal_substeps'],
            source_group=source['group'],forcing_profile=source['profile'],motion=source['motion'],path=path,sha256=digest(ROOT/path),
            current_backend_evidence=mode,complete_outputs=960,all_accuracy_pass=True,worst_error=q['worst_error'],worst_error_byK=q['worst_error_byK'],
            every_native_field_byte_equal=all(all(h==reference[r['step']][n] for n,h in r['output_sha256'].items()) for r in q['rows']),native_process=custody))
    result=dict(status='passed_all24development_full240_accuracy_and_custody',created_unix=time.time(),rows=rows,scene_count=24,group_count=len({r['source_group'] for r in rows}),
        complete_outputs=23040,every_K1to4_and_step1to240_required=True,inference_device='cuda',display_label='GPU (CUDA)',gpu_model='NVIDIA GeForce RTX 5090 Laptop GPU',
        pressure_dtype='float64',carrier_dtype='float32',output_dt_s=1/60,forcing_frequency_and_render_cadence_are_separate=True,maximum_error=1e-5,
        worst_error={name:max(r['worst_error'][name] for r in rows) for name in ['equation','pressure','velocity','tracer']},
        all_accuracy_pass=True,all_native_children_gracefully_reaped=True,candidate_own_history_without_teacher_resets=True,all24_development_verified=True,
        independent_confirmation_verified=False,expanded_reserved52_references_opened=0,latency_claim=False,full_generator_target_verified=False,
        matched_aspect_scaling_verified=False,phase600steps_started=False,cpu_model_forward=False,optimizer_updates=0,
        selected_candidate='experiments/full-ai-data-diversity-v1/debug/cube64_layout_transform_candidate_v1.py:Cube64LayoutTransformCandidate',
        selected_execution_equivalence=reg['selected_execution_equivalence'],selected_execution_equivalence_sha256=digest(ROOT/reg['selected_execution_equivalence']),
        batch_protocol_sha256=digest(args.protocol),batch_controller_sha256=digest(args.controller),auditor_source_sha256=digest(Path(__file__)))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('All24development scenes,23040complete states passed; worst',result['worst_error'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--controller',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
