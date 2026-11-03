"""Admit original readout populations without executing or decoding a model."""
from collections import Counter
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, RUN, read, save, binding, require
from component_material import observed_file


def run():
    output = RUN/'macro-readout-component-contract-01.json'
    require(not output.exists(), 'Preserve original readout comparison admission')
    material_path = RUN/'remaining-component-material-01.json'
    material = read(material_path)
    selected = [row for row in material['models'] if row['task'] == 'full_state_readout_component']
    require(len(selected) == 2 and {row['serialized_tensor_elements'] for row in selected} == {9,12},
            'Only the original 3/4-stage F64 coefficient fits are admitted')
    manifest_path = ROOT/selected[0]['inputs']['manifest']['path']
    manifest = read(manifest_path)
    progress_path = manifest_path.parent/'progress.json'
    progress = read(progress_path)
    protocol_path = ROOT/'docs/plans/full-ai-data-diversity-v1/macrostep-loss-view-protocol-01.json'
    protocol = read(protocol_path)
    require(binding(protocol_path)['sha256'] == manifest['protocol_sha256']
            and progress['manifest_sha256'] == binding(manifest_path)['sha256']
            and manifest['status'] == 'verified_all148_train_macrostep_loss_view'
            and manifest['training_scenes'] == 148 and manifest['training_N8_scenes'] == 29
            and manifest['training_N1_scenes'] == 119 and manifest['starts'] == [32,96]
            and not manifest['development_or_reserved_used'] and len(progress['rows']) == 592,
            'Exact original train-only loss view and population required')
    bank_path = ROOT/protocol['bank']
    bank = read(bank_path)
    require(bank['split'] == 'train' and len(bank['trajectories']) == 148, 'Original 148 training parents required')
    parents = {row['identifier']: row for row in bank['trajectories']}
    cases, fixed, references = [], [], {}
    expected = {}
    for row in progress['rows']:
        parent = parents[row['identifier']]
        require(parent['split'] == 'train' and parent['reference_substeps'] == row['teacher_substeps']
                and row['trainable_readout_active'] == (row['teacher_substeps'] == 8), 'Do not confuse fixed losses with learned readout use')
        key = (row['identifier'],row['start'])
        metadata = {name: row[name] for name in ('identifier','scene_index','counts_xyz','cells_per_metre','start','teacher_substeps','trainable_readout_active')}
        require(row['student_stages'] in (3,4) and (key not in expected or expected[key] == metadata),
                'S3/S4 must retain exactly matched case metadata')
        expected[key] = metadata
        if row['student_stages'] != 3:
            continue
        receipt_path = ROOT/parent['receipt']
        observed_file(ROOT,parent['receipt'],dict(sha256=parent['receipt_sha256']))
        scene = read(receipt_path)['scene']
        require(scene['counts'] == row['counts_xyz'] and scene['cells_per_metre'] == row['cells_per_metre'],
                'Grid and physical spacing must come from the original native scene')
        states = {value['step']: value for value in parent['states']}
        for step in (row['start'],row['start']+1):
            value = states[step]
            require(manifest['source_evidence'][value['path']] == value['sha256'], 'Original current/target archive binding required')
            references[value['path']] = dict(path=value['path'],sha256=value['sha256'])
        (cases if row['trainable_readout_active'] else fixed).append(dict(metadata,
            current_state=states[row['start']],target_state=states[row['start']+1],receipt=parent['receipt']))
    require(len(expected) == 296 and len(cases) == 58 and len(fixed) == 238, 'All original active/fixed windows required')
    for entry in selected:
        for spec in [entry['original_receipt'],entry['original_configuration'],*entry['sources'],
                     entry['inputs']['manifest'],entry['inputs']['normalization'],entry['inputs']['part_00'],entry['outputs']['weights.npz']]:
            observed_file(ROOT,spec['path'],spec)
    result = dict(status='original_macro_readout_populations_and_physical_boundary_admitted_not_benchmarked',
        created_unix=time.time(),producer=binding(Path(__file__)),material=binding(material_path),
        original_manifest=binding(manifest_path),original_progress=binding(progress_path),
        original_protocol=binding(protocol_path),original_bank=binding(bank_path),
        fit_ids=[entry['fit_item_id'] for entry in selected],physical_output_dt_s=1/60,
        stages=[3,4],raw_published_coefficients_preserved=True,
        active_scenes=29,active_full_field_readout_windows_per_fit=58,cases=cases,
        fixed_branch_scenes=119,fixed_branch_windows=238,fixed_branch_cases=fixed,
        by_active_cells_per_metre=dict(sorted(Counter(str(row['cells_per_metre']) for row in cases).items())),
        by_active_counts_xyz=dict(sorted(Counter('x'.join(map(str,row['counts_xyz'])) for row in cases).items())),
        original_reference_archive_bindings=list(references.values()),
        reference_archive_contents_rehashed=False,
        input_contract='Original causal frozen backbone supplies S+1complete stage fields for 29N8parents; targets never enter feature generation. GPU qualification must cover every constructor/helper before use.',
        output_contract='MacroReadout.read returns pressure F64 and velocity/tracer F32 at one next1/60s physical output. Internal3/4stages are not3/4future physical outputs.',
        timing_boundary='Owned CPU causal stage features,packing/H2D,unchanged CUDA readout,owned CPU full output/D2H/sync;stage generation and compilation separately measured. Bare readout latency is not full-step latency.',
        inactive_contract='119N1parents deploy the unchanged one-stage branch and bypass the learned head. Record238windows as not_applied; do not fabricate S+1features or count them as learned field predictions.',
        training_objective_diagnostic='All148parents/two windows/four original QR residual metrics may be evaluated on CUDA separately. Fixed N1losses do not depend on learned weights. CPU scoring must not implement a learned NumPy forward.',
        deployment_variants='Published raw F64 coefficients versus MacrostepStudent2^-40deployment rounding are distinct variants. Do not silently quantize original fit weights or transfer raw-head accuracy to a full generator.',
        required_before_execution=['Full transitive frozen backbone/source/build/parameter closure',
            'Original current/target archive content hashes and field ABI',
            'Actual CUDA constructor/direct-method guards; nn.Module forward hooks alone do not observe .read/.residuals',
            'Two fresh process phases,exact repeated fields,independent physical field/equation scores and density/grid groups',
            'All58active windows perfit;238fixed windows explicitly not_applied;separate original QR objective diagnostic'],
        new_model_forwards=0,CPU_neural_forwards=0,new_parameter_fits=0,new_reference_simulations=0,
        dataset_weight_or_project_copies=0,accuracy_qualified=False,latency_qualified=False,
        default_coverage_increased=False,full_portfolio_complete=False)
    save(output,result)
    print('Two readouts admitted:58active field windows each;238fixed bypass windows;no inference or coverage increase.',flush=True)


if __name__ == '__main__':
    run()
