"""Bind four original correction heads to their retained full-grid corpus."""
from collections import Counter
from pathlib import Path
import time
import zipfile

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require
from component_material import current_publication, observed_file


def run():
    output = RUN/'internal-residual-component-contract-01.json'
    require(not output.exists(), 'Preserve original correction admission')
    publication,_ = current_publication(ROOT)
    material_path = RUN/'remaining-component-material-01.json'
    material = read(material_path)
    selected = [row for row in material['models'] if row['task'] == 'full_state_correction_component']
    require(len(selected) == 4 and all(row['serialized_tensor_elements'] == 10922 for row in selected),
            'Exactly four original correction checkpoints required')
    manifest_path = ROOT/'runs/full-ai-data-diversity-v1/internal-residual-view-01/manifest.json'
    manifest = read(manifest_path)
    require(manifest['status'] == 'verified_all148_internal_residual_training_view'
            and manifest['all148training_scenes_in_objective']
            and not manifest['development_or_reserved_used']
            and not manifest['expanded_reserved52_references_opened']
            and not manifest['future_native_labels_used_for_features'], 'Original train-only view required')
    active = {row['identifier'] for row in manifest['scene_rows'] if row['teacher_substeps'] == 8}
    fixed = {row['identifier'] for row in manifest['scene_rows'] if row['teacher_substeps'] == 1}
    starts = [0,8,16,32,64,96,128,192,239]
    parts = manifest['parts']
    require(manifest['starts'] == starts and len(active) == 29 and len(fixed) == 119
            and len(parts) == 783 and len(manifest['control_rows']) == 1071
            and {(p['identifier'],p['start'],p['group_index']) for p in parts}
                == {(name,start,group) for name in active for start in starts for group in range(3)}
            and all(p['width'] == (2,2,3)[p['group_index']] for p in parts),
            'All original active group requests and fixed controls required')
    archive_path = manifest_path.parent/manifest['archive']['path']
    archive_spec = observed_file(ROOT,archive_path.relative_to(ROOT),manifest['archive'])
    members = [manifest['sparse_part'],*manifest['geometry_parts'],*parts]
    with zipfile.ZipFile(archive_path) as archive:
        require(len(archive.infolist()) == len(members)
                and set(archive.namelist()) == {p['member'] for p in members}
                and all(archive.getinfo(p['member']).file_size == p['bytes'] for p in members),
                'Complete original archive member inventory required')
    models,sources = [],{}
    for entry in selected:
        require(entry['parameter_dtypes'] == ['<f4'], 'Original F32 checkpoint required')
        for spec in [entry['original_receipt'],entry['original_configuration'],*entry['sources'],
                     *entry['outputs'].values(),*entry['inputs'].values()]:
            observed_file(ROOT,spec['path'],spec)
            sources[spec['path']] = spec['sha256']
        original = read(ROOT/entry['inputs']['manifest']['path'])
        # S1/S2 used the full view; S3/S4 retain a compact recipe referring to it.
        if 'original_manifest' in original:
            require(original['original_manifest'] == str(manifest_path.relative_to(ROOT))
                    and original['active_current_states'] == 261
                    and original['starts'] == starts, 'Compact fit must bind this same full training population')
        else:
            require(entry['inputs']['manifest']['sha256'] == binding(manifest_path)['sha256'],
                    'Original full-view fit must use this exact manifest')
        head_sources = {name:next(row for row in entry['sources'] if row['original_name'] == name)
                        for name in ('internal_residual_network_v1.py','internal_residual_network_v2.py',
                                     'internal_residual_loss_v1.py')}
        probe_path = (ROOT/entry['outputs']['weights.npz']['path']).parent/'probe.npy'
        receipt = read(ROOT/entry['original_receipt']['path'])
        probe = observed_file(ROOT,probe_path.relative_to(ROOT),receipt['outputs']['probe.npy'])
        history = read(ROOT/entry['outputs']['history.json']['path'])
        require(history['probe']['index'] == 0 and history['probe']['field_order']
                == ['pressure','vx','vy','vz','tracer'], 'Original full-field GPU probe required')
        models.append(dict(fit_item_id=entry['fit_item_id'],name=entry['name'],sources=head_sources,
            weights=entry['outputs']['weights.npz'],inventory=entry['selected_inventory'],
            probe=probe,probe_abi=history['probe'],history=entry['outputs']['history.json'],
            original_receipt=entry['original_receipt'],original_configuration=entry['original_configuration']))
        sources[probe['path']] = probe['sha256']
    require(all({name:value['sha256'] for name,value in m['sources'].items()}
                == {name:value['sha256'] for name,value in models[0]['sources'].items()} for m in models),
            'Shared original network/loss implementation must be byte-identical across all fits')
    for p in Path(__file__).parent.glob('*.py'):
        spec = binding(p);sources[spec['path']] = spec['sha256']
    for path in (manifest_path,material_path,publication,ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json'):
        spec = binding(path);sources[spec['path']] = spec['sha256']
    for spec in (selected[0]['inputs']['normalization'],selected[0]['inputs']['optimization_normalization']):
        require(spec['sha256'] == manifest['network_normalization_sha256' if '/network-' in spec['path']
                                         else 'optimization_normalization_sha256'], 'Frozen normalization identity differs')
    save(output,dict(status='original_four_internal_residual_heads_and783requests_bound_not_benchmarked',
        created_unix=time.time(),producer=binding(Path(__file__)),publication=binding(publication),
        material=binding(material_path),manifest=binding(manifest_path),archive=archive_spec,models=models,
        normalization=selected[0]['inputs']['normalization'],
        optimization_normalization=selected[0]['inputs']['optimization_normalization'],
        source_sha256=sources,expected_GPU='NVIDIA GeForce RTX 5090 Laptop GPU',host_cpu=4,threads=1,
        active_parents=29,active_current_states=261,active_requests_per_fit=783,
        fixed_bypass_parents=119,fixed_bypass_windows=1071,starts=starts,internal_widths=[2,2,3],
        by_cells_per_metre=dict(sorted(Counter(str(p['cells_per_metre']) for p in parts).items())),
        by_counts_xyz=dict(sorted(Counter('x'.join(map(str,p['counts_xyz'])) for p in parts).items())),
        input_contract='42F32cell-lattice channels from the original supplied fine-history internal group context. Only features enter the head;only h,dt,rho,base fields and predicted deltas enter the frozen coupled response. Targets are scoring-only.',
        output_contract='Four F32normalized carrier deltas;original frozen learned response emits pressure/velocity/tracer F64 for a2/2/3fine-substep group. No final complete-generator carrier cast is added.',
        comparison_scope='Original consumed training internal-request view;oracle stage-context component accuracy is not causal full-trajectory/full-step accuracy. Groups are partial internal advances,not3future physical outputs.',
        timing_units='GPU ms/residual-correction request;not GPU ms/physical simulation step',
        retained_corpus_read_in_place=True,archive_copied=False,archive_extracted=False,
        source_and_weight_changes=False,accuracy_qualified=False,latency_qualified=False,
        CPU_neural_forwards=0,new_parameter_fits=0,new_reference_generation=0,
        model_dataset_or_project_copies=0,native_mutations=0,full_portfolio_complete=False))
    print('Four original correction fits bound;783active requests each;original9.77GBcorpus verified in place.',flush=True)


if __name__ == '__main__':
    run()
