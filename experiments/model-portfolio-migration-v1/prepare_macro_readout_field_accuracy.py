"""Freeze original causal stage assets and all active readout windows."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, RUN, read, save, binding, require
from component_material import current_publication, observed_file


def run():
    output = RUN/'macro-readout-field-GPU-accuracy-protocol-01.json'
    require(not output.exists(), 'Preserve original field accuracy registration')
    publication,_ = current_publication(ROOT)
    contract_path = RUN/'macro-readout-component-contract-01.json'
    readiness_path = RUN/'macro-readout-backbone-material-01.json'
    contract,readiness = read(contract_path),read(readiness_path)
    original_path = ROOT/contract['original_protocol']['path']
    original = read(original_path)
    require(readiness['all_active_native_archive_bytes_match']
            and len(readiness['unchanged_original_sources']) == 1698
            and not readiness['changed_original_sources'] and not readiness['missing_original_sources'],
            'The entire original frozen backbone/source population must match')
    sources = dict(original['source_sha256'])
    for path in Path(__file__).parent.glob('*.py'):
        spec = binding(path);sources[spec['path']] = spec['sha256']
    material_path = RUN/'remaining-component-material-01.json'
    entries = {row['fit_item_id']:row for row in read(material_path)['models']}
    models = []
    for fid in contract['fit_ids']:
        entry = entries[fid]
        source = next(row for row in entry['sources'] if row['original_name'] == 'macrostep_readout_v1.py')
        stages = read(ROOT/entry['original_configuration']['path'])['stages']
        models.append(dict(fit_item_id=fid,stages=stages,source=source,weights=entry['outputs']['weights.npz'],
            inventory=entry['selected_inventory'],original_design=entry['inputs']['part_00']))
        for spec in [entry['original_receipt'],entry['original_configuration'],*entry['sources'],*entry['outputs'].values(),
                     *entry['inputs'].values()]:
            observed_file(ROOT,spec['path'],spec);sources[spec['path']] = spec['sha256']
    for spec in [*readiness['active_current_target_archives'],*readiness['active_native_receipts']]:
        sources[spec['path']] = spec['sha256']
    for path in (contract_path,readiness_path,original_path,material_path,
                 ROOT/contract['original_progress']['path'],ROOT/contract['original_bank']['path'],publication):
        spec = binding(path);sources[spec['path']] = spec['sha256']
    old_modal = read(RUN/'modal-component-GPU-protocol-01.json')
    for name,sha in old_modal['source_sha256'].items():
        if Path(name).is_absolute():
            sources[name] = sha
    for name,sha in sources.items():
        with (ROOT/name).open('rb') as stream:
            import hashlib
            require(hashlib.file_digest(stream,'sha256').hexdigest() == sha, 'Frozen macro field input changed: '+name)
    save(output,dict(status='registered_before_original_macro_readout_field_GPU_accuracy',created_unix=time.time(),
        producer=binding(Path(__file__)),contract=binding(contract_path),readiness=binding(readiness_path),
        original_protocol=binding(original_path),source_sha256=sources,models=models,
        normalization=entries[contract['fit_ids'][0]]['inputs']['normalization'],
        backbone_arguments=[original[key] for key in ('transport_pair','shared_pair','build','control_build')],
        expected_GPU='NVIDIA GeForce RTX 5090 Laptop GPU',host_cpu=4,threads=1,
        output_root='runs/model-portfolio-migration-v1/macro-readout-field-GPU-accuracy-01',
        cases_per_fit=58,fixed_bypass_windows=238,physical_output_dt_s=1/60,physical_outputs_per_case=1,
        phases=[0,1],worker_deadline_seconds=900,maximum_CUDA_allocated_bytes=8*1024**3,
        maximum_RSS_bytes=8*1024**3,primary_timing_samples=0,latency_qualified=False,
        concurrent_compute_allowed_for_accuracy=True,
        native_targets_forwarded_to_causal_generation=False,
        objective_diagnostic='Original QR factors recomputed on CUDA only after causal outputs; targets stay outside the predictor. Factor parity is recorded separately from physical-field metrics.',
        constructor_policy='Unchanged original constructor under CUDA default-device scope; all module params/buffers checked CUDA and all module forwards guarded. CPU typed artifact decode/static quantization,CSR metadata,fixed basis and known scalar packets retain original setup arithmetic. No CPU state-dependent learned stage.',
        readout_policy='Exact raw published F64coefficients; do not apply MacrostepStudent2^-40readout rounding.',
        source_reference_population='29consumed N8training parents,two starts32/96;not independentholdout or drift.119N1parents/238windows explicitly not_applied to the learned field readout.',
        worker_score='Original captured Torch full-field diagnostic on owned CPU results;independent NumPy physical field/equation reduction required.',
        independent_metric_tolerance=dict(relative=1e-10,absolute=1e-12),
        new_optimizer_updates=0,new_reference_generation=False,model_dataset_or_project_copies=0,
        full_portfolio_complete=False))
    print('Two original raw readouts ×58active field windows ×two fresh phases registered;accuracy only.',flush=True)


if __name__ == '__main__':
    run()
