"""Freeze a matched shared-spectral decoder/cache comparison, without inference."""
import argparse
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from component_material import current_publication, observed_file


def run(protocol_name='shared-component-GPU-protocol-01.json', output_name='shared-component-GPU-benchmark-01', reconciliation_path=None):
    publication,_ = current_publication(ROOT)
    require(Path(protocol_name).name == protocol_name and Path(output_name).name == output_name, 'Owned protocol/output names required')
    protocol_path = RUN/protocol_name
    output = RUN/output_name
    require(not protocol_path.exists() and not output.exists(), 'Fresh original shared protocol required')
    contract_path = RUN/'modal-component-comparison-contract-01.json';contract = read(contract_path)
    material = read(ROOT/contract['material']['path']);fid = 'fit-9c914f77b1f2ec81'
    entry = next(row for row in material['models'] if row['fit_item_id'] == fid)
    require(entry['serialized_tensor_elements'] == 4 and entry['parameter_dtypes'] == ['<f8'], 'Original four F64 coefficients required')
    original_protocol = DOC.parent/'full-ai-data-diversity-v1/shared-spectral-qualification-protocol-01.json'
    original = read(original_protocol)
    model = next(row for row in entry['sources'] if row['original_name'] == 'shared_spectral_model_v1.py')
    basis_path = ROOT/'experiments/full-ai-tight-targets-v1/modal_basis.py';basis = binding(basis_path)
    require(original['source_sha256'][model['path']] == model['sha256']
            and original['source_sha256'][basis['path']] == basis['sha256'], 'Exact original qualified model/basis code required')
    manifest = read(ROOT/entry['inputs']['manifest']['path'])
    named_overlap = set(manifest['source_scene_identifiers']) & {row['identifier'] for row in contract['rows']}
    archive_overlap = set(manifest['source_evidence'].values()) & {row['source_sha256'] for row in contract['rows']}
    require(not named_overlap and not archive_overlap and manifest['training_scenes'] == 148
            and manifest['mode_rows'] == 535394, 'Original selected training closure differs')
    sources = {}
    for path in Path(__file__).parent.glob('*.py'):
        spec = binding(path);sources[spec['path']] = spec['sha256']
    for spec in [entry['original_receipt'],entry['original_configuration'],*entry['sources'],*entry['outputs'].values(),
                 entry['inputs']['manifest'],entry['inputs']['normalization'],basis,binding(contract_path),
                 contract['material'],binding(original_protocol)]:
        observed_file(ROOT,spec['path'],spec);sources[spec['path']] = spec['sha256']
    previous = read(RUN/'modal-component-GPU-protocol-01.json')
    names = {row['source'] for row in contract['rows']}
    names.update(str((ROOT/name).parent.relative_to(ROOT)/'receipt.json') for name in list(names))
    names.add('runs/pressure-methods-v4/v4-student50-control-project-01/portable_model.py')
    for name in names:
        spec = dict(sha256=previous['source_sha256'][name]);observed_file(ROOT,name,spec);sources[name] = spec['sha256']
    for name,sha in previous['source_sha256'].items():
        if Path(name).is_absolute():
            sources[name] = sha
    reconciliation = None
    if reconciliation_path is not None:
        reconciliation = read(reconciliation_path)
        require(reconciliation['status'] == 'original_shared_mode_shadowing_source_correction_verified', 'Verified source correction required')
        spec = binding(reconciliation_path);sources[spec['path']] = spec['sha256']
    runtime = ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json'
    save(protocol_path,dict(schema='rheon.shared-component-GPU-benchmark.v1',created_unix=time.time(),
        contract=binding(contract_path),publication_acceptance=binding(publication),runtime_manifest=binding(runtime),
        source_sha256=sources,fit_ids=[fid],selected=dict(fit_item_id=fid,weights=entry['outputs']['weights.npz'],
            model=model,basis=basis,manifest=entry['inputs']['manifest'],normalization=entry['inputs']['normalization']),
        original_qualification_protocol=binding(original_protocol),modes=['unprepared','prepared'],
        reconciliation=binding(reconciliation_path) if reconciliation_path is not None else None,
        reconciled_failure=reconciliation['failed_result'] if reconciliation is not None else None,
        phase_order=[dict(fit_item_id=fid,mode=mode,phase=phase) for mode in ('unprepared','prepared') for phase in (0,1)],
        output_root=str(output.relative_to(ROOT)),worker='experiments/model-portfolio-migration-v1/shared_component_benchmark.py',
        expected_GPU='NVIDIA GeForce RTX 5090 Laptop GPU',host_cpu=4,threads=1,cases_per_fit=144,
        warmups_per_case=5,samples_per_case=10,calibration_pairs_per_case=4,
        worker_deadline_seconds=600,aggregate_deadline_seconds=3000,minimum_free_disk_bytes=1024**3,
        maximum_CUDA_allocated_bytes=2*1024**3,maximum_RSS_bytes=8*1024**3,prediction_dtype='float64',
        tf32=False,autocast=False,deterministic=True,primary_observer_removed=True,
        basis_policy='Exact original standaloneCUDAqualification bases(shape,cuda). Both cache modes useidentical GPUbases. Do not claim bitwiseparity withfullgenerator variants that build their bases onCPU.',
        timing_scope='Owned nativeCPUF64 RHS/units,validation/pack/H2D,q=RHS*rho/h,unchanged captured SharedSpectralResponse.decode,ownedCPUF64pressure/D2H/sync. No reference labels forwarded. FixedGPUbasis preparation separate.',
        response_policy='Unprepared:original decode constructs learned response everyrequest. Prepared:four CUDAresponse_grid calls in separatelytimed setup;reuse unchanged responses,withoriginaldecode clone/DCfix/reconstruction stilltimed. Requireall144outputs exactacross modes/processes;retainone archive.',
        observer_policy='Unprepared request invokesSharedSpectralResponse.forward;prepareddecodeusesprecomputedCUDAresponse andhasnoNNforwardcallback inside request. Setup forwards observedCUDA. Pairon/offtimingsstillretained;prepared differences aretimingnoise,notevidenceof callbackcost.',
        timing_units='ms/native_RHS_pressure_request',physical_simulation_steps=0,paired_observer_overhead_subtracted=False,
        resource_scope='Before/every36cases/after compute snapshots;not continuous exclusivity',
        population='144legacy native frames/16scenes/fourgrids/cpm8and16. Selectedfit training closure148scenes/1184archives/535394feature rows haszero namedscene/archiveSHA overlap;posthocdiagnostic,not independentholdout.',
        training_scene_name_overlap=0,training_archive_SHA_overlap=0,CPU_neural_forward_allowed=False,
        new_model_fits=0,new_reference_generation=False,weight_or_dataset_copies=0,
        producer=binding(Path(__file__)),full_portfolio_complete=False))
    print('Shared four-F64-coefficient decoder/two cache policies/fourfreshCUDAphases frozen;no inference.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-name',default='shared-component-GPU-protocol-01.json')
    parser.add_argument('--output-name',default='shared-component-GPU-benchmark-01')
    parser.add_argument('--reconciliation',type=Path)
    args = parser.parse_args()
    run(args.protocol_name,args.output_name,args.reconciliation.resolve() if args.reconciliation else None)
