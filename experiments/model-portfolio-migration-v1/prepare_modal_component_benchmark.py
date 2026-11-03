"""Freeze source, original native frames and the four-fit CUDA measurement plan."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require
from component_material import current_publication, observed_file


def run():
    publication,_ = current_publication(ROOT)
    protocol_path = RUN/'modal-component-GPU-protocol-01.json'
    output = RUN/'modal-component-GPU-benchmark-01'
    require(not protocol_path.exists() and not output.exists(), 'Fresh original modal protocol required')
    contract_path = RUN/'modal-component-comparison-contract-01.json'
    contract = read(contract_path)
    require(contract['status'] == 'four_modal_component_matched_native_field_contract_admitted_adapter_pending'
            and len(contract['fits']) == 4 and len(contract['rows']) == 144, 'Original admitted four-fit contract required')
    material = read(ROOT/contract['material']['path'])
    sources = {}
    for path in Path(__file__).parent.glob('*.py'):
        spec = binding(path);sources[spec['path']] = spec['sha256']
    sources[str(contract_path.relative_to(ROOT))] = binding(contract_path)['sha256']
    sources[contract['material']['path']] = contract['material']['sha256']
    ids = [row['fit_item_id'] for row in contract['fits']]
    for entry in material['models']:
        if entry['fit_item_id'] not in ids:
            continue
        for spec in [entry['original_receipt'],entry['original_configuration'],*entry['sources'],*entry['outputs'].values(),
                     entry['inputs']['manifest'],entry['inputs']['normalization']]:
            observed_file(ROOT,spec['path'],spec);sources[spec['path']] = spec['sha256']
    for row in contract['rows']:
        observed_file(ROOT,row['source'],dict(sha256=row['source_sha256']))
        sources[row['source']] = row['source_sha256']
        receipt = (ROOT/row['source']).parent/'receipt.json'
        spec = binding(receipt);sources[spec['path']] = spec['sha256']
    operator_path = ROOT/'runs/pressure-methods-v4/v4-student50-control-project-01/portable_model.py'
    spec = binding(operator_path);sources[spec['path']] = spec['sha256']
    previous = read(RUN/'point-component-GPU-protocol-01.json')
    for name,sha in previous['source_sha256'].items():
        if Path(name).is_absolute():
            sources[name] = sha
    runtime = ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json'
    save(protocol_path,dict(schema='rheon.modal-component-GPU-benchmark.v1',created_unix=time.time(),
        contract=binding(contract_path),publication_acceptance=binding(publication),runtime_manifest=binding(runtime),
        source_sha256=sources,fit_ids=ids,phase_order=[dict(fit_item_id=fid,phase=phase) for fid in ids for phase in (0,1)],
        output_root=str(output.relative_to(ROOT)),worker='experiments/model-portfolio-migration-v1/modal_component_benchmark.py',
        expected_GPU='NVIDIA GeForce RTX 5090 Laptop GPU',host_cpu=4,threads=1,cases_per_fit=144,
        warmups_per_case=5,samples_per_case=10,calibration_pairs_per_case=4,
        worker_deadline_seconds=600,aggregate_deadline_seconds=5400,minimum_free_disk_bytes=1024**3,
        maximum_CUDA_allocated_bytes=2*1024**3,maximum_RSS_bytes=8*1024**3,
        prediction_dtype='float64',tf32=False,autocast=False,deterministic=True,
        timing_scope=contract['execution_boundary'],timing_units='ms/native_RHS_pressure_request',
        case_order='Canonical0..143then143..0 in two fresh processes for each original fit',
        primary_observer_removed=True,paired_observer_overhead_subtracted=False,
        resource_scope='Before/every36cases/after snapshots;not continuous GPU exclusivity',
        CPU_neural_forward_allowed=False,new_model_fits=0,new_reference_generation=False,
        weight_or_dataset_copies=0,producer=binding(Path(__file__)),full_portfolio_complete=False))
    print('Four-modal/eight-worker/144native-frame GPU protocol frozen; no neural execution.',flush=True)


if __name__ == '__main__':
    run()
