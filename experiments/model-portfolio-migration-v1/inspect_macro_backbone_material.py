"""Read-only original backbone/field closure inventory; no model imports."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, RUN, read, save, binding, require
from component_material import observed_file


def run():
    output = RUN/'macro-readout-backbone-material-01.json'
    require(not output.exists(), 'Preserve original backbone readiness observation')
    contract_path = RUN/'macro-readout-component-contract-01.json'
    contract = read(contract_path)
    protocol_path = ROOT/contract['original_protocol']['path']
    require(binding(protocol_path) == contract['original_protocol'], 'Original loss-view protocol changed')
    protocol = read(protocol_path)
    original, changed, missing = [], [], []
    for name, expected in protocol['source_sha256'].items():
        path = ROOT/name
        if not path.is_file():
            missing.append(dict(path=name,original_sha256=expected));continue
        actual = observed_file(ROOT,name)
        (original if actual['sha256'] == expected else changed).append(dict(actual,original_sha256=expected))
    references = {}
    receipts = {}
    for case in contract['cases']:
        receipt = case['receipt']
        if receipt not in receipts:
            receipts[receipt] = observed_file(ROOT,receipt)
        for key in ('current_state','target_state'):
            spec = case[key]
            current = observed_file(ROOT,spec['path'],spec)
            require(all(value['dtype'] == ('<f8' if name in ('pressure','rhs','native_input_residual','pressure_base') else '<f4')
                        for name,value in spec['fields'].items()), 'Original typed native field metadata required')
            references[spec['path']] = current
    require(len(contract['cases']) == 58 and len(references) == 116 and len(receipts) == 29,
            'Exact original29parents/two current-target windows required')
    result = dict(status='original_macro_backbone_material_inventory_complete_requires_import_closure_qualification',
        created_unix=time.time(),producer=binding(Path(__file__)),contract=binding(contract_path),
        original_protocol=binding(protocol_path),original_source_files=len(protocol['source_sha256']),
        unchanged_original_sources=original,changed_original_sources=changed,missing_original_sources=missing,
        active_current_target_archives=list(references.values()),active_native_receipts=list(receipts.values()),
        active_current_target_archive_bytes=sum(row['bytes'] for row in references.values()),
        all_active_native_archive_bytes_match=True,field_array_contents_decoded=False,
        interpretation='The historical protocol freezes a broad1698file population. Changed or missing unrelated files '
            'are not permission to change a causal dependency. Every actual imported helper, constructor, '
            'kernel build and parameter read must still match its original closure or receive an explicit disposition.',
        required_next='Select and prove the complete actually used source/build/parameter closure before CUDA restoration; '
            'then verify decoded field hashes,shape,dtype and current/target isolation on all58windows.',
        numpy_imported=False,torch_imported=False,new_model_forwards=0,CPU_neural_forwards=0,
        new_reference_generation=False,model_dataset_or_project_copies=0,
        accuracy_or_timing_qualified=False,default_benchmark_coverage_increased=False,full_portfolio_complete=False)
    save(output,result)
    print('Original broad sources:',len(original),'unchanged,',len(changed),'changed,',len(missing),'missing;',
          len(references),'active native archives hash-verified in place.',flush=True)


if __name__ == '__main__':
    run()
