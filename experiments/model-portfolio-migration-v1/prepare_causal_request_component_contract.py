"""Select a matched original training population for two causal q-request heads."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require
from component_material import current_publication, observed_file


def run():
    output = RUN/'causal-request-component-contract-01.json'
    require(not output.exists(), 'Preserve original causal-component admission')
    publication,_ = current_publication(ROOT)
    material_path = RUN/'remaining-component-material-01.json';material = read(material_path)
    ids = ('fit-a9e199535346c3d9','fit-ccb969376d283836')
    selected = [];population = None
    for fid in ids:
        entry = next(row for row in material['models'] if row['fit_item_id'] == fid)
        for spec in [entry['original_receipt'],entry['original_configuration'],*entry['sources'],*entry['outputs'].values(),
                     entry['inputs']['manifest'],entry['inputs']['normalization']]:
            observed_file(ROOT,spec['path'],spec)
        config = read(ROOT/entry['original_configuration']['path'])
        manifest = read(ROOT/entry['inputs']['manifest']['path'])
        normalization = read(ROOT/entry['inputs']['normalization']['path'])
        require(config['horizons'] == 1 and entry['task'] == 'pressure_request_component'
                and entry['parameter_dtypes'] == ['<f4'] and manifest['status'] == 'verified'
                and not manifest['development_or_reserved_used'] and not manifest['current_pressure_input']
                and not manifest['reference_pressure_input'], 'Original one-step causal q-component contract required')
        current = {key:manifest[key] for key in ('parents','shards','archive_sha256','archive_bytes','normalization_sha256')}
        if population is None:
            population = current
        require(current == population, 'Both original heads must share exact source population and normalization')
        require(len(manifest['parents']) == 16 and all(parent['split'] == 'train' and parent['states'] == 129 for parent in manifest['parents']),
                'All16original129state training parents required')
        selected.append(dict(fit_item_id=fid,parameters=entry['serialized_tensor_elements'],configuration=entry['original_configuration'],
            weights=entry['outputs']['weights.npz'],manifest=entry['inputs']['manifest'],normalization=entry['inputs']['normalization'],
            scales=normalization['scales'],shards=[entry['inputs'][f'part_{i:02d}'] for i in range(9)]))
    save(output,dict(status='two_causal_q_components_matched2048transition_contract_admitted_adapter_pending',created_unix=time.time(),
        material=binding(material_path),publication_acceptance=binding(publication),fits=selected,
        original_scene_count=16,transition_cases=2048,current_indices='0..127for each129state originalparent',horizons=1,
        population='All2048retained fitting transitions;not independentholdout. Fourgridcontexts,cpm8and16. Trainingcodecshards remain in place;decode213993244bytearchive inRAM,never materialize another dataset.',
        input_contract='Original data.inputs API:normalizedcurrentvelocityF32,normalizedknownforcingF32,physicalscalars[h,dt,rho,mu]F32. No currentpressure orfuturepressure/q labels forwarded.',
        timing_scope='OwnedCPUalready-normalizedinputarrays,validation/packing/H2D,unchanged capturedCausalRequestNet,ownedCPUF32normalizedfutureq/D2H/sync. Data decoding,originalnormalization andknownforceprogram construction separate. This is a componentrequest,not a fullsimulationstep or150CNNpressure-request boundary.',
        output_contract='NormalizedfutureqF32,one lead;relativeerror against originalnormalizedtrainingtargets. Physicalqabsoluteerrors mayuse declaredfixedqscale forreporting only. No pressureinverse oradvection ininference.',
        precision_limit='Source labels arequantizedF32nativeq;manifest reportsworst q quantizationrelativeL2=3.7352611838562534e-8. Do notmistakeagreement withthese labelsforF64nativepressure precision.',
        planned_fresh_GPU_processes=4,planned_phases_per_fit=2,planned_primary_samples_per_case=10,
        planned_warmups_per_case=5,planned_paired_calibrations_per_case=4,
        CPU_neural_forward_allowed=False,reference_labels_forwarded=False,read_field_arrays=False,new_model_forwards=0,
        new_parameter_fits=0,new_reference_generation=False,weight_or_dataset_copies=0,benchmark_coverage_increased=False,
        producer=binding(Path(__file__)),full_portfolio_complete=False))
    print('Twooriginalone-step qheads share2048fitting transitions;actualdecoder/GPUadapter qualification pending.',flush=True)


if __name__ == '__main__':
    run()
