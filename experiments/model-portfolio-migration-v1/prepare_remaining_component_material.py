"""Bind20original component contracts in place;do not import a model runtime.

Byte hashing includes original data archives,streamed once per unique file.
No array is decoded,no model is executed,and no source/data/checkpoint copied.
"""
import ast
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

from import_common_generator_measurements import ROOT,RUN,DOC,PROJECT,read,save,binding
from component_material import require,observed_file,selected_inventory,current_publication

OUTPUT = RUN/'remaining-component-material-01.json'
REPORT = DOC/'reports/remaining-component-material-readiness-01.json'
CORE_PROJECTS = {
    'fit-11a2f63a94a9168e':['runs/full-ai-tight-targets-v1/sampler-head-project-02','runs/full-ai-tight-targets-v1/sampler-head-project-03'],
    'fit-9c914f77b1f2ec81':['runs/full-ai-data-diversity-v1/shared-spectral-project-01'],
}
ROLE_CONTRACTS = {
    'viscosity_component':dict(output='Dimensionless negative symmetric-strain stiffness on free face degrees of freedom.',
        scoring='Per-pattern masked vector relativeL2,absoluteRMS/max,targetnorm anddenominator-floor flag. Training64/development16 reported separately.',
        population='Original80native-qualified synthetic teacher patterns;not temporal simulation scenes.',
        latency_unit='ms/local_viscosity_request',physical_simulation_steps=0),
    'transport_component':dict(output='Negative sampled point velocity:four destination kinds times xyz channels.',
        scoring='Per-pattern/per-destination masked vector relativeL2 andabsolute errors;original teacher masks/splits.',
        population='Original native-qualified point-sampling teacher patterns;not a full particle backtrace or state advance.',
        latency_unit='ms/local_sampling_request',physical_simulation_steps=0),
    'recurrent_transport_component':dict(output='Middle displacement and recurrent endpoint displacement from point,patch,kind andalpha.',
        scoring='Separate middle/endpoint vector relativeL2 andabsolute errors byscene,destination kind andphysical displacement scale.',
        population='Original train-only recurrent-sampler feature/label view;stage context is supplied,not a complete simulation advance.',
        latency_unit='ms/recurrent_sampling_request',physical_simulation_steps=0),
    'pressure_request_component':dict(output='Mean-free futureq pressure request from current velocity andknown controls.',
        scoring='Same-target future-request relativeL2 andabsolute errors;report labelF32quantization,dt,grid,cells/m andscene. Futureq/pressure are scoring-only.',
        population='Original causal-request native trajectory view,with original1/60physical horizon;no future labels forwarded.',
        latency_unit='ms/causal_pressure_request_prediction',physical_simulation_steps=0),
    'pressure_component':dict(output='Captured local modal/pressure response;decode or solver composition is separately labeled.',
        scoring='Preserve actual per-model target/operator. Modal response,decoded pressure andfull hierarchy accuracy are distinct. Bootstrap cohorts retain anisotropic spacing/fluid density.',
        population='Select compatible original native targets perinterface;do not silently force variable-grid128requests onto fixed16cubed contracts.',
        latency_unit='ms/captured_pressure_component_request',physical_simulation_steps=0),
    'full_state_correction_component':dict(output='Normalized state/residual correction using captured baseline/stage features.',
        scoring='Per-field correction relativeL2/absolute errors withoriginal normalization andface masks. Supplied baseline/context andunchanged bypasses explicit.',
        population='Original internal-residual view;oracle/stage-context component score is not a causal complete-generator score.',
        latency_unit='ms/residual_correction_request',physical_simulation_steps=0),
    'full_state_readout_component':dict(output='Learned stage/readout mixture;full stage generation cost is outside a bare readout call.',
        scoring='Original readout response plusfield reconstruction when available;label Gram/loss-view scores versus physical fields separately.',
        population='Original macrostep readout view;preserve3/4stage source andactual physical horizon without inventing intermediate advances.',
        latency_unit='ms/stage_readout_request',physical_simulation_steps=0),
    'guided_pressure_component':dict(output='Pressure-guided residual over the inherited captured base generator.',
        scoring='Use original guided runtime andtrial/state selection. Separate learned residual fromcomplete base+residual prediction;no future labels in causal input.',
        population='Original24scene guided training archive andcaptured model-source archive;not a new held-out population.',
        latency_unit='ms/declared_guided_request',physical_simulation_steps=None),
    'guided_shared_component':dict(output='Shared guided residual over the inherited captured base generator.',
        scoring='Use original shared runtime andtrial/state selection. Separate residual cost/error fromcomplete composition;keep masks/known controls explicit.',
        population='Original24scene guided training archive andcaptured model-source archive;not a new held-out population.',
        latency_unit='ms/declared_guided_request',physical_simulation_steps=None),
}


def specs(value):
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and 'sha256' in value:
            yield value
        else:
            for child in value.values():
                yield from specs(child)
    elif isinstance(value,list):
        for child in value:
            yield from specs(child)


def interfaces(path):
    tree = ast.parse(path.read_bytes())
    wanted = {'__init__','forward','predict','predict_pressure','decode','reconstruct','response_grid','construct','features','sample'}
    rows = []
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in wanted:
            rows.append(dict(name=node.name,line=node.lineno,arguments=ast.unparse(node.args)))
    return rows


def run():
    require(not OUTPUT.exists() and not REPORT.exists(),'Fresh component material required;preserve original attempts')
    publication,header = current_publication(ROOT)
    index_path = PROJECT/'notes/portfolio-index.json'
    index_binding = binding(index_path)
    index = read(index_path)
    coverage_path = DOC/'reports/common-fit-benchmark-coverage-03.json'
    coverage = read(coverage_path)
    pending = set(coverage['pending_fit_ids'])
    fits = {fit['item_id']:fit for fit in index['fits']}
    require(len(pending) == 20 and len(fits) == 205 and pending <= fits.keys(),'Exact20remaining/205fit population required')
    reviewed = {}
    report_bindings = {}
    for key in ('method_reviews','method_review_batches','method_review_addenda'):
        for spec in specs(index['sources'].get(key)):
            require(binding(ROOT/spec['path']) == {k:spec[k] for k in ('path','sha256')},'Published source-review authority changed')
            report_bindings[spec['path']] = {k:spec[k] for k in ('path','sha256')}
            for row in read(ROOT/spec['path']).get('rows',[]):
                fid = row.get('fit_item_id')
                if fid in pending:
                    require(fid not in reviewed or reviewed[fid][1] == row,'Conflicting original component reviews')
                    reviewed[fid] = (report_bindings[spec['path']],row)
    require(reviewed.keys() == pending,'Every remaining component needs its original reviewed receipt')
    observed = {}

    def bind(path,expected=None):
        relative = Path(path)
        full = ROOT/relative
        if full.is_symlink():
            relative = full.resolve().relative_to(ROOT)
        key = str(relative)
        if key not in observed:
            observed[key] = observed_file(ROOT,relative,expected)
        result = observed[key]
        if expected:
            require(result['sha256'] == expected['sha256'] and ('bytes' not in expected or result['bytes'] == expected['bytes']),
                    'Original material binding differs: '+key)
        return dict(result)

    def resolve(expected,candidates):
        for value in candidates:
            path = Path(value)
            if (ROOT/path).is_file():
                # A named but changed current helper is not an original captured
                # source. Continue only to explicitly content-bound keepers.
                result = bind(path)
                if result['sha256'] == expected['sha256'] and ('bytes' not in expected or result['bytes'] == expected['bytes']):
                    return result
        keeper = PROJECT/'artifacts/sha256'/expected['sha256']
        require(keeper.is_file(),'Original component material unavailable: '+str(candidates))
        return bind(keeper.relative_to(ROOT),expected)

    rows = []
    for fid in sorted(pending):
        authority,review = reviewed[fid]
        fit = fits[fid]
        require(fit['task'] in ROLE_CONTRACTS,'Component role has no scientific contract')
        receipt_binding = bind(review['original_receipt']['path'],review['original_receipt'])
        receipt = read(ROOT/receipt_binding['path'])
        require(receipt_binding['sha256'] == fit['receipt_sha256'] and receipt['mode'] == 'train'
                and receipt.get('optimizer_updates',0) > 0,'Original indexed training receipt required;no restore fabrication')
        if receipt.get('status') != 'complete':
            require(receipt['schema'] in ('rheon.guided-pressure-managed-fit.v1','rheon.guided-shared-managed-fit.v1')
                    and receipt['parameter_fits'] == 1 and receipt['accepted_steps'] > 0,'Schema-specific completed guided fitting evidence required')
        closure = receipt['closure']
        config_spec = review.get('original_configuration') or review.get('training_configuration')
        project_candidates = []
        if config_spec:
            config_binding = resolve(config_spec,[config_spec['path']])
            project_candidates.append(str(Path(config_spec['path']).parent))
        else:
            project_candidates = CORE_PROJECTS[fid]
            config_binding = resolve(dict(sha256=closure['config_sha256']),[str(Path(p)/'config.json') for p in project_candidates])
        config = read(ROOT/config_binding['path'])
        require(config_binding['sha256'] == closure.get('config_sha256',closure.get('experiment_sha256')),
                'Original configuration closure differs')
        documents = []
        for project in project_candidates:
            path = ROOT/project/'project.json'
            if path.is_file():
                documents.append((path.parent,read(path),bind(path.relative_to(ROOT))))
        source_specs = review.get('captured_source_artifacts') or review.get('sources',[])
        expected_sources = closure.get('sources')
        witness_binding = None
        if expected_sources is None:
            witness_binding = bind(review['source_witness']['path'],review['source_witness'])
            witness = read(ROOT/witness_binding['path'])
            require(witness_binding['sha256'] == closure['source_witness_sha256'],'Original source witness differs')
            expected_sources = witness['sources']
        sources = []
        for name,sha in sorted(expected_sources.items()):
            candidates = [item['path'] for item in source_specs if item.get('original_name',Path(item['path']).name) == name and item['sha256'] == sha]
            candidates += [str(Path(project)/name) for project in project_candidates]
            spec = resolve(dict(sha256=sha),candidates)
            sources.append(dict(spec,original_name=name,interfaces=interfaces(ROOT/spec['path']) if name.endswith('.py') else []))
        output_dir = Path(receipt_binding['path']).parent
        outputs = {}
        for name in ('weights.npz','normalization.json','history.json'):
            expected = receipt['outputs'][name]
            candidates = [str(output_dir/name)]
            if name == 'history.json':
                candidates.insert(0,review['history']['path'])
            outputs[name] = resolve(expected,candidates)
        require(any(row['stage'] == 'selected' and row['sha256'] == outputs['weights.npz']['sha256'] for row in fit['checkpoints']),
                'Selected original archive does not belong to indexed fit')
        history = read(ROOT/outputs['history.json']['path'])
        inventory,inventory_owner = selected_inventory(receipt,history,outputs['weights.npz']['sha256'])
        input_candidates = {}
        for item in review.get('training_inputs',[]):
            input_candidates.setdefault(item['slot'],[]).append(item['path'])
        for base,document,_ in documents:
            for item in document.get('data',[]):
                path = Path(item['path'])
                full = ROOT/path if path.parts[0] in ('runs','projects','experiments') else base/path
                input_candidates.setdefault(item['id'],[]).append(str(full.relative_to(ROOT)))
        if review.get('training_archive_path'):
            input_candidates.setdefault('part_00',[]).append(review['training_archive_path'])
        inputs = {}
        for slot,expected in closure['inputs'].items():
            inputs[slot] = resolve(expected,input_candidates.get(slot,[]))
        require('normalization_sha256' not in closure or closure['normalization_sha256'] in
                [outputs['normalization.json']['sha256'],inputs.get('normalization',{}).get('sha256')],
                'Original normalization closure differs')
        metadata = {}
        for slot,spec in inputs.items():
            if slot in ('manifest','teacher_manifest','splits','normalization','optimization_normalization'):
                value = read(ROOT/spec['path'])
                metadata[slot] = {key:value[key] for key in ('schema','status','counts_xyz','target','train_indices','development_indices',
                    'development_or_reserved_used','normalization_sha256','worst_q_quantization_relative_l2') if key in value}
                for key in ('inventory','arrays'):
                    if isinstance(value.get(key),dict):
                        metadata[slot][key] = {name:{k:array[k] for k in ('shape','dtype') if k in array}
                            for name,array in value[key].items() if isinstance(array,dict)}
        contract = dict(ROLE_CONTRACTS[fit['task']],status='source_role_bound_adapter_and_scorer_not_yet_qualified',
            pressure_error_is_equation_residual=False,independent_holdout=False,CPU_neural_forward_allowed=False,
            future_targets_forwarded=False,setup_and_warmup_separate_from_primary=True,
            primary_timing_observer_removed=True,paired_observer_calibration_separate=True,
            density_and_grid_dimensions_separate=True,source_precision_and_parameter_values_preserved=True,
            GPU_constructor_restore_publication_qualification_required=True)
        if receipt['schema'] == 'rheon-local-pressure-training-receipt-v1':
            contract.update(family='bootstrap_anisotropic_pressure',fixed_counts_xyz=[16,16,16],fluid_density_kg_m3=1.25,
                adaptation='Original CPU-only physical/operator guards require a separately qualified GPU execution variant;no CPU neural fallback.')
        rows.append(dict(fit_item_id=fid,name=fit['name'],task=fit['task'],checkpoint_origin='original_train_output',
            original_receipt=receipt_binding,receipt_schema=receipt['schema'],receipt_status=receipt.get('status'),
            original_configuration=config_binding,configuration_location_kind='reviewed' if config_spec else 'byte-identical captured project candidates',
            matching_project_candidates=project_candidates,project_documents=[spec for _,_,spec in documents],
            source_review=authority,source_witness=witness_binding,sources=sources,outputs=outputs,inputs=inputs,
            selected_inventory=inventory,selected_inventory_owner=inventory_owner,
            serialized_tensor_elements=sum(math.prod(spec['shape']) for spec in inventory.values()),
            parameter_dtypes=sorted({spec['dtype'] for spec in inventory.values()}),input_metadata=metadata,benchmark_contract=contract,
            original_method=review['method'],original_purpose=review['purpose'],GPU_adapter_qualified=False,accuracy_qualified=False,latency_qualified=False))
        print(fid,fit['task'],'original material androle bound',flush=True)
    require(len(rows) == len({row['fit_item_id'] for row in rows}) == 20 and binding(index_path) == index_binding,
            'Complete20fit population andunchanged index required')
    require('torch' not in sys.modules and 'numpy' not in sys.modules,'Readiness must not import model/array runtimes')
    save(OUTPUT,dict(status='all20_original_component_material_and_source_roles_bound',created_unix=time.time(),root=str(ROOT),models=rows,
        source_fit_index=index_binding,coverage=binding(coverage_path),publication_acceptance=binding(publication),
        unique_original_files_observed=len(observed),unique_original_bytes_hashed=sum(v['bytes'] for v in observed.values()),
        byte_observation='Opaque data/weight/source bytes streamed in place,once perunique path;no array decoding or atomic whole-corpus snapshot claim. Actual GPU adapters must freshly verify their selected material.',
        producer=binding(Path(__file__)),helper=binding(Path(__file__).with_name('component_material.py')),
        new_model_forwards=0,new_parameter_updates=0,new_reference_generation=0,native_mutations=0,model_or_dataset_copies=0,
        GPU_adapters_qualified=0,full_portfolio_complete=False))
    save(REPORT,dict(status='20component_original_material_ready_for_role_specific_GPU_adapter_qualification',created_unix=time.time(),fits=20,
        by_task=dict(Counter(row['task'] for row in rows)),parameter_dtypes=dict(Counter(dtype for row in rows for dtype in row['parameter_dtypes'])),
        unique_original_files_observed=len(observed),unique_original_bytes_hashed=sum(v['bytes'] for v in observed.values()),
        material=binding(OUTPUT),producer=binding(Path(__file__)),source_files=sum(len(row['sources']) for row in rows),
        benchmark_coverage_increased=False,source_metadata_qualified=True,numeric_arrays_decoded=0,new_model_forwards=0,
        original_train_receipts_preserved=True,CPU_restore_receipts_fabricated=0,native_mutations=0,model_or_dataset_copies=0,
        all20_GPU_or_numerical_contracts_qualified=False,full_portfolio_complete=False))
    print('All20components bound in place;no model,NumPy,Torch,dataset copy or scientific benchmark claim.',flush=True)


if __name__ == '__main__':
    run()
