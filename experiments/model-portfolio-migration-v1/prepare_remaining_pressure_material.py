"""Bind the23pending pressure fits to their original training outputs in place.

These are training checkpoints, never fabricated restore receipts. No model
module, NumPy, Torch or simulation dataset is opened. Matching named weights
are recorded only as candidates for a later source/policy equivalence check.
"""
from collections import defaultdict
from pathlib import Path
import time

from import_common_generator_measurements import ROOT,RUN,DOC,PROJECT,read,save,binding,require

OUTPUT = RUN/'remaining-pressure-material-01.json'
REPORT = DOC/'reports/remaining-pressure-material-readiness-01.json'


def run():
    require(not OUTPUT.exists() and not REPORT.exists(),'Preserve original remaining-pressure material')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_bound = binding(index_path)
    index = read(index_path)
    coverage_path = DOC/'reports/common-fit-benchmark-coverage-01.json'
    coverage = read(coverage_path)
    pending = {r['fit_item_id'] for r in coverage['rows'] if r['default_benchmark'] is None and r['task'] == 'pressure_proposal'}
    fits = {r['item_id']:r for r in index['fits']}
    require(len(pending) == 23 and pending <= fits.keys(),'Exact23pending pressure-proposal fits required')
    reviews,specs = {},{}
    for spec in index['sources']['method_review_batches']:
        path = ROOT/spec['path']
        require(binding(path) == spec,'Published captured-review authority changed')
        for row in read(path).get('rows',[]):
            if row.get('fit_item_id') in pending:
                reviews[row['fit_item_id']],specs[row['fit_item_id']] = row,spec
    require(reviews.keys() == pending,'Every pending fit needs a published original-source review')
    timing_path = DOC/'reports/pressure-portfolio-GPU-timing-01.json'
    matched_weights = defaultdict(list)
    for name,model in read(timing_path)['models'].items():
        matched_weights[model['weights_sha256']].append(name)
    rows = []
    for fit_id in sorted(pending):
        row,fit = reviews[fit_id],fits[fit_id]
        receipt_path = ROOT/row['original_receipt']['path']
        witness_path = ROOT/row['source_witness']['path']
        require(binding(receipt_path) == row['original_receipt']
                and binding(witness_path) == row['source_witness'],'Reviewed original receipt/witness bytes differ')
        receipt,witness = read(receipt_path),read(witness_path)
        project,artifacts = witness_path.parent,receipt_path.parent
        config_path = project/'config.json'
        config_bound = binding(config_path)
        closure = receipt['closure']
        require(receipt['status'] == 'complete' and receipt['mode'] == 'train'
                and row['original_receipt']['sha256'] == fit['receipt_sha256']
                and closure['source_witness_sha256'] == row['source_witness']['sha256'],
                'Completed indexed training receipt and its exact source witness required')
        configuration_hashes = [closure[k] for k in ('experiment_sha256','config_sha256') if k in closure]
        require(configuration_hashes and all(h == config_bound['sha256'] for h in configuration_hashes),
                'Original training configuration closure differs')
        sources = []
        for name,expected in sorted(witness['sources'].items()):
            source = project/name
            observed = binding(source)
            require(source.resolve().is_relative_to(project.resolve()) and observed['sha256'] == expected,
                    'Original captured source path or bytes differ')
            sources.append(observed)
        outputs = {}
        for name in ('weights.npz','normalization.json','history.json'):
            path = artifacts/name
            observed = dict(binding(path),bytes=path.stat().st_size)
            expected = receipt['outputs'][name]
            require(observed['sha256'] == expected['sha256'] and observed['bytes'] == expected['bytes'],
                    'Original selected training output differs')
            outputs[name] = observed
        require(outputs['normalization.json']['sha256'] == closure['normalization_sha256']
                and any(c['sha256'] == outputs['weights.npz']['sha256'] and c['stage'] == 'selected' for c in fit['checkpoints']),
                'Selected fit weights or original normalization binding differs')
        rows.append(dict(fit_item_id=fit_id,name=fit['name'],checkpoint_origin='original_train_output',
            original_receipt=row['original_receipt'],project=str(project.relative_to(ROOT)),artifacts=str(artifacts.relative_to(ROOT)),
            original_configuration=config_bound,source_witness=row['source_witness'],sources=sources,outputs=outputs,
            original_optimizer_updates=receipt['optimizer_updates'],captured_architecture=receipt['architecture'],
            matching_measured_named_weight_candidates=matched_weights[outputs['weights.npz']['sha256']],
            policy_equivalence_qualified=False,GPU_adapter_qualified=False,source_review_authority=specs[fit_id]))
    require(binding(index_path) == index_bound,'Working fit index changed during material binding')
    save(OUTPUT,dict(status='all23_pending_pressure_original_train_outputs_and_sources_bound',created_unix=time.time(),
        root=str(ROOT),models=rows,source_fit_index=index_bound,coverage=binding(coverage_path),producer=binding(Path(__file__)),
        native_mutations=0,new_model_forwards=0,payload_copies=0,CPU_restore_receipts_fabricated=0,
        qualification='Source/checkpoint custody only. Existing named-pressure loader requires a CPU restore receipt; these train outputs require an explicit source-preserving GPU-only adapter. Matching checkpoint bytes alone do not qualify policy or timing equivalence.',
        next_execution_requires_serial_native_publication_finished=True,full_portfolio_complete=False))
    save(REPORT,dict(status='23_original_train_checkpoint_materials_ready_for_GPU_adapter_qualification',created_unix=time.time(),
        fits=23,source_files_verified=sum(len(r['sources']) for r in rows),
        fits_with_measured_named_weight_candidate=sum(bool(r['matching_measured_named_weight_candidates']) for r in rows),
        material=binding(OUTPUT),producer=binding(Path(__file__)),
        benchmark_coverage_increased=False,new_model_forwards=0,simulation_datasets_opened=0,native_mutations=0,payload_copies=0,
        full_portfolio_complete=False))
    print('Bound23original pressure training checkpoints and captured sources; no model execution, dataset read or restore fabrication.',flush=True)


if __name__ == '__main__':
    run()
