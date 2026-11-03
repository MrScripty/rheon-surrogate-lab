"""Bind one common retained native case set for full-generator GPU benchmarking.

Only reference fields/metadata are read. No simulation, model, project mutation
or copied array archive. This is case qualification, not a measured benchmark.
"""
import math
from pathlib import Path
import time

import numpy as np

from full_state_gpu_timing import field_sizes
from review_direction_pressure_fits import operator
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
OUTPUT = DOC/'reports/joint-generator-case-preflight-01.json'
FIELDS = ('pressure','vx','vy','vz','tracer')


def run():
    require(not OUTPUT.exists(), 'Preserve original case qualification')
    reviewed_path = DOC/'reports/captured-joint-temporal-generator-method-reviews-01.json'
    reviewed = read(reviewed_path)
    first = reviewed['rows'][0]
    manifest_path = ROOT/first['training_manifest']['path']
    require(binding(manifest_path) == first['training_manifest'], 'Reviewed training scene identities changed')
    manifest = read(manifest_path)
    bank_path = ROOT/'runs/full-ai-next-priorities-v1/temporal-data-01/bank.json'
    bank = read(bank_path)
    capture_source = ROOT/'experiments/full-ai-next-priorities-v1/capture_temporal.py'
    client_source = ROOT/'experiments/full-ai-next-priorities-v1/native_client.py'
    numerical_source = ROOT/'experiments/pressure-methods-v1/numerics.py'
    require(bank['status'] == 'completed' and bank['reference_method'] == 'full-spectral-on-actual-native-requests'
            and binding(capture_source)['sha256'] == bank['capture_source_sha256']
            and binding(client_source)['sha256'] == bank['native_client_sha256']
            and binding(numerical_source)['sha256'] == bank['numerical_reference_sha256'], 'Original reference capture implementation differs')
    build_path = ROOT/'runs/full-ai-next-priorities-v1/temporal-build-02/build-receipt.json'
    build = read(build_path)
    binary_path = build_path.parent/'rheon-temporal-probe'
    require(build['status'] == 'qualified_build' and binding(binary_path)['sha256'] == build['binary_sha256'],
            'Original reference native probe binary differs')
    models = []
    for row in reviewed['rows']:
        require(all(binding(ROOT/spec['path']) == spec for spec in row['sources']), 'Reviewed model source changed')
        receipt_path = ROOT/row['original_receipt']['path']
        require(binding(receipt_path) == row['original_receipt'], 'Original model receipt changed')
        receipt = read(receipt_path)
        weights = receipt_path.parent/'weights.npz'
        require(binding(weights)['sha256'] == receipt['outputs']['weights.npz']['sha256'], 'Named generator weights changed')
        models.append(dict(fit_item_id=row['fit_item_id'],identifier=row['generator_identifier'],K=row['horizons'],
            project_path=str((ROOT/row['original_configuration']['path']).parent.relative_to(ROOT)),
            configuration=row['original_configuration'],receipt=row['original_receipt'],weights=binding(weights),
            normalization=next(x for x in row['training_inputs'] if x['slot']=='normalization'),sources=row['sources']))
    cases, observed = [], {}
    maximum_reference_equation_error = 0.
    for parent in manifest['parents']:
        receipt_path = ROOT/parent['receipt_source']
        require(binding(receipt_path)['sha256'] == parent['receipt_sha256'], 'Original native capture receipt changed')
        receipt = read(receipt_path)
        require(receipt['state_contract_sha256'] == bank['state_contract_sha256'] and receipt['split'] == 'train',
                'Case reference contract/exposure differs')
        scene = receipt['scene']
        sizes = field_sizes(scene['counts'])
        for start in (0,24,64,120):
            frames = []
            for step in range(start,start+5):
                record = receipt['states'][step]
                path = receipt_path.parent/record['path']
                require(record['step'] == step and math.isclose(record['time_s'],step/60,rel_tol=1e-14,abs_tol=1e-14)
                        and binding(path)['sha256'] == record['sha256'] and path.stat().st_size == record['bytes'],
                        'Original current/target frame identity or timestamp differs')
                with np.load(path,allow_pickle=False) as arrays:
                    require(set(arrays.files) == set(FIELDS)|{'rhs'}, 'Exact native state and label-only RHS required')
                    for field in (*FIELDS,'rhs'):
                        value = arrays[field]
                        expected_dtype = '<f8' if field in ('pressure','rhs') else '<f4'
                        require(value.dtype.str == expected_dtype and value.size == sizes['pressure' if field=='rhs' else field]
                                and np.isfinite(value).all(), 'Native physical field dtype/extent/finite contract differs')
                    if step > start:
                        rhs = arrays['rhs'].reshape(tuple(reversed(scene['counts'])))
                        p = arrays['pressure'].reshape(rhs.shape)
                        denominator = float(np.square(rhs).sum())
                        residual = rhs-operator(p,1/scene['cells_per_metre'],1000.)
                        error = math.sqrt(float(np.square(residual).sum())/denominator) if denominator else float(np.linalg.norm(residual))
                        require(error < 1e-6, 'Recorded target pressure/RHS does not satisfy the common sealed FV operator')
                        maximum_reference_equation_error = max(maximum_reference_equation_error,error)
                item = dict(path=str(path.relative_to(ROOT)),sha256=record['sha256'],bytes=record['bytes'],step=step,time_s=record['time_s'])
                observed[item['path']] = item
                frames.append(item)
            cases.append(dict(identifier=parent['identifier'],split='train',scene=scene,start=start,output_dt_s=1/60,
                current=frames[0],reference_leads=frames[1:],reference_receipt=binding(receipt_path)))
    require(len(cases) == 64 and len(models) == 4 and len(observed) == 320, 'Full shared four-start/16-scene population required')
    value = dict(status='64_retained_full_state_cases_and_original_reference_operator_verified',created_unix=time.time(),
        models=models,cases=cases,unique_native_state_files=320,native_state_bytes_read=sum(r['bytes'] for r in observed.values()),
        reference_equation_relative_l2_max=maximum_reference_equation_error,reference_method=bank['reference_method'],
        reference_native_probe_binary_sha256=build['binary_sha256'],reference_native_probe_build=binding(build_path),
        reference_pressure_solver='Original external full spectral pressure on the captured native integrator requests; not a fresh native-PCG or new-Rheon solver benchmark.',
        current_input_fields=list(FIELDS),RHS_is_reference_label_only=True,exposure='All16scenes are original fitting parents; consumed diagnostics, no independent holdout claim.',
        planned_timing_boundary='Native five-field CPU input through validation/views, source packing/known forcing, H2D, joint forward, physical reconstruction and allKowned CPU output fields/D2H/synchronization. Report block and block/K; Kphysical outputs at1/60s each.',
        fresh_model_inference_executed=False,GPU_timing_qualified=False,new_model_forwards=0,new_reference_generation=False,
        payload_copies=0,native_mutations=0,full_portfolio_complete=False,producer=binding(Path(__file__)),
        evidence=[binding(p) for p in [reviewed_path,manifest_path,bank_path,capture_source,client_source,numerical_source]])
    OUTPUT.write_bytes(canonical(value)+b'\n')
    print('64common native full-state cases verified,320original files; reference equation max',maximum_reference_equation_error)


if __name__ == '__main__':
    run()
