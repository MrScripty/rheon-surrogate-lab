"""Select one matched native population for four original modal pressure fits.

This is metadata admission, not an adapter, numerical benchmark or new dataset.
Original field archives will be freshly verified and decoded by GPU workers.
"""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from component_material import current_publication, observed_file

FITS = ('fit-83a63b7141c12414', 'fit-561e02c747b96415',
        'fit-b48ebd35d5c9448d', 'fit-aaf0a360c4c9622b')
UNIT_FIELDS = ('schema', 'pressure_dtype', 'rhs_dtype', 'basis', 'q_units', 'pressure_gauge')


def native_rows(manifest):
    rows = []
    for geometry in manifest['geometries']:
        for sample in geometry['samples']:
            if 'source_sha256' not in sample:
                continue
            rows.append(dict(key=geometry['key'], counts_zyx=geometry['counts_zyx'],
                **{key:sample[key] for key in ('identifier', 'state', 'cells_per_metre', 'source', 'source_sha256')}))
    return rows


def run():
    output = RUN/'modal-component-comparison-contract-01.json'
    require(not output.exists(), 'Preserve the original modal admission')
    publication, _ = current_publication(ROOT)
    material_path = RUN/'remaining-component-material-01.json'
    material = read(material_path)
    original = {row['fit_item_id']:row for row in material['models']}
    canonical, units, model_sha, basis_sha = None, None, None, None
    fits = []
    for fid in FITS:
        row = original[fid]
        require(row['task'] == 'pressure_component' and row['parameter_dtypes'] == ['<f8']
                and row['serialized_tensor_elements'] == 38673, 'Original full-F64 modal table required')
        for spec in [row['original_receipt'], row['original_configuration'], *row['sources'],
                     *row['outputs'].values(), row['inputs']['manifest'], row['inputs']['normalization']]:
            observed_file(ROOT, spec['path'], spec)
        manifest = read(ROOT/row['inputs']['manifest']['path'])
        norm = read(ROOT/row['inputs']['normalization']['path'])
        selected_units = {key:norm[key] for key in UNIT_FIELDS}
        selected_rows = native_rows(manifest)
        model = next(spec for spec in row['sources'] if spec['original_name'] == 'learned_modal_pressure.py')
        basis = next(spec for spec in row['sources'] if spec['original_name'] == 'modal_basis.py')
        if canonical is None:
            canonical, units, model_sha, basis_sha = selected_rows, selected_units, model['sha256'], basis['sha256']
        require(selected_rows == canonical and selected_units == units and model['sha256'] == model_sha
                and basis['sha256'] == basis_sha, 'Matched original native rows, units and exact model/basis implementation required')
        fits.append(dict(fit_item_id=fid, weights=row['outputs']['weights.npz'], model=model, basis=basis,
            manifest=row['inputs']['manifest'], normalization=row['inputs']['normalization'],
            original_fit_native_rows=len(selected_rows), original_fit_total_rows=manifest['samples'],
            original_extra_query_rows=manifest['samples']-len(selected_rows)))
    require(len(canonical) == 144 and len({row['identifier'] for row in canonical}) == 16
            and {row['cells_per_metre'] for row in canonical} == {8,16}
            and len({row['key'] for row in canonical}) == 4, 'Exact original144native frames/16scenes/fourgrids/two densities required')
    for row in canonical:
        path = ROOT/row['source']
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT),
                'Original native field path must remain available in place')
    save(output, dict(status='four_modal_component_matched_native_field_contract_admitted_adapter_pending',
        created_unix=time.time(), material=binding(material_path), publication_acceptance=binding(publication),
        fits=fits, rows=canonical, units=units, source_model_sha256=model_sha, source_basis_sha256=basis_sha,
        population='Same144native fitting frames from16original scenes,allfourfits consumed these source states. Mixed refinement also consumed128extra queries,excluded from this matched144population;not independent holdout.',
        input_boundary='Owned original native CPU F64 RHS plus fluid mass density1000kg/m3 andcell spacing1/cells_per_metre. No future pressure orvelocity/tracer labels forwarded.',
        execution_boundary='Validate/pack/H2D,convertq=RHS*rho/h,fixed full captured CUDA DCT-II,unchanged published F64 LearnedModalPressure forward,exact captured inverse transform,cell-zero gauge,ownedCPUpressure D2H/sync. Setup/cache of prescribed basis separate. Recompute learned exponentiated gains in every timed request;no unmeasured learned response cache.',
        timing_units='GPU ms/native RHS pressure request;not a simulation step or the CNN base/residual five-input request boundary',
        accuracy='Physical pressure mean-free relativeL2,gauge pressure absoluteRMS/max,and existing accepted native equation residual against current native RHS. Independent retained-output scoring;group separately bycells_per_metre andcounts_xyz,andjoint shape/density.',
        planned_fresh_GPU_processes=8, planned_phases_per_fit=2, samples_per_case=10, warmups_per_case=5,
        paired_calibration_pairs_per_case=4, primary_observer_removed=True, CPU_neural_forward_allowed=False,
        read_field_arrays=False, source_field_verification='Existence only at admission;fresh archive SHA/field shapes/dtypes/finite checks required in each GPUworker before prediction.',
        no_common_150fit_latency_ranking_claim=True, pressure_solver_full_state_generator=False,
        benchmark_coverage_increased=False, new_model_forwards=0, new_parameter_fits=0,
        new_reference_generation=False, weight_or_dataset_copies=0, producer=binding(Path(__file__)),
        full_portfolio_complete=False))
    print('Four original modal fits share144native source frames andexact model/basis code;fieldGPUadapter remains pending.', flush=True)


if __name__ == '__main__':
    run()
