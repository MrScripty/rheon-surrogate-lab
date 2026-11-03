"""Review C41/C42 supplemental operator feedback without importing models."""
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from captured_pressure_review import load_fit
from prepare_method_review_revisions import revised_card, verified_review_receipt
from review_pressure_budget_coverage_fits import audit_bank
from review_water_learning_sequence import binding, canonical, population_identity, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC/'reports/captured-targeted-supplement-method-reviews-01.json'


def audit_supplement(facts):
    spec = facts['config']['supplemental_feedback']
    primary = facts['history']['analytical_feedback']
    saved = facts['history']['supplemental_feedback']
    require(spec['seed'] == saved['seed'] == 2026101607 and spec['weight'] == saved['weight'] == .1,
            'Supplement stream seed/weight differs')
    reuse = spec['source'] == 'primary'
    grids = facts['config']['analytical_feedback']['grids'] if reuse else spec['grids']
    require(saved['reuses_primary'] == reuse and len(saved['grids']) == len(grids)
            and saved['primary_bank_description_sha256'] == primary['bank_description_sha256'],
            'Supplement bank provenance differs')
    rng = np.random.Generator(np.random.PCG64(spec['seed']))
    selected = np.concatenate([rng.permutation(len(grids)) for _ in range(math.ceil(5000/len(grids)))])[:5000]
    order = (selected*128 + rng.integers(128, size=5000, dtype=np.int64)).astype('<i8')
    require(hashlib.sha256(order.tobytes()).hexdigest() == saved['sampling_order_sha256']
            and np.bincount(selected, minlength=len(grids)).tolist() == saved['grid_exposures']
            and saved['additional_examples'] == 5000, 'Supplement exposure/order differs')
    if reuse:
        require(canonical(saved['grids']) == canonical(primary['grids'])
                and saved['bank_description_sha256'] == primary['bank_description_sha256'],
                'Control must reuse the original primary tensors/descriptors')
    else:
        require(len(grids) == 6 and all(g['cells_per_metre'] == 16 for g in grids)
                and {tuple(g['counts_xyz']) for g in grids} == {
                    (16,23,25),(23,25,16),(25,16,23),(10,7,25),(7,25,10),(25,10,7)},
                'Targeted grid population differs')
        for grid, evidence in zip(grids, saved['grids'], strict=True):
            shape = tuple(reversed(grid['counts_xyz']))
            high = tuple(n-1 for n in shape)
            fixed = [(0,0,1),(0,1,0),(1,0,0),(0,1,2),(3,2,1),(4,4,4),(0,0,high[2]),high]
            fixed = [tuple(min(k,n-1) for k,n in zip(f,shape)) for f in fixed]
            rng = np.random.Generator(np.random.PCG64(spec['seed']))
            descriptions = []
            for index in range(32):
                if index < 8:
                    components = [(fixed[index],1.)]
                else:
                    components = []
                    for _ in range(1 if index % 2 else 3):
                        frequency = tuple(int(rng.integers(n)) for n in shape)
                        if not any(frequency):
                            frequency = (0,0,1)
                        components.append((frequency,float(rng.uniform(.5,1.5))))
                descriptions.extend(dict(components_zyx=components,
                    amplitude_pa=primary['amplitude_rms_pa']*factor) for factor in (-2.,-.5,.5,2.))
            bank = evidence['bank']
            within = rng.integers(128, size=1, dtype=np.int64).astype('<i8')
            require(evidence['grid'] == grid and tuple(bank['shape_zyx']) == shape
                    and canonical(bank['samples']) == canonical(descriptions)
                    and hashlib.sha256(canonical(descriptions)).hexdigest() == bank['bank_description_sha256']
                    and hashlib.sha256(within.tobytes()).hexdigest() == bank['sampling_order_sha256'],
                    'Targeted pattern/amplitude/within-bank draw differs')
    projected = 0 if reuse else 128*(32*sum(math.prod(g['counts_xyz']) for g in grids)+24*len(grids))
    require(saved['resident_tensor_bytes'] == projected
            and saved['combined_resident_tensor_bytes'] == primary['resident_tensor_bytes']+projected <= 805306368
            and saved['amplitude_rms_pa'] == primary['amplitude_rms_pa']
            and hashlib.sha256(canonical(saved['grids'])).hexdigest() == saved['bank_description_sha256'],
            'Supplement resident budget/amplitude/bank identity differs')
    return dict(reuses_primary=reuse,bank_grids=len(grids),bank_fields=128*len(grids),
        supplemental_proposals=5000,grid_exposures=saved['grid_exposures'],
        additional_resident_bytes=projected,combined_resident_bytes=saved['combined_resident_tensor_bytes'],
        supplemental_descriptors_and_order_independently_rederived=True)


def run():
    require(not OUTPUT.exists(), 'Preserve original source-review report')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    index = read(index_path)
    work = read(ROOT/'runs/model-portfolio-migration-v1/discovery/model-worklist-05.json')
    material = {r['id']:r for r in read(ROOT/'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')['models']}
    score = read(DOC/'reports/pressure-common-score-01.json')['models']
    timing_path = DOC/'reports/pressure-portfolio-GPU-timing-01.json'
    timing = read(timing_path)
    baseline_path = DOC/'reports/captured-pressure-budget-coverage-method-reviews-01.json'
    baseline = next(r for r in read(baseline_path)['rows'] if r['named_checkpoint'] == 'C40')
    for spec in (baseline['history'], baseline['training_population'], baseline['original_receipt']):
        require(binding(ROOT/spec['path']) == spec, 'Reviewed C40 baseline changed')
    initial = read(ROOT/baseline['history']['path'])['checkpoints']['checkpoint-000.npz']['inventory']
    population = population_identity(read(ROOT/baseline['training_population']['path'])['parents'])
    rows, notes, facts = [], [], []
    for name in ('C41','C42'):
        f = load_fit(name, work=work, material=material, index=index, score=score)
        meta, config, history = f['metadata'], f['config'], f['history']
        require((meta['actual_train_trajectories'],meta['actual_train_frames'],meta['actual_validation_trajectories'],meta['actual_validation_frames']) == (16,32,64,128)
                and meta['actual_parameter_count'] == 370016 and meta['initialized_tensor_count'] == 19
                and f['inventory'] == initial and population_identity(f['split']['parents']) == population,
                'Matched physical population/initial linear tensors differ')
        require(config['conditioning'] == 'physical-residual' and config['normalization'] == 'none'
                and config['correction'] == dict(activation='linear',steps=2)
                and config['output_scale'] == 'previous-pressure-rms'
                and config['objective'] == 'canonical-intermediate-contraction'
                and config['training']['updates'] == 5000 and config['training']['batch_size'] == 1
                and config['training']['learning_rate'] == .001
                and config['training']['schedule'] == [dict(learning_rate=.0001,update=3000),dict(learning_rate=.00001,update=4500)]
                and config['training']['seed'] == 20261015 and config['training']['sampling_seed'] == 2026102501,
                'Matched optimizer/deployment policy differs')
        primary = audit_bank(f)
        extra = audit_supplement(f)
        source = (f['project']/'train.py').read_text()
        model_source = (f['project']/'model.py').read_text()
        analytical_source = (f['project']/'analytical.py').read_text()
        require('extra=model.training_loss(network,supplemental_sample,normalization,' in source
                and '(supplemental_weight*extra).backward()' in source
                and 'extra=model.training_loss(network,analytical_sample,normalization,' in source
                and 'return final+physical_residual.INTERMEDIATE_CONTRACTION_WEIGHT*intermediate' in model_source
                and "samples,_,parent=primary" in analytical_source
                and "'reuses_primary':specification['source']=='primary'" in analytical_source,
                'Executed supplemental/intermediate training path differs')
        reuse = name == 'C41'
        require(extra['reuses_primary'] == reuse, 'Named control/targeted arm differs')
        common = score[name]['worst']
        perf = timing['models'][name]['summary']
        row = dict(meta, **primary, supplemental_feedback=extra,
            configured_pressure_corrections=2,matched_C40_initial_weights_and_physical_population=True,
            actual_train_cells_per_metre=[8,16],actual_validation_cells_per_metre=[8,16],retained_timing=binding(timing_path),
            method='A pressure-only 370,016-parameter multiscale 3D convolutional correction network with 19 weight tensors. Executed layers remove GroupNorm, SiLU and biases despite inherited cubic-groupnorm naming. The physical equation residual q=(rho/h)r is divided by train-only q RMS before F32 convolution. Two shared-weight corrections recompute the F64 residual, scale by train previous-pressure RMS, apply cell-zero gauge and add in F64. dt is validated and reflected in externally assembled fields, not a separate CNN input. This sealed uniform cubic-voxel Neumann policy advances neither velocity nor tracer and accepts no independent geometry mask.',
            purpose=('Control the extra training work with a second independently sampled proposal from the existing eight manufactured grids.' if reuse else 'Spend the same extra proposal/update budget on six targeted rectangular manufactured grids instead of resampling the existing bank.'),
            experiment='Both arms start from the same seeded linear tensors and physical sample order as C40: 5,000 Adam updates, batch1, initial rate0.001 with drops at3000/4500, model seed20261015 and physical sampling seed2026102501. Physical training uses16trajectories/32frames at8/16cells per metre; validation uses64trajectories/128frames, axes at most16cells, dt1/60s. Each update combines one physical example, one primary manufactured proposal at weight0.1 and a supplemental proposal at weight0.1. All three paths use final loss plus0.1nonfinal relative equation-residual squared. The primary bank has1,024fields across eight grids,625exposures/grid; the independent supplemental stream uses seed2026101607. '+('C41 reuses the same1,024primary tensors with625extra exposures/grid and zero additional resident tensors.' if reuse else 'C42 adds768fields across16x23x25 and10x7x25 with their cyclic permutations, all at16cells/metre; supplemental exposures are833/834per grid. Combined resident tensor projection is664,041,472bytes.')+' These are pressure-operator fields, not new CFD trajectories or additional optimizer updates.',
            outcome=f'Last logged training objective {history["history"][-1]["loss"]:.17g}; final retained canonical validation objective {history["validation"]["final"]:.17g}. Common128-request worst mean-free pressure relative L2 {common["pressure_mean_free_relative_l2"]:.17g}; equation relative L2 {common["common_equation_relative_l2"]:.17g}. Complete GPU pressure-call median of case medians {perf["median_case_median_ms"]:.17g}ms and worst case median {perf["worst_case_median_ms"]:.17g}ms. Timing includes native CPU input validation/packing, transfers, pressure work, independent CPU output and synchronization, excluding setup/scoring; it is not a physical-step or drift measurement.',
            lesson=('Relative to C40, the extra primary stream lowers worst pressure error but raises worst equation error. More feedback work has a metric-dependent effect; the changed weighted training objective cannot be compared as the same scalar loss.' if reuse else 'C42 improves both common worst pressure and equation errors over C41 at equal optimizer and proposal counts. Compared with C40, pressure improves but equation error remains worse. Geometry, stream size and per-grid exposure change together; this is diagnostic evidence for targeted feedback, not held-out generalization or extra physical-scene coverage.'),
            reproduction='Use original typed recipe/config/runtime, exact captured source/input closures, train-only normalization and initial tensors. Recreate primary and supplemental manufactured descriptors, amplitudes, private seeds and balanced order; preserve three weighted loss paths and combined memory budget. Historical trainers include CPU parity/deployment forwards; qualify a GPU-only reproducer before rerunning. Clean retraining, exact final-checkpoint reproduction and new-Rheon generation remain unqualified.')
        receipt = verified_review_receipt(row)
        note = PROJECT/row['note_path']
        raw = note.read_bytes()
        revised, suffix = revised_card(raw,row)
        require(raw.count(receipt) == revised.count(receipt) == 1 and revised.endswith(suffix),
                'Original receipt/configuration suffix changed')
        rows.append(row)
        notes.append(binding(note))
        facts.append(f)
    require(np.array_equal(facts[0]['sample_order'],facts[1]['sample_order'])
            and facts[0]['inventory'] == facts[1]['inventory']
            and canonical(facts[0]['history']['analytical_feedback']) == canonical(facts[1]['history']['analytical_feedback']),
            'Matched arm primary bank/order/initial state differs')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/n['path']) == n for n in notes),
            'Native notes/index changed during review')
    require('torch' not in sys.modules, 'Review imported a neural runtime')
    value = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,
        source_reviewed_fits=2,new_model_forwards=0,native_mutations=0,new_reference_generation=False,
        clean_reproduction_qualified=False,original_note_bindings=notes,baseline=binding(baseline_path),
        overview_addition='\n\n## Targeted supplemental operator feedback\n\nC41/C42 match optimizer updates, physical population, initial tensors and primary manufactured stream. They spend an additional weighted training proposal on existing versus targeted rectangular operator fields. C42 improves both common worst metrics against C41, while equation error remains worse than C40. These are pressure-request diagnostics, not expanded physical trajectories, full-state speed or drift qualification.\n',
        producer=binding(Path(__file__)),helpers=[binding(ROOT/'experiments/model-portfolio-migration-v1'/name) for name in
            ['captured_pressure_review.py','review_pressure_budget_coverage_fits.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(value)+b'\n')
    print('Two supplemental-feedback fits source-reviewed; no native mutation or neural execution.')


if __name__ == '__main__':
    run()
