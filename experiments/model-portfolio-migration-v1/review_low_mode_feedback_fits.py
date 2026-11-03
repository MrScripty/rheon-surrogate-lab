"""Review C49--C51 native low-spatial-mode training without neural execution."""
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

from captured_native_pressure_review import load_fit
from prepare_method_review_revisions import revised_card, verified_review_receipt
from review_direction_pressure_fits import audit_direction
from review_pressure_budget_coverage_fits import audit_bank
from review_targeted_supplement_fits import audit_supplement
from review_water_learning_sequence import binding, canonical, population_identity, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC/'reports/captured-low-mode-feedback-method-reviews-01.json'


def check_spatial_basis():
    """Check the mathematical interpretation, not unexecuted Torch parity."""
    maximum = 0.
    for length in (5,7,8,16,23,25,40,45):
        basis = np.cos(np.pi/length*np.arange(4)[:,None]*(np.arange(length)[None,:]+.5))*np.sqrt(2./length)
        basis[0] /= np.sqrt(2.)
        error = float(np.max(np.abs(basis@basis.T-np.eye(4))))
        require(error < 1e-13, 'Orthonormal low spatial-mode interpretation differs')
        maximum = max(maximum,error)
    return dict(axes_modes=[0,1,2,3],nonconstant_modes_on_axes_at_least4=63,
        mathematical_orthogonality_max_error=maximum,Torch_numerical_parity_executed=False,
        scope='Independent DCT-II mathematical interpretation over selected lengths, not model execution or complete implementation conformance.')


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
    baseline_path = DOC/'reports/captured-direction-pressure-method-reviews-01.json'
    baseline = next(r for r in read(baseline_path)['rows'] if r['named_checkpoint'] == 'C45')
    initial = read(ROOT/baseline['history']['path'])['checkpoints']['checkpoint-000.npz']['inventory']
    population = population_identity([p for p in read(ROOT/baseline['training_population']['path'])['parents'] if p['entry']['split'] == 'train'])
    basis = check_spatial_basis()
    rows, notes, facts, source_hashes = [], [], [], []
    for name, weight in [('C49',.1),('C50',.01),('C51',.03)]:
        f = load_fit(name,work=work,material=material,index=index,score=score)
        meta, config, history = f['metadata'], f['config'], f['history']
        require((meta['actual_train_trajectories'],meta['actual_train_frames'],meta['actual_validation_trajectories'],meta['actual_validation_frames']) == (16,32,0,0)
                and f['inventory'] == initial and population_identity(f['split']['parents']) == population
                and meta['actual_parameter_count'] == 370016 and meta['initialized_tensor_count'] == 19,
                'Native physical training-only population/initial state differs')
        require(config['band_feedback'] == dict(max_mode=3,signal_floor_fraction=.1,stages='first-and-final',weight=weight)
                and config['first_pressure_weight'] == 0 and config['correction'] == dict(activation='linear',steps=2)
                and config['objective'] == 'canonical-intermediate-contraction'
                and config['training']['updates'] == 5000 and config['training']['batch_size'] == 1
                and config['training']['seed'] == 20261015 and config['training']['sampling_seed'] == 2026102501
                and config['training']['learning_rate'] == .001
                and config['training']['schedule'] == [dict(learning_rate=.0001,update=3000),dict(learning_rate=.00001,update=4500)],
                'Low-mode training weight/budget policy differs')
        primary, supplemental = audit_bank(f), audit_supplement(f)
        direction = audit_direction(f,'late')
        model = (f['project']/'model.py').read_text()
        band = (f['project']/'band_feedback.py').read_text()
        diagnostic = (f['project']/'training_diagnostic.py').read_text()
        require("feedback=(band_feedback.loss(pressures[0],sample,BAND_FEEDBACK)" in model
                and '+band_feedback.loss(prediction,sample,BAND_FEEDBACK))/2' in model
                and "result=result+BAND_FEEDBACK['weight']*feedback" in model
                and 'return result.flatten(1)[:, 1:]' in band
                and 'target = target - target.mean(dim=(2, 3, 4), keepdim=True)' in band
                and "floor = specification['signal_floor_fraction'] ** 2 * target.flatten(1).square().sum(1)" in band
                and 'torch.maximum(signal, floor).clamp_min(1e-30)' in band
                and 'all(p["entry"]["split"] == "train" for p in parents)' in diagnostic,
                'Executed low-spatial-mode objective/diagnostic owner differs')
        source_hashes.append({n:binding(f['project']/n)['sha256'] for n in ['model.py','train.py','band_feedback.py','physical_residual.py','training_diagnostic.py']})
        common, perf = score[name]['worst'], timing['models'][name]['summary']
        lesson = {
            'C49':'The0.1low-mode arm is a training-objective change with unchanged two-correction inference. Its history field named validation is explicitly computed over fitting parents. Do not treat that scalar or the wrapper consumed-validation probe as independent held-out generalization.',
            'C50':'Compared with C49, weight0.01 lowers the common worst pressure error but raises worst equation error. It has the lowest pressure tail among this three-weight sweep, while C51 has a lower equation tail. The training history validation label is a train-only diagnostic.',
            'C51':'Weight0.03 improves both common worst metrics over C49. Compared with C50 it trades a higher pressure tail for a lower equation tail, so no single weight dominates all metrics. These spatial modes are not temporal frequencies, and the auxiliary loss adds no inference stage.'}[name]
        row = dict(meta,**primary,supplemental_feedback=supplemental,direction_feedback=direction,
            configured_pressure_corrections=2,low_spatial_mode_feedback=dict(config['band_feedback']),
            actual_train_cells_per_metre=[8,16],actual_validation_cells_per_metre=[],retained_timing=binding(timing_path),
            method='A one-channel370,016-parameter linear multiscale3D convolutional pressure-correction core with19weight tensors; captured transforms remove GroupNorm, SiLU and biases. Input q=(rho/h)r uses train-only RMS. Two F32 corrections recompute the F64 sealed cubic-voxel Neumann residual and reconstruct scaled cell-zero-gauged F64 pressure. dt is validated, not an independent neural input channel. A training-only separable orthonormal DCT-II penalty retains axes modes0..3, removes DC and supervises63nonconstant spatial modes at the first and final correction. The mean of those two losses is weighted; denominator is the larger of target low-mode energy and0.1squared times full mean-free target energy, floored at1e-30. This emits pressure only, with no added deployed pass or physical advancement.',
            purpose=f'Assess low-spatial-mode pressure feedback weight{weight:g} while retaining the existing two-correction core and broad physical/manufactured/captured-direction training streams.',
            experiment=f'5,000CUDA Adam updates, batch1, learning rate0.001 with drops at3000/4500; model seed20261015 and physical sample seed2026102501. All arms share initial tensor inventories,16physical fitting trajectories/32frames at8/16cells per metre, dt1/60s, axes at most16cells. The native fitting packet contains no validation trajectories. Each update also draws primary1,024fields/eightgrids, supplemental768fields/sixrectangles and one late captured direction among192cases in21balanced groups; stream weights are0.1. These add15,000auxiliary training proposals without adding CFD parents. Existing0.1nonfinal equation contraction and final losses remain, first-pressure weight is0, and low-mode weight is{weight:g}. Native evidence.py preserves the original profile as training-profile.json and emits one already-consumed validation probe after fitting; that probe is not a new optimizer example or held-out claim.',
            outcome=f'Last logged weighted training objective {history["history"][-1]["loss"]:.17g}; final diagnostic retained under the misleading history key validation {history["validation"]["final"]:.17g}, explicitly evaluated on training parents with zero held-out evaluations. Common128-request worst mean-free pressure relative L2 {common["pressure_mean_free_relative_l2"]:.17g}; equation relative L2 {common["common_equation_relative_l2"]:.17g}. Complete GPU pressure-call median of case medians {perf["median_case_median_ms"]:.17g}ms; worst case median {perf["worst_case_median_ms"]:.17g}ms. Timing includes CPU input/output ownership, validation/packing, transfers, all pressure work and synchronization, excluding setup/scoring. These are pressure-request diagnostics, not full physical-step latency or drift.',
            lesson=lesson,
            reproduction='Use original typed native evidence recipe/config/runtime, full witnessed source, exact input/output closures, fixed initial tensors and train-only normalization. Preserve original trainer profile mapping to training-profile.json, train-only diagnostic override, private physical/manufactured/direction sample orders, DCT-II gauge removal and energy floor. Keep the consumed validation probe separate from optimizer data and generalization. The historical evidence wrapper performs CPU neural forwards after CUDA fitting; a GPU-only reproducer must be qualified before rerunning. No clean retraining, optimizer resume or new-Rheon regeneration is qualified.')
        receipt = verified_review_receipt(row)
        note = PROJECT/row['note_path']
        raw = note.read_bytes()
        revised, suffix = revised_card(raw,row)
        require(raw.count(receipt) == revised.count(receipt) == 1 and revised.endswith(suffix), 'Original native receipt/configuration suffix differs')
        rows.append(row)
        notes.append(binding(note))
        facts.append(f)
    require(all(h == source_hashes[0] for h in source_hashes[1:])
            and all(np.array_equal(f['sample_order'],facts[0]['sample_order']) for f in facts[1:])
            and all(canonical(f['history']['direction_feedback']) == canonical(facts[0]['history']['direction_feedback']) for f in facts[1:]),
            'Weight sweep changes executed model/trainer/source or physical/direction order')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/n['path']) == n for n in notes), 'Current notes/index changed')
    require('torch' not in sys.modules, 'Review imported a neural runtime')
    value = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,
        source_reviewed_fits=3,new_model_forwards=0,native_mutations=0,new_reference_generation=False,
        clean_reproduction_qualified=False,original_note_bindings=notes,baseline=binding(baseline_path),spatial_basis_interpretation=basis,
        overview_addition='\n\n## Low-spatial-mode pressure loss and native diagnostic scope\n\nC49–C51 retain the same two-correction core, initial tensors and sample streams while varying a first/final low-mode DCT-II loss weight0.1/0.01/0.03. C50 has the lower common pressure tail; C51 has the lower equation tail. Their history key validation is explicitly a train-only diagnostic, and the native wrapper adds an already-consumed validation probe. Neither supplies independent generalization evidence. Spatial modes and internal corrections are distinct from temporal frequency and generated physical frames.\n',
        producer=binding(Path(__file__)),helpers=[binding(ROOT/'experiments/model-portfolio-migration-v1'/name) for name in
            ['captured_native_pressure_review.py','review_direction_pressure_fits.py','review_pressure_budget_coverage_fits.py','review_targeted_supplement_fits.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(value)+b'\n')
    print('Three low-mode native fits reviewed; train-only validation label and original profile custody explicit; no neural execution.')


if __name__ == '__main__':
    run()
