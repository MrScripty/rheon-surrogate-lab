"""Audit C43/C45--C48 training directions and first-correction supervision.

Reads retained NumPy arrays and independently applies the sealed finite-volume
operator. Imports no captured model and performs no neural forward or fit.
"""
from collections import Counter
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from captured_pressure_review import load_fit
from prepare_method_review_revisions import revised_card, verified_review_receipt
from review_pressure_budget_coverage_fits import audit_bank
from review_targeted_supplement_fits import audit_supplement
from review_water_learning_sequence import binding, canonical, population_identity, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC/'reports/captured-direction-pressure-method-reviews-01.json'
SPECS = [('C43','early',0.),('C45','late',0.),('C46','late',1.),
         ('C47','late',.1),('C48','late',.25)]


def operator(pressure, spacing, rho):
    """Independent sum of signed interior-face fluxes; exterior flux is zero."""
    result = np.zeros_like(pressure)
    weight = spacing*spacing/(rho*spacing)
    for axis in (2,1,0):
        flux = np.diff(pressure, axis=axis)*weight
        lower, upper = [slice(None)]*3, [slice(None)]*3
        lower[axis], upper[axis] = slice(None,-1), slice(1,None)
        result[tuple(lower)] -= flux
        result[tuple(upper)] += flux
    return result


def audit_direction(facts, stage):
    config, history = facts['config'], facts['history']
    inputs = {r['id']:ROOT/r['path'] for r in read(facts['project']/'project.json')['data']}
    manifest_path, payload_path = inputs['direction_manifest'], inputs['direction_dataset']
    manifest = read(manifest_path)
    spec, saved = config['direction_feedback'], history['direction_feedback']
    require(binding(manifest_path)['sha256'] == spec['manifest_sha256'] == saved['manifest_sha256']
            and stage == spec['stage'] == saved['stage']
            and spec['seed'] == saved['seed'] == 2026102701
            and spec['weight'] == saved['weight'] == .1, 'Captured direction policy differs')
    require(manifest['schema'] == 'rheon-pressure-direction-bank-v1'
            and set(manifest['stages']) == {'early','late'}
            and manifest['source_corrections'] == 8 and manifest['optimizer_updates'] == 0
            and binding(inputs['splits'])['sha256'] == manifest['packet_splits_sha256'],
            'Direction bank/physical selection lineage differs')
    train = {p['entry']['trajectory_id']:p for p in facts['split']['parents'] if p['entry']['split'] == 'train'}
    require(set(manifest['train_trajectory_ids']) == set(train), 'Direction bank uses different physical train IDs')
    source = manifest['source']
    require(source == saved['source']
            and binding(ROOT/source['project']/'source-witness.json')['sha256'] == source['source_witness_sha256']
            and binding(ROOT/source['artifacts']/'weights.npz')['sha256'] == source['weights_sha256']
            and binding(ROOT/source['artifacts']/'normalization.json')['sha256'] == source['normalization_sha256']
            and hashlib.sha256(canonical(facts['normalization'])).hexdigest() == source['normalization_sha256'],
            'Frozen source weights/normalization/witness differ')
    declared = manifest['stages'][stage]
    require(binding(payload_path)['sha256'] == declared['sha256'] == saved['payload_sha256']
            and payload_path.stat().st_size == declared['bytes'], 'Selected early/late array payload differs')
    expected_cases = 144 if stage == 'early' else 192
    require(len(manifest['cases']) == saved['cases'] == expected_cases and not manifest['excluded'],
            'Actual selected direction population differs')
    groups, cells, scalar_bytes, worst = {}, [], 0, 0.
    kinds, real_frames = Counter(), set()
    with np.load(payload_path, allow_pickle=False) as arrays:
        expected_keys = set()
        for index, case in enumerate(manifest['cases']):
            identity = case['identity']
            kinds[identity['kind']] += 1
            require(identity['kind'] in ('real','manufactured'), 'Unknown direction case kind')
            shape = tuple(reversed(case['counts_xyz']))
            require(all(2 <= n <= 128 for n in shape) and all(case[k] > 0 for k in ('rho','dt','cell_size_m')),
                    'Invalid direction geometry/scalars')
            if identity['kind'] == 'real':
                parent = train[identity['trajectory_id']]
                require(identity['frame'] in {frame['index'] for frame in parent['frames']} and case['counts_xyz'] == parent['counts_xyz']
                        and case['cell_size_m'] == parent['cell_size_m'] and case['dt'] == parent['entry']['dt_s'],
                        'Captured real direction is outside physical train frames')
                real_frames.add((identity['trajectory_id'],identity['frame']))
            values = {}
            for field in ('rhs','reference'):
                key = f'case_{index:03d}_{field}'
                expected_keys.add(key)
                value, description = arrays[key], declared['arrays'][key]
                require(value.dtype.str == description['dtype'] == '<f8'
                        and list(value.shape) == description['shape'] == [1,1,*shape]
                        and value.nbytes == description['bytes']
                        and hashlib.sha256(value.tobytes()).hexdigest() == description['sha256']
                        and np.isfinite(value).all(), 'Direction array identity/type/finite shape differs')
                values[field] = value[0,0]
            rhs, reference = values['rhs'], values['reference']
            require(reference.flat[0] == 0 and float(np.square(rhs).sum()) > 0,
                    'Direction label gauge or nonzero residual differs')
            residual = rhs-operator(reference,case['cell_size_m'],case['rho'])
            error = math.sqrt(float(np.square(residual).sum())/float(np.square(rhs).sum()))
            worst = max(worst,error)
            require(error < 1e-9 and case['stages'][stage]['operator_relative_error'] < 1e-9,
                    'Retained direction label violates the independent sealed FV equation')
            rms = math.sqrt(float(np.square(reference).mean()))
            require(math.isclose(rms,manifest['amplitude_rms_pa'],rel_tol=5e-14,abs_tol=1e-14),
                    'Direction correction amplitude is not normalized to train-only RMS')
            groups.setdefault(identity['group'],[]).append(index)
            cells.append(math.prod(shape))
            scalar_bytes += 32*math.prod(shape)+24
        require(set(arrays.files) == expected_keys == set(declared['arrays']), 'Direction inventory has missing/extra arrays')
    require(kinds['real'] == len(real_frames) == 32 and kinds['manufactured'] == expected_cases-32,
            'Direction real-frame/manufactured split differs')
    names = sorted(groups)
    rng = np.random.Generator(np.random.PCG64(spec['seed']))
    group_order = np.concatenate([rng.permutation(len(names)) for _ in range(math.ceil(5000/len(names)))])[:5000]
    order = np.asarray([rng.choice(groups[names[int(i)]]) for i in group_order], dtype='<i8')
    require(saved['groups'] == names and saved['group_exposures'] == np.bincount(group_order,minlength=len(names)).tolist()
            and hashlib.sha256(order.tobytes()).hexdigest() == saved['sampling_order_sha256']
            and saved['forward_cells_per_correction'] == sum(cells[int(i)] for i in order)
            and saved['additional_examples'] == 5000, 'Direction group/case sampling work differs')
    resident = history['analytical_feedback']['resident_tensor_bytes']+history['supplemental_feedback']['resident_tensor_bytes']
    require(saved['resident_tensor_bytes'] == scalar_bytes
            and saved['combined_resident_tensor_bytes'] == scalar_bytes+resident <= 805306368,
            'Captured direction counted residency/budget differs')
    return dict(stage=stage,cases=expected_cases,physical_train_frames=32,manufactured_cases=expected_cases-32,
        balanced_groups=len(groups),group_exposures=saved['group_exposures'],optimizer_proposals=5000,
        forward_cells_per_correction=saved['forward_cells_per_correction'],counted_resident_bytes=scalar_bytes,
        independent_label_operator_relative_l2_max=worst,all_array_hashes_verified=True,
        sampling_order_independently_rederived=True,manifest=binding(manifest_path),payload=binding(payload_path),
        source_weights_sha256=source['weights_sha256'],source_normalization_sha256=source['normalization_sha256'])


def run():
    require(not OUTPUT.exists(), 'Preserve original review report')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    index = read(index_path)
    work = read(ROOT/'runs/model-portfolio-migration-v1/discovery/model-worklist-05.json')
    material = {r['id']:r for r in read(ROOT/'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')['models']}
    score = read(DOC/'reports/pressure-common-score-01.json')['models']
    timing_path = DOC/'reports/pressure-portfolio-GPU-timing-01.json'
    timing = read(timing_path)
    baseline_path = DOC/'reports/captured-targeted-supplement-method-reviews-01.json'
    baseline = next(row for row in read(baseline_path)['rows'] if row['named_checkpoint'] == 'C42')
    initial = read(ROOT/baseline['history']['path'])['checkpoints']['checkpoint-000.npz']['inventory']
    population = population_identity(read(ROOT/baseline['training_population']['path'])['parents'])
    lessons = {
        'C43':'C43 improves both common worst metrics over C42. Against the matched late-direction C44, early feedback gives lower worst pressure error but higher worst equation error. Early and late describe residual stages of a fixed eight-correction source, not physical simulation time or extra generated frames.',
        'C45':'C45 is the zero-first-pressure-weight control on the expanded 192-case direction bank. It differs from C44 in 48 additional manufactured long-axis directions as well as regenerated bank identity; do not attribute that comparison solely to an objective coefficient.',
        'C46':'Relative to C45, a weight1.0 first-correction objective lowers worst equation error but raises worst pressure error. A lower residual tail does not guarantee a lower pressure tail, and this training term does not add inference passes.',
        'C47':'A weight0.1 first-correction objective improves both common worst errors over C45 and C46. This favors a gentler training weight on the retained diagnostic population; independent seeds, held-out domains and rollouts remain unqualified.',
        'C48':'Weight0.25 improves worst equation error over C45 but slightly raises worst pressure error. C47 at0.1 has lower values of both common worst errors. More first-correction supervision is not monotonically better.'}
    rows, notes, facts = [], [], []
    for name, stage, first_weight in SPECS:
        f = load_fit(name,work=work,material=material,index=index,score=score)
        meta, config, history = f['metadata'], f['config'], f['history']
        require(f['inventory'] == initial and population_identity(f['split']['parents']) == population
                and (meta['actual_train_trajectories'],meta['actual_train_frames'],meta['actual_validation_trajectories'],meta['actual_validation_frames']) == (16,32,64,128)
                and meta['actual_parameter_count'] == 370016 and meta['initialized_tensor_count'] == 19,
                'Matched physical population/initial tensors differ')
        require(config['correction'] == dict(activation='linear',steps=2)
                and config['conditioning'] == 'physical-residual' and config['normalization'] == 'none'
                and config['objective'] == 'canonical-intermediate-contraction'
                and config['output_scale'] == 'previous-pressure-rms'
                and config['training']['updates'] == 5000 and config['training']['batch_size'] == 1
                and config['training']['seed'] == 20261015 and config['training']['sampling_seed'] == 2026102501
                and config['training']['learning_rate'] == .001
                and config['training']['schedule'] == [dict(learning_rate=.0001,update=3000),dict(learning_rate=.00001,update=4500)]
                and config.get('first_pressure_weight',0.) == first_weight, 'Captured deployment/optimizer/objective differs')
        primary, supplemental = audit_bank(f), audit_supplement(f)
        direction = audit_direction(f,stage)
        trainer = (f['project']/'train.py').read_text()
        model = (f['project']/'model.py').read_text()
        physical = (f['project']/'physical_residual.py').read_text()
        require('extra = model.training_loss(network, direction_sample, normalization,' in trainer
                and '(direction_weight*extra).backward()' in trainer
                and "module.register_parameter('bias',None)" in physical
                and 'if isinstance(layer,nn.SiLU):block.layers[index]=nn.Identity()' in physical,
                'Actual direction-loss or linear-core path differs')
        if name != 'C43':
            require('if FIRST_PRESSURE_WEIGHT and pressures:' in model
                    and 'first=pressure_loss(pressures[0],sample,normalization) if criterion is None else criterion(pressures[0])' in model
                    and 'result=result+FIRST_PRESSURE_WEIGHT*first' in model,
                    'First-pressure objective must use physical loss on physical data and selected criterion on auxiliary streams')
        common, perf = score[name]['worst'], timing['models'][name]['summary']
        row = dict(meta,**primary,supplemental_feedback=supplemental,direction_feedback=direction,
            first_correction_pressure_weight=first_weight,configured_pressure_corrections=2,
            matched_C42_initial_weights_and_physical_population=True,actual_train_cells_per_metre=[8,16],actual_validation_cells_per_metre=[8,16],
            retained_timing=binding(timing_path),
            method='A pressure-only 370,016-parameter multiscale 3D CNN with19weight tensors, one physically scaled residual input q=(rho/h)r and train-only RMS normalization. Executed transforms remove GroupNorm, SiLU and convolution biases despite inherited model labels. Two shared-weight F32 corrections recompute the F64 sealed Neumann residual, apply train previous-pressure RMS and cell-zero gauge, then accumulate in F64. dt is validated at the call boundary, not an independent neural input channel. Training-only captured directions and first-correction losses do not add deployed passes or advance velocity/tracer.',
            purpose=(f'Add train-only {stage} residual directions from the frozen C42 eight-correction policy while retaining primary and supplemental feedback.' if name == 'C43' else f'Test first-correction pressure feedback at weight{first_weight:g} using the expanded late-direction bank and unchanged two-correction inference.'),
            experiment=f'5,000Adam updates, batch1, initial rate0.001 with drops at3000/4500; model seed20261015 and physical sampling seed2026102501. The same16physical training trajectories/32frames at8/16cells per metre and dt1/60s, with axes at most16cells, supply fitting; validation uses64trajectories/128frames. Each update also uses one of1,024primary manufactured fields, one of768supplemental rectangular fields and one of{direction["cases"]}captured direction cases, each stream weighted0.1. Direction seed2026102701 balances{direction["balanced_groups"]}groups over5,000extra proposals; {direction["physical_train_frames"]}cases derive from existing real frames and{direction["manufactured_cases"]}from manufactured fields. '+('The early direction bank has144cases and no added long-axis bank.' if name == 'C43' else 'The192-case bank adds48manufactured directions on8x12x37 and12x8x45 and cyclic permutations, at16cells/metre. The first-correction objective uses canonical physical loss for real data and the selected analytical loss for auxiliary streams.')+f' Existing final loss and0.1intermediate residual contraction remain; additional first-correction loss weight is{first_weight:g}. These are residual-stage directions, not new CFD scenes or simulation timesteps.',
            outcome=f'Last logged weighted training objective {history["history"][-1]["loss"]:.17g}; retained final canonical validation objective {history["validation"]["final"]:.17g}. On128common consumed requests, worst mean-free pressure relative L2 {common["pressure_mean_free_relative_l2"]:.17g}; equation relative L2 {common["common_equation_relative_l2"]:.17g}. Complete GPU pressure-call median of case medians {perf["median_case_median_ms"]:.17g}ms, worst case median {perf["worst_case_median_ms"]:.17g}ms. Timing includes CPU validation/packing, H2D, all pressure work, independent CPU output/D2H/synchronization; setup/scoring excluded. No physical-step or trajectory-drift result follows.',
            lesson=lessons[name],
            reproduction='Use the captured typed train recipe/config/runtime, full source witness, physical inputs/order, normalization and initial tensors. Preserve primary/supplemental banks and exact frozen C42 direction-source weights, eight-correction extraction, independent label generation, paired manifest/payload, selected early/late stage, balanced private order and first-correction objective. This audit checks retained arrays against the sealed FV equation without rerunning extraction or training. Historical recipes contain CPU neural forwards; qualify a GPU-only execution route before reproduction. Clean retraining/new-Rheon regeneration remains unqualified.')
        original = (PROJECT/row['note_path']).read_bytes()
        receipt = verified_review_receipt(row)
        revised, suffix = revised_card(original,row)
        require(original.count(receipt) == revised.count(receipt) == 1 and revised.endswith(suffix), 'Original receipt/configuration suffix changed')
        rows.append(row)
        notes.append(binding(PROJECT/row['note_path']))
        facts.append(f)
    require(all(np.array_equal(facts[0]['sample_order'],f['sample_order']) for f in facts[1:]), 'Physical sample orders differ')
    require(all(canonical(facts[1]['history']['direction_feedback']) == canonical(f['history']['direction_feedback']) for f in facts[2:]),
            'First-pressure weight arms do not share their direction stream')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/x['path']) == x for x in notes), 'Native notes/index changed')
    require('torch' not in sys.modules, 'Review imported a neural runtime')
    value = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,
        source_reviewed_fits=5,new_model_forwards=0,native_mutations=0,new_reference_generation=False,
        clean_reproduction_qualified=False,original_note_bindings=notes,baseline=binding(baseline_path),
        overview_addition='\n\n## Captured residual directions and first-correction supervision\n\nC43 adds early directions, while C45–C48 share an expanded late-direction bank and sweep first-correction loss weight0/1/0.1/0.25. The0.1arm improves both common worst metrics over the other weight arms; large supervision is not monotonically better. Captured directions come from an existing fixed source and manufactured operator fields, not new physical scenes. Early/late describe pressure residual stages, not simulation timing.\n',
        producer=binding(Path(__file__)),helpers=[binding(ROOT/'experiments/model-portfolio-migration-v1'/name) for name in
            ['captured_pressure_review.py','review_pressure_budget_coverage_fits.py','review_targeted_supplement_fits.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(value)+b'\n')
    print('Five direction/first-correction fits reviewed; original array labels independently checked; no neural execution.')


if __name__ == '__main__':
    run()
