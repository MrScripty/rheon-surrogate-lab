"""Explain thirteen unnamed pipeline/conditioning fits without neural execution."""
from collections import Counter
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from captured_replay_pressure_review import check_tensors
from prepare_method_review_revisions import revised_card, verified_review_receipt
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT / 'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC / 'reports/captured-pressure-qualification-method-reviews-01.json'
SPECS = [
 ('fit-135202a799d78633','water-multigrid-learning-v1/multigrid-smoke',10,370016),
 ('fit-312d4f0dbc44117c','water-early-pressure-v1/smoke',1,370016),
 ('fit-5b3b1aef2efcee62','water-conditioning-v1/gpu-resident',500,371729),
 ('fit-6d1558cced039d83','water-learning-sequence-v1/pipeline-smoke',2,371729),
 ('fit-733a339f563172ab','water-residual-learning-v1/residual-smoke',2,370433),
 ('fit-803a07c660a70a93','water-late-residual-v1/smoke',1,370016),
 ('fit-9fa632f2b8082525','water-conditioning-v1/gpu-prepared',500,371729),
 ('fit-a5328ae3e2b2180f','water-stability-learning-v1/stability-smoke',2,370016),
 ('fit-acb450db5fec60b9','water-conditioning-v1/gpu-baseline',500,371729),
 ('fit-becf4ad5b5526ae4','water-conditioning-v1/gpu-batch4',500,371729),
 ('fit-befe8d913f41d1c8','water-pressure-recovery-v1/smoke',1,370016),
 ('fit-c69f29dfba8954b0','water-conditioning-v1/amplitude-smoke',2,371729),
 ('fit-c8638134ae71a368','water-multigrid-learning-v1/multigrid-cohort16-smoke',10,370016)]


def bank_draws(seed, shape, updates):
    rng = np.random.Generator(np.random.PCG64(seed))
    fixed = [(0,0,1),(0,1,0),(1,0,0),(0,1,2),(3,2,1),(4,4,4),(0,0,shape[2]-1),tuple(n-1 for n in shape)]
    components = []
    for index in range(32):
        if index < 8:
            terms = [(tuple(min(k,n-1) for k,n in zip(fixed[index],shape)),1.)]
        else:
            terms = []
            for _ in range(1 if index%2 else 3):
                mode = tuple(int(rng.integers(n)) for n in shape)
                terms.append((mode if any(mode) else (0,0,1),float(rng.uniform(.5,1.5))))
        components.extend(terms for _ in range(4))
    return components,rng.integers(128,size=updates,dtype=np.int64).astype('<i8')


def audit_auxiliary(config, history, inputs, updates):
    result = {}
    for kind in ['analytical_feedback','supplemental_feedback']:
        saved = history.get(kind)
        if not saved:
            continue
        spec = config[kind]
        require(saved['additional_examples'] == updates and saved['weight'] == spec['weight'], 'Actual auxiliary proposal budget differs')
        seed = spec.get('seed',2026101507)
        if 'grids' in saved:
            grids = saved['grids']
            rng = np.random.Generator(np.random.PCG64(seed))
            grid_order = np.concatenate([rng.permutation(len(grids)) for _ in range(math.ceil(updates/len(grids)))])[:updates]
            order = (grid_order*128 + rng.integers(128,size=updates,dtype=np.int64)).astype('<i8')
            require(np.bincount(grid_order,minlength=len(grids)).tolist() == saved['grid_exposures'], 'Auxiliary grid exposure differs')
            count = 128*len(grids)
            for entry in grids:
                bank = entry['bank']
                shape = tuple(reversed(entry['grid']['counts_xyz']))
                terms, _ = bank_draws(seed,shape,1)
                require([canonical(t) for t in terms] == [canonical(s['components_zyx']) for s in bank['samples']],
                        'Manufactured spatial frequency/mixture descriptors differ')
        else:
            terms, order = bank_draws(seed,tuple(saved['shape_zyx']),updates)
            require([canonical(t) for t in terms] == [canonical(s['components_zyx']) for s in saved['samples']],
                    'Single-grid manufactured descriptors differ')
            count = 128
        require(hashlib.sha256(order.tobytes()).hexdigest() == saved['sampling_order_sha256'], 'Independent auxiliary sample order differs')
        result[kind] = dict(fields=count,proposals=updates,weight=spec['weight'],sampling_order_rederived=True,
                            grid_count=len(saved.get('grids',[None])))
    saved = history.get('direction_feedback')
    if saved:
        spec = config['direction_feedback']
        manifest = read(inputs['direction_manifest'])
        require(binding(inputs['direction_manifest'])['sha256'] == spec['manifest_sha256'] == saved['manifest_sha256']
                and spec['stage'] == saved['stage'] and saved['additional_examples'] == updates,
                'Captured direction stage/manifest/proposal budget differs')
        groups = {}
        for i, case in enumerate(manifest['cases']):
            groups.setdefault(case['identity']['group'],[]).append(i)
        names = sorted(groups)
        rng = np.random.Generator(np.random.PCG64(spec['seed']))
        group_order = np.concatenate([rng.permutation(len(names)) for _ in range(math.ceil(updates/len(names)))])[:updates]
        order = np.asarray([rng.choice(groups[names[int(i)]]) for i in group_order],dtype='<i8')
        require(saved['groups'] == names and saved['cases'] == len(manifest['cases'])
                and saved['group_exposures'] == np.bincount(group_order,minlength=len(names)).tolist()
                and hashlib.sha256(order.tobytes()).hexdigest() == saved['sampling_order_sha256'],
                'Independent captured-direction group/case order differs')
        result['direction_feedback'] = dict(cases=len(manifest['cases']),stage=spec['stage'],groups=len(names),
            proposals=updates,weight=spec['weight'],sampling_order_rederived=True,label_arrays_decoded=False)
    return result


def run():
    require(not OUTPUT.exists(), 'Preserve original qualification-fit reviews')
    index_path = PROJECT / 'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    cards = {r['item_id']:r for r in read(index_path)['fits']}
    work = {r['item_id']:r for r in read(ROOT/'runs/model-portfolio-migration-v1/discovery/model-worklist-05.json')['fit_cards']}
    rows, notes, states = [], [], {}
    for fid,folder,updates,parameters in SPECS:
        project = ROOT / 'runs' / folder / 'project'
        receipt_path = ROOT / work[fid]['receipt_paths'][0]
        train = receipt_path.parent
        receipt, history = read(receipt_path), read(train/'history.json')
        config, witness = read(project/'config.json'), read(project/'source-witness.json')
        require(binding(receipt_path)['sha256'] == cards[fid]['receipt_sha256'] == work[fid]['receipt_sha256']
                and not cards[fid].get('method_review_status') and not work[fid]['named_entries']
                and receipt['status'] == 'complete' and receipt['mode'] == 'train'
                and receipt['optimizer_updates'] == config['training']['updates'] == updates,
                'Exact original unnamed qualification fit required')
        for name,key in [('config.json','experiment_sha256'),('source-witness.json','source_witness_sha256'),
                         ('runtime-manifest.json','runtime_manifest_sha256')]:
            require(binding(project/name)['sha256'] == receipt['closure'][key], 'Qualification source/config/runtime differs')
        sources = [binding(project/n) for n in sorted(witness['sources'])]
        require(all(s['sha256'] == witness['sources'][Path(s['path']).name] for s in sources), 'Complete original qualification source changed')
        document, recipe = read(project/'project.json'), read(project/'train-recipe.json')
        require(recipe['entrypoint'] == str((project/'train.py').relative_to(ROOT)), 'Original qualification entrypoint differs')
        for spec in [*document['recipes'],*recipe['sources']]:
            require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Qualification typed recipe/source differs')
        inputs = {r['id']:ROOT/r['path'] for r in document['data']}
        bound_inputs = []
        for slot,expected in receipt['closure']['inputs'].items():
            p = inputs[slot]
            b = binding(p)
            require(dict(sha256=b['sha256'],bytes=p.stat().st_size) == expected, 'Original qualification input differs')
            bound_inputs.append(dict(b,slot=slot,bytes=expected['bytes']))
        for name,expected in receipt['outputs'].items():
            p = train/name
            require(dict(sha256=binding(p)['sha256'],bytes=p.stat().st_size)
                    == {k:expected[k] for k in ('sha256','bytes')}, 'Original qualification output differs')
        split = read(train/'splits.json')
        parents = split['parents']
        eligible = [i for i,p in enumerate(parents) if p['entry']['split'] == 'train']
        validation = [p for p in parents if p['entry']['split'] == 'validation']
        train_frames = sum(len(parents[i]['frames']) for i in eligible)
        require(len(eligible) in (1,16) and train_frames in (1,32) and len(validation) == 64
                and sum(len(p['frames']) for p in validation) == 128
                and all(p['entry']['dt_s'] == 1/60 and p['density_kg_m3'] == 1000 for p in parents),
                'Actual qualification physical population/units differ')
        rng = np.random.Generator(np.random.PCG64(config['training']['sampling_seed']))
        order = []
        for _ in range(updates):
            batch = []
            for _ in range(config['training']['batch_size']):
                p = eligible[int(rng.integers(len(eligible)))]
                batch.append((p,int(rng.integers(len(parents[p]['frames'])))))
            order.append(batch)
        require(hashlib.sha256(np.asarray(order,dtype='<i8').tobytes()).hexdigest() == receipt['closure']['sample_order_sha256'],
                'Independent qualification physical sample order differs')
        for name,spec in history['checkpoints'].items():
            require(sum(math.prod(v['shape']) for v in spec['inventory'].values()) == parameters,
                    'Actual qualification parameter count differs')
            values = check_tensors(train/name,spec['inventory'])
            if name == 'checkpoint-000.npz':
                initial = values
        selected = f'checkpoint-{updates:03d}.npz'
        require(receipt['outputs']['weights.npz']['sha256'] == receipt['outputs'][selected]['sha256']
                and binding(train/'normalization.json')['sha256'] == receipt['closure']['normalization_sha256'],
                'Final qualification candidate/normalization differs')
        final = check_tensors(train/'weights.npz',history['checkpoints'][selected]['inventory'])
        states[folder] = dict(initial=initial,final=final)
        auxiliary = audit_auxiliary(config,history,inputs,updates)
        conditioning = config.get('conditioning') == 'amplitude-and-units'
        physical = config.get('conditioning') == 'physical-residual'
        corrections = config.get('correction',{}).get('steps',1)
        strategy = config.get('execution',{})
        if physical:
            linear = config['correction']['activation'] == 'linear'
            method=f'A {parameters:,}-parameter F32 one-channel multiscale pressure CNN uses physical q=(rho/h)r and {corrections} internal correction(s), recomputing the F64 equation residual between corrections. GroupNorm is removed; ' + ('SiLU and all convolution biases are also removed.' if linear else 'SiLU and convolution biases remain.')
            method += f' The captured output scale is {config.get("output_scale","previous-pressure-rms")}; cell-zero-gauged corrections accumulate in F64. The objective is {config.get("objective","canonical")}. dt is validated metadata, not an independent neural channel.'
        else:
            method=f'A {parameters:,}-parameter seven-channel F32 multiscale CNN predicts an additive pressure correction from native residual, previous pressure, physical XYZ coordinates, fluid density and dt. Four encoder levels, global-mean bottleneck and skip decoder retain non-affine GroupNorm and SiLU. Correction scaling/gauge and equation checks use F64.'
            if conditioning:
                method += ' Per-sample residual RMS is normalized relative to train RMS and restored at output; fixed units rescale rho by 1,000 and dt by 1/60. This changes conditioning without adding parameters.'
        method += ' It emits pressure only and advances no physical simulation time.'
        role = 'CUDA managed training/restore qualification'
        if updates == 500:
            role = 'compare training execution strategies with the same one-frame amplitude-conditioned task'
        elif 'multigrid' in folder:
            role = 'qualify manufactured multi-grid feedback before a longer scientific fit'
        elif 'residual' in folder or 'stability' in folder or 'recovery' in folder:
            role = 'qualify the captured physical residual/objective/auxiliary-stream path before its full-budget study'
        if folder.endswith('gpu-prepared'):
            require(strategy.get('prepared_fields') is True, 'Prepared-field qualification must actually prepare inputs')
        row = dict(fit_item_id=fid,note_path=cards[fid]['card_path'],original_receipt=binding(receipt_path),
            history=binding(train/'history.json'),sources=sources,source_witness=binding(project/'source-witness.json'),
            original_configuration=binding(project/'config.json'),original_recipe=binding(project/'train-recipe.json'),
            training_inputs=bound_inputs,training_population=binding(train/'splits.json'),parameters=parameters,
            optimizer_updates=updates,training_trajectories=len(eligible),training_frames=train_frames,
            validation_trajectories=64,validation_frames=128,auxiliary_training=auxiliary,
            sampled_order_independently_rederived=True,method=method,
            purpose=f'This unnamed fit is retained to {role}. It is distinct from any later named final-budget candidate.',
            experiment=f'{updates} CUDA Adam update(s), batch {config["training"]["batch_size"]}, rate {config["training"]["learning_rate"]}, seed {config["training"]["seed"]}, private physical sampling seed {config["training"]["sampling_seed"]}. The actual optimizer population has {len(eligible)} physical parent(s)/{train_frames} frame(s); validation has 64 parents/128 frames. Training metadata contains grid resolutions {sorted({1/parents[i]["cell_size_m"] for i in eligible})} cells per metre and fixed 1/60 s steps. Execution strategy is {strategy}. Active auxiliary streams: {auxiliary}. Manufactured fields and frozen captured correction directions are additional operator examples, not independent simulation scenes. Original typed input/output closures, every retained checkpoint and sample orders verify without importing model code.',
            outcome=f'Original final sampled loss {history["history"][-1]["loss"]:.17g}; recorded validation objective {history["validation"]["initial"]:.17g} → {history["validation"]["final"]:.17g}. Final-update weights remain selected; no optimizer-resume state is retained. These are historical objective values, not worst pressure relative error. A matched current GPU benchmark for this unnamed fit, full-state accuracy and own-history drift remain pending.',
            lesson=('The baseline, resident and prepared variants fit one identical physical frame; batch four repeats that same frame four times per update. Those names describe execution, not four-scene coverage. Exact retained tensor comparison can establish value parity, while historical training throughput observations alone cannot establish current inference speed or broad generalization.' if updates == 500 else
                    'A one-, two- or ten-update qualification run proves execution/custody coverage at that budget, not a full scientific training result. Validation population size does not enlarge the optimizer population. Folder names such as early/late do not establish direction stage: preserve the actual recorded auxiliary manifest and stage.'),
            reproduction='Use the exact captured typed train/restore recipe, source witness, runtime, original input blobs, whole-parent selection, train-only scales, private sample order and all checkpoint inventories. Preserve execution strategy and separately counted auxiliary streams. The historical trainer performs CPU publication/restore neural forwards; qualify a GPU-only route before rerunning under current policy. Clean retraining, new-Rheon regeneration and a current common benchmark for this unnamed fit remain unqualified. Original receipts/configuration and historical Trial ancestry remain unchanged.')
        raw = (PROJECT/row['note_path']).read_bytes()
        original = verified_review_receipt(row)
        revised,suffix = revised_card(raw,row)
        require(raw.count(original) == revised.count(original) == 1 and revised.endswith(suffix), 'Original qualification facts changed')
        rows.append(row)
        notes.append(binding(PROJECT/row['note_path']))
    variants = ['gpu-baseline','gpu-resident','gpu-prepared','gpu-batch4']
    baseline = states['water-conditioning-v1/gpu-baseline']
    parity = {}
    for variant in variants:
        values = states['water-conditioning-v1/'+variant]
        require(all(values['initial'][n].tobytes() == baseline['initial'][n].tobytes() for n in baseline['initial']),
                'Execution variants must retain identical initial coefficients')
        parity[variant] = all(values['final'][n].tobytes() == baseline['final'][n].tobytes() for n in baseline['final'])
    require(parity['gpu-resident'] and parity['gpu-prepared'], 'Resident/prepared retained final tensors differ')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/n['path']) == n for n in notes)
            and 'torch' not in sys.modules, 'Read-only/no-neural review invariant failed')
    result = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,
        source_reviewed_fits=13,original_note_bindings=notes,execution_variant_final_tensor_parity=parity,
        new_model_forwards=0,native_mutations=0,new_reference_generation=False,clean_reproduction_qualified=False,
        producer=binding(Path(__file__)),helpers=[binding(Path(__file__).with_name(n)) for n in
            ['captured_replay_pressure_review.py','prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(result)+b'\n')
    print('Thirteen qualification fits reviewed; actual physical budgets, auxiliary sampling and execution tensor parity verified.')


if __name__ == '__main__':
    run()
