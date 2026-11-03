"""Review six unnamed control/qualification fits, preserving each fit identity."""
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from captured_replay_pressure_review import check_tensors
from prepare_method_review_revisions import revised_card, verified_review_receipt
from review_current_pressure_feedback import audit_current
from review_pressure_qualification_fits import audit_auxiliary
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC/'reports/captured-pressure-execution-control-method-reviews-01.json'
SPECS = [
 ('fit-047049129e1e2833','pressure-spatial-coarse-v1/control-project-01',5000,'evidence.py'),
 ('fit-136881e20046e15a','pressure-throughput-feedback-v1/lowband-smoke-project-03',1,'evidence.py'),
 ('fit-a928b38bd18e1c6a','pressure-throughput-feedback-v1/control-project-04',5000,'evidence.py'),
 ('fit-c675a37816966f2b','dependency-migration-v1/training-project-6',1,'smoke_train.py'),
 ('fit-fb8ce15096322ec9','dependency-migration-v1/typed-pressure-project-02',1,'typed_pressure_evidence.py'),
 ('fit-fc820329dacc4d2c','dependency-migration-v1/training-project-6',1,'smoke_train.py'),
]


def run():
    require(not OUTPUT.exists(),'Preserve original execution-control reviews')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    cards = {f['item_id']:f for f in read(index_path)['fits']}
    work = {f['item_id']:f for f in read(ROOT/'runs/model-portfolio-migration-v1/discovery/model-worklist-05.json')['fit_cards']}
    material = read(ROOT/'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')['models']
    rows,notes,states = [],[],{}
    for fid,folder,updates,entrypoint in SPECS:
        fit,card,project = work[fid],cards[fid],ROOT/'runs'/folder
        candidates = [ROOT/p for p in fit['receipt_paths']]
        require(all(binding(p)['sha256'] == fit['receipt_sha256'] == card['receipt_sha256'] for p in candidates)
                and not card.get('method_review_status') and not fit['named_entries'],
                'Original unnamed fit/receipt aliases differ')
        receipt_path = candidates[0]
        training = receipt_path.parent
        receipt,history = read(receipt_path),read(training/'history.json')
        config,witness = read(project/'config.json'),read(project/'source-witness.json')
        manifest,recipe = read(project/'project.json'),read(project/'train-recipe.json')
        inputs = {r['id']:ROOT/r['path'] for r in manifest['data']}
        runtime_path = inputs['runtime_manifest'] if 'runtime_manifest' in inputs else project/'runtime-manifest.json'
        require(receipt['mode'] == 'train' and receipt['status'] == 'complete'
                and receipt['optimizer_updates'] == config['training']['updates'] == updates
                and config['training']['batch_size'] == 1
                and config['correction'] == dict(activation='linear',steps=2),
                'Original execution-control completed budget differs')
        for name,key in [('config.json','experiment_sha256'),('source-witness.json','source_witness_sha256')]:
            require(binding(project/name)['sha256'] == receipt['closure'][key],'Original execution-control closure differs')
        require(binding(runtime_path)['sha256'] == receipt['closure']['runtime_manifest_sha256']
                == witness['runtime_manifest_sha256'],'Original declared runtime differs')
        sources = [binding(project/n) for n in sorted(witness['sources'])]
        require(all(s['sha256'] == witness['sources'][Path(s['path']).name] for s in sources),
                'Complete original execution-control source differs')
        require(recipe['entrypoint'] == str((project/entrypoint).relative_to(ROOT))
                and recipe['runtime'] == 'managed_torch_python_v1','Actual execution-control recipe wrapper differs')
        for spec in [*manifest['recipes'],*recipe['sources']]:
            require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'],'Typed wrapper/source binding differs')
        bound_inputs = []
        for slot,expected in receipt['closure']['inputs'].items():
            path = inputs[slot]
            observed = binding(path)
            require(dict(sha256=observed['sha256'],bytes=path.stat().st_size) == expected,'Original execution-control input differs')
            bound_inputs.append(dict(observed,slot=slot,bytes=expected['bytes']))
        locations,output_paths = [],{}
        renamed = entrypoint in ('evidence.py','typed_pressure_evidence.py')
        for name,expected in receipt['outputs'].items():
            retained = 'training-profile.json' if name == 'profile.json' and renamed else name
            path = training/retained
            native_only = fid == 'fit-fb8ce15096322ec9' and name in {
                'checkpoint-000.npz','profile.json','splits.json','summary.json'}
            if native_only:
                require(not path.exists(),'Expected prospective partial download layout changed')
                path = PROJECT/'artifacts/sha256'/expected['sha256']
            require(dict(sha256=binding(path)['sha256'],bytes=path.stat().st_size)
                    == {k:expected[k] for k in ('sha256','bytes')},'Original execution-control output/profile differs')
            output_paths[name] = path
            locations.append(dict(original_name=name,retained_path=str(path.relative_to(ROOT)),sha256=expected['sha256'],
                resolved_in_existing_native_content_store=native_only,missing_from_partial_local_download=native_only))
        require(binding(output_paths['normalization.json'])['sha256'] == receipt['closure']['normalization_sha256']
                and history['diagnostic_population'] == receipt['diagnostic_population'] == 'train_only'
                and history['heldout_evaluations'] == receipt['heldout_evaluations'] == 0
                and not receipt['scientific_acceptance_or_model_promotion'],'Train-only execution diagnostics scope differs')
        parents = read(output_paths['splits.json'])['parents']
        eligible = [i for i,p in enumerate(parents) if p['entry']['split'] == 'train']
        require(len(parents) == len(eligible) == 16 and sum(len(p['frames']) for p in parents) == 32
                and all(p['entry']['dt_s'] == 1/60 and p['density_kg_m3'] == 1000 for p in parents),
                'Execution-control physical population differs')
        rng = np.random.Generator(np.random.PCG64(config['training']['sampling_seed']))
        order = []
        for _ in range(updates):
            parent = eligible[int(rng.integers(len(eligible)))]
            order.append([(parent,int(rng.integers(len(parents[parent]['frames']))))])
        require(hashlib.sha256(np.asarray(order,dtype='<i8').tobytes()).hexdigest() == receipt['closure']['sample_order_sha256'],
                'Execution-control private physical draw order differs')
        initial_inventory = history['checkpoints']['checkpoint-000.npz']['inventory']
        require(sum(math.prod(v['shape']) for v in initial_inventory.values()) == 370016 and len(initial_inventory) == 19,
                'Actual execution-control parameter inventory differs')
        initial = check_tensors(output_paths['checkpoint-000.npz'],initial_inventory)
        final = check_tensors(output_paths['weights.npz'],history['checkpoints'][f'checkpoint-{updates:03d}.npz']['inventory'])
        require(receipt['outputs']['weights.npz']['sha256'] == receipt['outputs'][f'checkpoint-{updates:03d}.npz']['sha256'],
                'Execution-control selected final archive differs')
        auxiliary = audit_auxiliary(config,history,inputs,updates)
        current = audit_current(dict(project=project,config=config,history=history)) if config.get('current_feedback') else None
        exact_named = [m['id'] for m in material if m['weights_sha256'] == receipt['outputs']['weights.npz']['sha256']]
        band = config.get('band_feedback',{}).get('weight',0.)
        purposes = {
            'fit-047049129e1e2833':'Establish the base-network control for adding the learned spatial coarse branch; this is a separate completed fit with parameter bytes equal to C53.',
            'fit-136881e20046e15a':'Qualify one actual CUDA update with first/final low-spatial-mode loss at weight 0.1 before a full fitting run.',
            'fit-a928b38bd18e1c6a':'Establish the full-length zero-low-mode-weight throughput/control arm; its final parameter archive equals named C45.',
            'fit-c675a37816966f2b':'Exercise the managed Torch training/restore lifecycle with one unchanged CUDA optimizer update and train-only diagnostics.',
            'fit-fb8ce15096322ec9':'Qualify a prospective typed pressure fit, exact checkpoint/profile publication and one already-consumed bound field probe.',
            'fit-fc820329dacc4d2c':'Exercise the CPU/CUDA scheduler-overlap qualification with one managed CUDA fit; this does not measure a faster deployed model.'}
        if entrypoint == 'smoke_train.py':
            require('train.evaluate = diagnostic' in (project/entrypoint).read_text(),'Train-only smoke owner differs')
        else:
            require('(output / "profile.json").rename(output / "training-profile.json")' in (project/entrypoint).read_text(),
                    'SDK training-profile custody differs')
        if fid == 'fit-fb8ce15096322ec9':
            lifecycle = read(ROOT/'runs/dependency-migration-v1/typed-pressure-live-01/receipt.json')
            require(lifecycle['project_id'] == manifest['id'],'Original prospective lifecycle project differs')
        row = dict(fit_item_id=fid,note_path=card['card_path'],named_checkpoint=fit['name'],
            original_receipt=binding(receipt_path),receipt_aliases=[binding(p) for p in candidates],
            history=binding(training/'history.json'),sources=sources,source_witness=binding(project/'source-witness.json'),
            original_configuration=binding(project/'config.json'),original_recipe=binding(project/'train-recipe.json'),
            original_runtime=binding(runtime_path),
            training_population=binding(output_paths['splits.json']),training_inputs=bound_inputs,original_output_locations=locations,
            actual_parameter_count=370016,initialized_tensor_count=19,actual_train_trajectories=16,actual_train_frames=32,
            actual_validation_trajectories=0,actual_validation_frames=0,optimizer_updates=updates,
            sample_order_independently_rederived=True,auxiliary_training=auxiliary,current_feedback=current,
            byte_equal_named_parameter_archives=exact_named,clean_reproduction_qualified=False,new_model_forwards=0,
            method='A 370,016-parameter linear, bias-free F32 multiscale pressure CNN with one physically scaled residual channel and 19 weight tensors. Two shared neural pressure corrections recompute the F64 sealed cubic-grid equation residual, apply fixed train-only scales and accumulate cell-zero-gauged F64 pressure. Inherited GroupNorm/SiLU labels do not describe its executed linear blocks. It returns pressure, not a physical simulation step; no learned spatial coarse branch is present.',
            purpose=purposes[fid],
            experiment=f'{updates:,} CUDA Adam updates, batch one, rate {config["training"]["learning_rate"]:g}, model seed {config["training"]["seed"]}, physical-index seed {config["training"]["sampling_seed"]}, schedule {config["training"]["schedule"]}. Sixteen physical fitting trajectories/32 frames at 8/16 cells per metre, dt=1/60 s and fluid density 1000 kg/m3. Both manufactured banks and the fixed captured-direction bank remain active; '+
                ('an additional frozen current-policy bank supplies late directions. ' if current else '')+
                f'First/final low-spatial-mode feedback weight is {band:g}. The actual entrypoint is {entrypoint}; diagnostics use training parents only. '+
                ('The typed evidence wrapper also queries one already-consumed validation frame for publication; it supplies no independent holdout.' if renamed else 'The smoke wrapper does not supply a scientific model-promotion decision.'),
            outcome=f'Retained train-only final objective {history["validation"]["final"]:.17g}; last optimizer loss {history["history"][-1]["loss"]:.17g}. '+
                (f'Final parameter archive is byte-identical to {", ".join(exact_named)}; this does not merge original fit receipts or establish identical execution contracts. ' if exact_named else '')+
                'The three one-update pipeline/prospective/scheduler fits share exact initial and final parameter values. Four prospective outputs missing from its partial local download are freshly verified in the existing native content store using the original receipt hashes; no replacement or copy is created. No fresh common GPU inference, complete simulation-step timing or drift benchmark is added for these fit identities. Historical scheduler/training wall time and normalized diagnostic loss are not deployed latency or worst pressure relative error.',
            lesson='Keep lifecycle qualification, completed optimizer budgets and scientific accuracy qualification distinct. Repeating an unchanged fit for publication or scheduling can reproduce parameter bytes without creating new scene diversity or an architectural improvement. Byte-equal weights may share retained storage later, but original fit/Run evidence remains separate. A one-update smoke is not a full training result or generalization test.',
            reproduction='Retain the exact typed entrypoint and its explicit recipe source bindings, original config/runtime/source witness, all physical/manufactured/captured inputs, train-only scales, draw order and both checkpoints. Preserve native profile renaming and any original consumed publication probe. The historical trainer/restore/publication invokes CPU neural inference; qualify GPU-only equivalents before replay. No original receipt alias, Trial ancestry or scientific scalar is rewritten. Clean retraining/new-Rheon generation and long own-history drift remain pending.')
        raw = (PROJECT/row['note_path']).read_bytes()
        original = verified_review_receipt(row)
        revised,suffix = revised_card(raw,row)
        require(raw.count(original) == revised.count(original) == 1 and revised.endswith(suffix),'Original execution-control facts changed')
        rows.append(row)
        notes.append(binding(PROJECT/row['note_path']))
        states[fid] = (initial,final)
    matched = ['fit-c675a37816966f2b','fit-fb8ce15096322ec9','fit-fc820329dacc4d2c']
    require(all(all(states[fid][j][k].tobytes() == states[matched[0]][j][k].tobytes()
            for k in states[matched[0]][j]) for fid in matched for j in (0,1)),
            'Matched pipeline/prospective/scheduler tensor parity differs')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/n['path']) == n for n in notes)
            and 'torch' not in sys.modules,'Read-only execution-control invariant failed')
    result = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=6,
        original_note_bindings=notes,new_model_forwards=0,native_mutations=0,new_reference_generation=False,
        clean_reproduction_qualified=False,matched_one_update_initial_final_tensors_exact=True,producer=binding(Path(__file__)),
        helpers=[binding(Path(__file__).with_name(n)) for n in ['captured_replay_pressure_review.py','review_current_pressure_feedback.py',
            'review_pressure_qualification_fits.py','prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(result)+b'\n')
    print('Six unnamed controls/qualification fits reviewed; exact receipt/recipe/data/checkpoint custody and three-way parameter parity checked.')


if __name__ == '__main__':
    run()
