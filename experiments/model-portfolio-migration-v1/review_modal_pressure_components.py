"""Review four geometry-specific F64 pressure fits; never execute learned gains."""
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from prepare_method_review_revisions import revised_card, verified_review_receipt
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
BASE = ROOT/'runs/full-ai-tight-targets-v1'
OUTPUT = DOC/'reports/captured-modal-pressure-component-method-reviews-01.json'
SPECS = [
    ('fit-83a63b7141c12414','modal-pressure-project-02','modal-pressure-pair-01','train_modal_pressure.py'),
    ('fit-561e02c747b96415','modal-precision-project-02','modal-precision-pair-02','train_modal_precision.py'),
    ('fit-b48ebd35d5c9448d','decoder-modal-project-09','decoder-modal-pair-09','train_modal_precision.py'),
    ('fit-aaf0a360c4c9622b','mixed-modal-project-11b','mixed-modal-pair-11','train_modal_augmented.py')]


def run():
    require(not OUTPUT.exists(), 'Preserve original modal component reviews')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    cards = {r['item_id']:r for r in read(index_path)['fits']}
    rows, notes, source_versions, previous_data = [], [], {}, None
    for sequence,(fid,folder,pair,trainer_name) in enumerate(SPECS):
        project, train = BASE/folder,BASE/pair/'train'
        receipt_path = train/'receipt.json'
        receipt, config, history = read(receipt_path),read(project/'config.json'),read(train/'history.json')
        initial_fit, mixed = sequence == 0, sequence == 3
        updates = 800 if initial_fit else 80
        require(receipt['status'] == 'complete' and receipt['mode'] == 'train'
                and receipt['optimizer_updates'] == config['updates'] == history['optimizer_updates'] == updates
                and not receipt['causal_full_state_generator'] and receipt['parameters'] == receipt['trainable_parameters'] == 38673
                and receipt['execution']['device'] == 'cuda' and receipt['execution']['dtype'] == 'float64',
                'Original pressure-component fit role/budget differs')
        require(binding(project/'config.json')['sha256'] == receipt['closure']['config_sha256'], 'Original configuration differs')
        sources = [binding(project/n) for n in sorted(receipt['closure']['sources'])]
        require(all(s['sha256'] == receipt['closure']['sources'][Path(s['path']).name] for s in sources), 'Original source closure changed')
        document, recipe = read(project/'project.json'), read(project/'train-recipe.json')
        require(recipe['entrypoint'] == str((project/trainer_name).relative_to(ROOT)), 'Actual trainer selection differs')
        for spec in [*document['recipes'],*recipe['sources']]:
            require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Typed recipe source changed')
        inputs = {r['id']:ROOT/r['path'] for r in document['data']}
        bound_inputs = []
        for slot,expected in receipt['closure']['inputs'].items():
            spec = binding(inputs[slot])
            require(dict(sha256=spec['sha256'],bytes=inputs[slot].stat().st_size) == expected, 'Original training input changed')
            bound_inputs.append(dict(spec,slot=slot,bytes=expected['bytes']))
        manifest, norm = read(inputs['manifest']),read(inputs['normalization'])
        require(manifest['status'] == 'verified_train_only_fixed_transform_view' and manifest['scenes'] == 16
                and manifest['samples'] == (272 if mixed else 144) and not manifest['development_or_reserved_used']
                and not norm['development_or_reserved_used']
                and manifest['normalization_sha256'] == receipt['closure']['normalization_sha256'] == binding(inputs['normalization'])['sha256']
                and history['training_scene_count'] == 16 and history['train_only']
                and history['full_batch_geometry_updates'] == 4*updates, 'Original training population/work differs')
        expected_shapes = {'g16_16_16':[16,16,16],'g11_11_9':[11,11,9],'g8_6_15':[8,6,15],'g32_32_32':[32,32,32]}
        require({g['key']:g['counts_zyx'] for g in manifest['geometries']} == expected_shapes,
                'Registered finite geometry set differs')
        require(sum(math.prod(s) for s in expected_shapes.values()) == 38673, 'Independent modal parameter count differs')
        for path,sha in manifest['source_evidence'].items():
            require(binding(ROOT/path)['sha256'] == sha, 'Original physical source evidence changed')
        require(cards[fid]['receipt_sha256'] == binding(receipt_path)['sha256'] and not cards[fid].get('method_review_status'),
                'Card belongs to another/already-reviewed fit')
        for name,expected in receipt['outputs'].items():
            p = train/name
            require(dict(sha256=binding(p)['sha256'],bytes=p.stat().st_size) == expected, 'Original fit output changed')
        for filename,key in [('initial-weights.npz','initial_inventory'),('weights.npz','final_inventory')]:
            inventory = receipt[key]
            require(inventory == history[key] and set(inventory) == {'log_gains.'+k for k in expected_shapes},
                    'Four complete parameter tensors required')
            with np.load(train/filename,allow_pickle=False) as arrays:
                require(set(arrays.files) == set(inventory), 'Complete retained parameter archive required')
                for name,spec in inventory.items():
                    value = arrays[name]
                    require(value.dtype.str == spec['dtype'] == '<f8' and list(value.shape) == spec['shape'] == expected_shapes[name.split('.')[1]]
                            and np.isfinite(value).all() and hashlib.sha256(value.tobytes()).hexdigest() == spec['sha256']
                            and value[0,0,0] == 0., 'Original F64 parameters or fixed DC state differ')
                    if initial_fit and key == 'initial_inventory':
                        require(np.count_nonzero(value) == 0, 'First modal fit must start from uniform non-DC gains')
        parent_fit = None
        if not initial_fit:
            parent = ROOT/norm['initial_pressure_pair']/'restore'
            parent_receipt = read(parent/'receipt.json')
            require(binding(parent/'weights.npz')['sha256'] == norm['initial_pressure_archive_sha256']
                    and parent_receipt['full_cpu_probe_bitwise_verified'] and parent_receipt['optimizer_updates'] == 0
                    and parent_receipt['final_inventory'] == receipt['initial_inventory'] == norm['initial_pressure_inventory'],
                    'Exact inherited learned initialization differs')
            for name,values in norm['initial_pressure_parameters'].items():
                value = np.asarray(values,dtype='<f8')
                require(hashlib.sha256(value.tobytes()).hexdigest() == receipt['initial_inventory'][name]['sha256'],
                        'JSON transport changed inherited F64 parameter bits')
            parent_fit = rows[-1]['fit_item_id']
            require(read(ROOT/rows[-1]['original_receipt']['path'])['final_inventory'] == receipt['initial_inventory'],
                    'Recorded sequence does not preserve exact previous fit parameters')
        geometry_rows, dataset = [], {}
        for i,g in enumerate(manifest['geometries']):
            p = inputs[f'part_{i:02d}']
            require(binding(p)['sha256'] == g['sha256'] and p.stat().st_size == g['bytes'], 'Original coefficient shard differs')
            with np.load(p,allow_pickle=False) as arrays:
                require(set(arrays.files) == {'q_coefficients','pressure_coefficients'}, 'Only request/reference label coefficients required')
                q,pressure = [arrays[n] for n in ('q_coefficients','pressure_coefficients')]
                n = 68 if mixed else 36
                require(all(a.dtype.str == '<f8' and list(a.shape) == [n,*g['counts_zyx']] and np.isfinite(a).all()
                            for a in (q,pressure)), 'Full coefficient dtype/extent differs')
                data_identity = {name:hashlib.sha256(a[:36].tobytes()).hexdigest() for name,a in [('q',q),('p',pressure)]}
                if sequence in (1,3):
                    require(data_identity == previous_data[g['key']], 'Native coefficient pairs changed in refinement/mixed augmentation')
                dataset[g['key']] = data_identity
                coverage = history['coverage'][g['key']]
                if initial_fit:
                    rms = np.sqrt(np.mean(q*q,axis=(1,2,3),keepdims=True))
                    valid = (q*pressure > 0)&(np.abs(q) > config['label_relative_q_floor']*rms)
                    valid[:,0,0,0] = False
                    observed = int(np.any(valid,axis=0).sum())
                    require(observed == coverage['observed_nonDC_modes'] and int(valid.sum()) == coverage['valid_sample_mode_labels']
                            and math.prod(g['counts_zyx'])-1-observed == coverage['unobserved_nonDC_modes'],
                            'Independent native log-label coverage differs')
                else:
                    require(coverage['observed_nonDC_modes'] == math.prod(g['counts_zyx'])-1
                            and coverage['native_samples'] == n and config['mode_scale_floor'] == 1e-12,
                            'Precision label/mode coverage differs')
                if mixed:
                    offline = g['offline_reference']
                    require(binding(inputs['manifest'].parent/offline['path'])['sha256'] == offline['sha256']
                            and g['original_native_samples'] == 36 and g['augmented_component_queries'] == 32,
                            'Original offline mixed-query label evidence differs')
                geometry_rows.append(dict(key=g['key'],counts_zyx=g['counts_zyx'],samples=n,source=binding(p),
                    original_native_pair_hashes=data_identity,label_coverage=coverage))
        previous_data = dataset
        model = (project/'learned_modal_pressure.py').read_text()
        trainer = (project/trainer_name).read_text()
        require('return q_coefficients * self.gains(key)' in model and 'values = self.log_gains[key].exp()' in model
                and 'mask[0, 0, 0] = 0.' in model, 'Actual finite-table modal predictor differs')
        source_versions[folder] = {Path(s['path']).name:s['sha256'] for s in sources}
        if initial_fit:
            require('torch.optim.Adam' in trainer and 'math.cos(math.pi * fraction)' in trainer,
                    'Original full-batch Adam/cosine schedule differs')
        else:
            require("torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=5,max_eval=8,history_size=48," in trainer
                    and history['coverage']['optimizer_closure_calls'] == history['history'][-1]['closure_calls'],
                    'Original LBFGS work accounting differs')
        qualifier = ('uniform zero log-gains' if initial_fit else 'the exact preceding learned checkpoint')
        method = 'A pressure-only F64 diagonal model with 38,673 stored log-gains across four registered grid shapes. Each mode has its own positive learned gain; four DC entries are stored but masked from prediction, leaving 38,669 active non-DC gains. Fixed full DCT-II transforms supply coordinates. The model multiplies supplied request coefficients by exponentiated gains; it does not advance velocity or tracer. Physical spacing enters the external q=(rho/h)RHS unit conversion, not an independent learned density channel. Unknown grid keys/shapes require a new exported or trained table.'
        purpose = ['Test whether a geometry-specific full modal capacity can fit native pressure requests in F64 from uniform gains, before reducing parameter sharing.',
            'Refine the completed modal model with per-mode weighted native coefficient least squares to improve numerical precision without prescribing analytical inverse-gain labels.',
            'Train the same learned gains against labels encoded in the fixed decoder’s actual coordinates, addressing numerical coordinate reconstruction rather than changing physical scenes.',
            'Excite all modes jointly with additional offline reference-labeled queries to improve modal coverage and separate weak native-mode signal from coefficient precision.'][sequence]
        training = ('All 144 original pairs from 16 consumed training scenes, nine snapshots per scene at 8/16 cells per metre. ' if not mixed else
            'The same 144 native pairs are retained bitwise, plus 128 synthetic mixed-mode queries, 32 per geometry. These are extra component queries, not 128 new simulations or scenes. ')
        training += f'Each full-batch update sees all four geometry groups. Initialization is {qualifier}. '
        training += ('800 CUDA Adam updates use cosine rate decay from 0.1 to 0.001. Same-sign native p/q modes above a 1e-10 relative request floor supervise log-pressure minus log-request magnitudes; no analytic eigenvalue initializer is loaded.' if initial_fit else
            f'80 CUDA LBFGS outer updates use strong-Wolfe search with {history["coverage"]["optimizer_closure_calls"]} recorded closure evaluations. Native coefficient errors are normalized by train-only per-mode pressure RMS with a 1e-12 relative floor; no p/q coefficient target is formed.')
        lesson = ['This is a finite geometry-specific gain table, not the later compact shared four-coefficient pressure component. Fitting all modes of four shapes does not prove unseen-shape or full-state generalization.',
            'An 80-step LBFGS refinement is not 80 single Adam forwards: line-search closure work matters. A tiny weighted training objective is not pressure relative error or simulation drift.',
            'Label-coordinate reconstruction can affect precision near floating-point limits without introducing new physical data. The fixed basis inverse used offline is distinct from an analytical physical pressure inverse at inference.',
            'Joint mode excitation broadens linear component queries but retains only 16 physical scenes and four geometries. Preserve the offline numerical-label provenance; do not call these independent holdout scenes or a complete generator benchmark.'][sequence]
        row = dict(fit_item_id=fid,note_path=cards[fid]['card_path'],original_receipt=binding(receipt_path),history=binding(train/'history.json'),
            sources=sources,original_configuration=binding(project/'config.json'),original_recipe=binding(project/'train-recipe.json'),
            training_inputs=bound_inputs,training_manifest=binding(inputs['manifest']),retained_parameter_count=38673,active_nonDC_parameters=38669,
            actual_train_trajectories=16,training_native_pressure_pairs=144,additional_synthetic_queries=128 if mixed else 0,
            original_geometry_data=geometry_rows,parent_fit_item_id=parent_fit,optimizer_updates=updates,
            optimizer_closure_calls=None if initial_fit else history['coverage']['optimizer_closure_calls'],
            method=method,purpose=purpose,experiment=training+' Original source/recipe/input/output identities and all F64 initial/final tensor bits verify without neural execution.',
            outcome=f'The original fit completes its declared {updates} outer updates. Final logged training objective is {history["history"][-1]["objective"]:.17g}; it is specific to this objective and population. All initial/final parameters and inherited checkpoints verify, and the original non-DC label coverage is independently checked. Fresh common GPU pressure accuracy/timing, full-state compositions, independent holdout and long-horizon drift are not measured by this review.',
            lesson=lesson,reproduction='Retain the exact typed recipe, trainer/objective/basis/model/configuration/runtime, all four original coefficient shards, normalization and original native-source hashes. Refinements also require the full exact inherited F64 inventory transported in normalization; decoder labels require the declared fixed coordinate encoding, and mixed queries require their offline reference archive and seed. Original final publication and restore probes run CPU neural inference and need GPU-only qualification before rerunning. Do not reuse tiny component loss as a complete-step accuracy claim; clean retraining and new-Rheon regeneration remain unqualified.')
        raw = (PROJECT/row['note_path']).read_bytes()
        original = verified_review_receipt(row)
        revised,suffix = revised_card(raw,row)
        require(raw.count(original) == revised.count(original) == 1 and revised.endswith(suffix), 'Original receipt/configuration suffix changed')
        rows.append(row)
        notes.append(binding(PROJECT/row['note_path']))
    require(len({s['learned_modal_pressure.py'] for s in source_versions.values()}) == 1
            and len({s['modal_basis.py'] for s in source_versions.values()}) == 1,
            'Pressure model or fixed basis changed across the refinement sequence')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/n['path']) == n for n in notes)
            and 'torch' not in sys.modules, 'Read-only/no-neural invariant failed')
    value = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=4,
        original_note_bindings=notes,new_model_forwards=0,native_mutations=0,new_reference_generation=False,
        clean_reproduction_qualified=False,producer=binding(Path(__file__)),helpers=[binding(Path(__file__).with_name(n)) for n in
            ['prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(value)+b'\n')
    print('Four geometry-specific F64 modal components reviewed; coefficient labels/parameters verified; no learned gain execution.')


if __name__ == '__main__':
    run()
