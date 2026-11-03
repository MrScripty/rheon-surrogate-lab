"""Review the two original fixed-grid pressure fits from exact retained bytes."""
from collections import Counter
import hashlib
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
OUTPUT = DOC / 'reports/captured-bootstrap-pressure-component-method-reviews-01.json'
SPECS = [('fit-0b2b6b6a8c42a47a','local-pressure-v1-workspace-5'),
         ('fit-4ca6c279be34e629','local-pressure-v1-fresh-bootstrap-2')]


def run():
    require(not OUTPUT.exists(), 'Preserve original bootstrap source review')
    index_path = PROJECT / 'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    cards = {c['item_id']:c for c in read(index_path)['fits']}
    rows, notes, originals, initial_inventories = [], [], [], []
    for fid, folder in SPECS:
        base = ROOT / 'runs' / folder
        project, train = base / 'project', base / 'execution/reopened-stage-00'
        receipt_path = train / 'receipt.json'
        receipt, history = read(receipt_path), read(train / 'history.json')
        config, witness = read(project / 'config.json'), read(project / 'source-witness.json')
        require(cards[fid]['receipt_sha256'] == binding(receipt_path)['sha256']
                and not cards[fid].get('method_review_status') and receipt['status'] == 'complete'
                and receipt['mode'] == 'train' and receipt['optimizer_updates'] == receipt['candidate_update'] == 200,
                'Original completed unreviewed bootstrap fit required')
        require(config['training'] == dict(batch_size=2,checkpoint_interval=20,learning_rate=.001,
                    model='groupnorm-v1',seed=20261014,updates=200), 'Exact bootstrap optimizer config differs')
        for name, key in [('config.json','experiment_sha256'), ('source-witness.json','source_witness_sha256'),
                          ('runtime-manifest.json','runtime_manifest_sha256')]:
            require(binding(project / name)['sha256'] == receipt['closure'][key], 'Original bootstrap source/config/runtime differs')
        sources = [binding(project / name) for name in sorted(witness['sources'])]
        require(all(s['sha256'] == witness['sources'][Path(s['path']).name] for s in sources),
                'Complete original bootstrap source changed')
        document, recipe = read(project / 'project.json'), read(project / 'train-recipe.json')
        for spec in [*document['recipes'], *recipe['sources']]:
            require(binding(project / spec['path'])['sha256'] == spec['sha256'], 'Original project-relative recipe/source differs')
        require(recipe['entrypoint'] == 'train.py' and recipe['runtime'] == 'captured_cpu_python_v1',
                'Historical bootstrap runtime/entrypoint differs')
        inputs = {r['id']:project / r['path'] for r in document['data']}
        bound_inputs = []
        for slot, expected in receipt['closure']['inputs'].items():
            p = inputs[slot]
            b = binding(p)
            require(dict(sha256=b['sha256'], bytes=p.stat().st_size) == expected, 'Original bootstrap input differs')
            bound_inputs.append(dict(b, slot=slot, bytes=expected['bytes']))
        for name, expected in receipt['outputs'].items():
            p = train / name
            require(dict(sha256=binding(p)['sha256'], bytes=p.stat().st_size)
                    == {k:expected[k] for k in ('sha256','bytes')}, 'Original bootstrap output differs')
        split, origin = read(inputs['splits']), read(inputs['origin'])
        require(split['experiment'] == config and origin['experiment_sha256'] == receipt['closure']['experiment_sha256']
                and split['capture_manifest_sha256'] == binding(inputs['origin'])['sha256']
                and len(origin['trajectories']) == 16 and not origin['fixture']
                and origin['status'] == 'development_complete', 'Original native capture/population differs')
        residual_rows = split['native_residual_origin']['rows']
        train_rows, validation_rows, observed_residual_rows = [], [], []
        geometries = set()
        for slot, name in [('train0','train-0.npz'),('train1','train-1.npz'),('validation','validation.npz')]:
            shard = split['shards'][name]
            require(shard['sha256'] == binding(inputs[slot])['sha256'], 'Captured shard identity differs')
            records = shard['rows']
            (validation_rows if slot == 'validation' else train_rows).extend(records)
            with np.load(inputs[slot], allow_pickle=False) as arrays:
                require(set(arrays.files) == set(shard['arrays']), 'Exact bootstrap array membership required')
                for field, spec in shard['arrays'].items():
                    value = arrays[field]
                    require(value.dtype.str == spec['dtype'] and list(value.shape) == spec['shape']
                            and value.nbytes == spec['bytes'] and np.isfinite(value).all()
                            and hashlib.sha256(value.tobytes()).hexdigest() == spec['sha256'],
                            'Exact bootstrap array bytes/type/shape differ')
                require(arrays['reference'].shape[1:] == (16,16,16) and np.all(arrays['rho'] == 1.25)
                        and np.all(arrays['dt'] == 1/60)
                        and np.array_equal(arrays['spacing'], arrays['extents']/16), 'Original anisotropic physical units differ')
                for i, row in enumerate(records):
                    geometries.add(tuple(float(x) for x in arrays['spacing'][i]))
                    require(row['frame'] == int(arrays['frame'][i]), 'Original physical row clock differs')
                    for field in ['base','rhs','reference']:
                        require(hashlib.sha256(arrays[field][i].tobytes()).hexdigest() == row['lineage'][field]['sha256'],
                                'Original per-frame native pressure/RHS lineage differs')
                    observed_residual_rows.append(dict(row_id=row['row_id'],
                        sha256=hashlib.sha256(arrays['native_input_residual'][i].tobytes()).hexdigest()))
        require(observed_residual_rows == residual_rows and len(train_rows) == 288 and len(validation_rows) == 96
                and len({r['trajectory_id'] for r in train_rows}) == 12
                and len({r['trajectory_id'] for r in validation_rows}) == 4
                and not ({r['trajectory_id'] for r in train_rows} & {r['trajectory_id'] for r in validation_rows}),
                'Complete native residual/whole-parent split differs')
        groups = {}
        for i, row in enumerate(train_rows):
            groups.setdefault(row['trajectory_id'], []).append(i)
        groups = list(groups.values())
        rng = np.random.Generator(np.random.PCG64(20261015))
        order = []
        for _ in range(200):
            batch = []
            for _ in range(2):
                parent = groups[int(rng.integers(12))]
                batch.append(parent[int(rng.integers(24))])
            order.append(batch)
        require(hashlib.sha256(np.asarray(order,dtype='<i8').tobytes()).hexdigest()
                == receipt['closure']['sample_order_sha256'], 'Independent bootstrap sample order differs')
        for name, spec in history['checkpoints'].items():
            require(sum(np.prod(v['shape']) for v in spec['inventory'].values()) == 372377
                    and len(spec['inventory']) == 38, 'Actual bootstrap parameter inventory differs')
            check_tensors(train / name, spec['inventory'])
        initial_inventories.append(history['checkpoints']['checkpoint-000.npz']['inventory'])
        require(receipt['outputs']['weights.npz']['sha256'] == receipt['outputs']['checkpoint-200.npz']['sha256']
                and binding(train / 'normalization.json')['sha256'] == receipt['closure']['normalization_sha256'],
                'Final-update selection/train-only normalization differs')
        code, gn = (project / 'original_model.py').read_text(), (project / 'gn_model.py').read_text()
        require('"hx", "hy", "hz", "rho", "dt"' in code
                and 'nn.GroupNorm(4, first.out_channels, eps=1e-5, affine=False)' in gn
                and 'spacing.device.type == rho.device.type == "cpu"' in code, 'Original architecture/CPU-only operator contract differs')
        row = dict(fit_item_id=fid, note_path=cards[fid]['card_path'], original_receipt=binding(receipt_path),
            history=binding(train / 'history.json'), sources=sources, source_witness=binding(project / 'source-witness.json'),
            original_configuration=binding(project / 'config.json'), original_recipe=binding(project / 'train-recipe.json'),
            training_inputs=bound_inputs, training_population=binding(train / 'splits.json'),
            parameters=372377, initialized_tensor_count=38, training_trajectories=12, validation_trajectories=4,
            training_frames=288, validation_frames=96, grid_counts_xyz=[16,16,16],
            spacing_xyz_m=[list(g) for g in sorted(geometries)], fluid_density_kg_m3=1.25,
            original_native_binary_sha256=split['native_residual_origin']['native_build_receipt']['binary_sha256'],
            method='A 372,377-parameter, 38-tensor F32 multiscale pressure CNN on a fixed 16³ grid. Ten inputs are native residual, previous pressure, physical XYZ coordinates, separate hx/hy/hz spacing maps, fluid density and dt. Four encoder levels, bottleneck/global-mean path and skip decoder use SiLU with non-affine four-group normalization inserted before each block activation. The additive pressure correction is rescaled and cell-zero gauged in F64. The physical operator supports anisotropic spacing with area/(rho*h) face weights. This produces pressure only, not a full simulation state.',
            purpose=('Qualify configured training, observational checkpoint capture and exact typed restoration of the original GroupNorm pressure architecture on measured multi-scene native data.' if len(rows) == 0 else
                     'Repeat the same configured training/restore procedure on a separately captured seed cohort, exercising capture/bootstrap reuse while retaining the original architecture and initialization.'),
            experiment='Two hundred historical CPU Adam updates, batch two, rate 0.001, seed 20261014, PCG64 sampling seed 20261015 and clipping. Twelve training trajectories × 24 frames and four validation trajectories × 24 frames use whole-parent splits. All grids have 16 cells per axis; physical extents vary over four shapes, including anisotropic cell spacing. Fluid density is 1.25 kg/m³ and dt=1/60 s throughout. Forcing includes steady 0.5 and moving 1.0 m/s² scenes. Eleven checkpoints at updates 0,20,…,200 are retained; only the final update is the candidate, with no optimizer-resume state. All original input arrays and per-frame pressure/RHS/residual hashes verify in place.',
            outcome=f'Original final sampled training loss {history["history"][-1]["loss"]:.17g}; original validation objective changes from {history["validation"]["initial"]:.17g} to {history["validation"]["final"]:.17g}. These normalized pressure/equation objectives are not worst pressure relative L2 and cannot be compared directly with the near-exact F64 anchors. No current common GPU benchmark, complete-step time or trajectory drift is available for these two original contracts.',
            lesson='Fixed grid counts do not imply fixed physical resolution: these scenes change physical extent and separate per-axis spacing, unlike later cubic-cell water models. The fluid density is also different from the later 1,000 kg/m³ convention. Preserve those operator/normalization contracts and original native identity. Matching initialized tensors and sample indices across cohorts do not make their physical field data identical or establish independent generalization.',
            reproduction='Use the exact project-relative train/restore recipes, witnessed flat source, original runtime, all five typed input slots, train-only normalization and every observational checkpoint. Native capture and residual lineage are retained down to each row. The old model and physical checks explicitly require CPU; qualify a GPU-only execution/physical-operator variant before any new forward under current policy. Clean retraining, new-Rheon regeneration and common pressure adapters remain unqualified. Original receipts/configuration remain unchanged.')
        raw = (PROJECT / row['note_path']).read_bytes()
        original = verified_review_receipt(row)
        revised, suffix = revised_card(raw, row)
        require(raw.count(original) == revised.count(original) == 1 and revised.endswith(suffix), 'Original bootstrap card facts changed')
        rows.append(row)
        notes.append(binding(PROJECT / row['note_path']))
        originals.append(receipt)
    require(initial_inventories[0] == initial_inventories[1]
            and originals[0]['closure']['sample_order_sha256'] == originals[1]['closure']['sample_order_sha256']
            and originals[0]['closure']['inputs']['train0'] != originals[1]['closure']['inputs']['train0'],
            'Seeded repeat must share initialization/order while retaining different data')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT / n['path']) == n for n in notes)
            and 'torch' not in sys.modules, 'Read-only/no-neural review invariant failed')
    result = dict(schema='rheon.captured-model-method-reviews.v1', created_unix=time.time(), rows=rows,
        source_reviewed_fits=2, original_note_bindings=notes, matching_initial_inventory_and_order=True,
        different_physical_data=True, all_original_input_arrays_verified=True,
        new_model_forwards=0, native_mutations=0, new_reference_generation=False, clean_reproduction_qualified=False,
        producer=binding(Path(__file__)), helpers=[binding(Path(__file__).with_name(n)) for n in
            ['captured_replay_pressure_review.py','prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(result) + b'\n')
    print('Two bootstrap pressure fits reviewed; anisotropic spacing, native field lineage and seeded repetition verified.')


if __name__ == '__main__':
    run()
