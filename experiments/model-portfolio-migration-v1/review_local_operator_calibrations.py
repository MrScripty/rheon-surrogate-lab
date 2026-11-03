"""Review learned point and viscosity calibrations without executing either model."""
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
BASE = ROOT/'runs/full-ai-next-priorities-v1'
OUTPUT = DOC/'reports/captured-local-operator-calibration-method-reviews-01.json'
SPECS = [
    ('fit-d106ea978eb8ed52','point-sampling-project-01','point-sampling-pair-01','point_sampling',12,.5,2026102041,2026102042,'point-weights.npz'),
    ('fit-d1c3dbdd9793117a','viscosity-calibration-project-01','viscosity-calibration-pair-01','viscosity_calibration',3,.25,2026102061,2026102062,'viscosity-weights.npz')]


def run():
    require(not OUTPUT.exists(), 'Preserve original local-operator reviews')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    cards = {r['item_id']:r for r in read(index_path)['fits']}
    rows, notes = [], []
    for fid,folder,pair,module,channels,rate,seed,sample_seed,export in SPECS:
        project,train = BASE/folder,BASE/pair/'train'
        receipt_path = train/'receipt.json'
        receipt,config,history = read(receipt_path),read(project/'config.json'),read(train/'history.json')
        require(receipt['status'] == 'complete' and receipt['mode'] == 'train'
                and receipt['optimizer_updates'] == config['updates'] == 1000
                and config['batch_size'] == 8 and config['learning_rate'] == rate
                and config['seed'] == seed and config['sampling_seed'] == sample_seed
                and not receipt['full_state_forecast'] and not receipt['native_sampling_at_candidate_inference']
                and history['physical_states_predicted'] == 0 and not history['temporal_scenes_used'],
                'Original calibration role/budget differs')
        require(binding(project/'config.json')['sha256'] == receipt['closure']['config_sha256'], 'Original config changed')
        sources = [binding(project/n) for n in sorted(receipt['closure']['sources'])]
        require(all(s['sha256'] == receipt['closure']['sources'][Path(s['path']).name] for s in sources), 'Captured source closure changed')
        document,recipe = read(project/'project.json'),read(project/'train-recipe.json')
        require(recipe['entrypoint'] == str((project/(module+'_train.py')).relative_to(ROOT)), 'Actual calibration trainer differs')
        for spec in [*document['recipes'],*recipe['sources']]:
            require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Typed recipe/source changed')
        inputs = {r['id']:ROOT/r['path'] for r in document['data']}
        bound_inputs = []
        for slot,expected in receipt['closure']['inputs'].items():
            spec = binding(inputs[slot])
            require(dict(sha256=spec['sha256'],bytes=inputs[slot].stat().st_size) == expected, 'Teacher/runtime input changed')
            bound_inputs.append(dict(spec,slot=slot,bytes=expected['bytes']))
        manifest = read(inputs['teacher_manifest'])
        require(manifest['status'] == 'native_qualified' and manifest['train_indices'] == list(range(64))
                and manifest['development_indices'] == list(range(64,80)) and not manifest['temporal_scenes_used']
                and not manifest['reserved_scenes_consumed'] and manifest['archive_sha256'] == binding(inputs['teacher'])['sha256'],
                'Original synthetic teacher split/provenance differs')
        with np.load(inputs['teacher'],allow_pickle=False) as arrays:
            require(set(arrays.files) == set(manifest['inventory']) == {'velocity','target','mask'}, 'Complete teacher array inventory required')
            for name,spec in manifest['inventory'].items():
                value = arrays[name]
                require(value.dtype.str == spec['dtype'] and list(value.shape) == spec['shape']
                        and np.isfinite(value).all() and hashlib.sha256(value.tobytes()).hexdigest() == spec['sha256'],
                        'Original teacher array bytes/dtype/shape differ')
            require(arrays['velocity'].shape == (80,3,13,13,13) and arrays['target'].shape == (80,channels,13,13,13)
                    and arrays['mask'].shape == (1,channels,13,13,13)
                    and set(np.unique(arrays['mask'])).issubset({0.,1.}) and np.count_nonzero(arrays['mask']) > 0,
                    'Actual native lattice/label mask differs')
        for name,expected in receipt['outputs'].items():
            p = train/name
            require(dict(sha256=binding(p)['sha256'],bytes=p.stat().st_size) == expected, 'Original calibration output changed')
        require(cards[fid]['receipt_sha256'] == binding(receipt_path)['sha256'] and not cards[fid].get('method_review_status'),
                'Card belongs to another/already-reviewed fit')
        count = channels*3*3**3
        require(receipt['parameters'] == receipt['trainable_parameters'] == count, 'Independent local kernel parameter count differs')
        for filename,key in [('initial-weights.npz','initial_inventory'),('weights.npz','final_inventory')]:
            inventory = receipt[key]
            require(inventory == history[key] and set(inventory) == {'projection.weight'}, 'Single projection tensor required')
            with np.load(train/filename,allow_pickle=False) as arrays:
                require(arrays.files == ['projection.weight'], 'Exact projection archive required')
                value = arrays['projection.weight']
                require(value.dtype.str == inventory['projection.weight']['dtype'] == '<f4'
                        and list(value.shape) == inventory['projection.weight']['shape'] == [channels,3,3,3,3]
                        and hashlib.sha256(value.tobytes()).hexdigest() == inventory['projection.weight']['sha256'],
                        'Retained F32 projection differs')
                if key == 'initial_inventory':
                    require(np.count_nonzero(value) == 0, 'Calibration must start from zero coefficients')
        rng = np.random.Generator(np.random.PCG64(sample_seed))
        order = np.asarray([rng.choice(64,size=8,replace=False) for _ in range(1000)],dtype='<i8')
        require(hashlib.sha256(order.tobytes()).hexdigest() == history['sampled_pattern_order_sha256']
                and history['optimizer'] == 'SGDfloat64' and history['learning_rate'] == rate
                and history['coefficient_accuracy_pass'], 'Original private sample/optimizer/diagnostic differs')
        # A later checkpoint's archive encoding can differ. Compare exact tensor
        # bytes rather than pretending unequal NPZ hashes mean changed weights.
        deployed = BASE/'b2-fixed-viscosity-project-01/expert-assets'/export
        with np.load(deployed,allow_pickle=False) as arrays:
            require(hashlib.sha256(arrays['projection.weight'].tobytes()).hexdigest() == receipt['final_inventory']['projection.weight']['sha256'],
                    'Later generator uses different point/viscosity coefficient values')
        trainer = (project/(module+'_train.py')).read_text()
        model = (project/(module+'_model.py')).read_text()
        require('torch.optim.SGD(network.parameters(),lr=config[\'learning_rate\'])' in trainer
                and 'nn.Conv3d(3' in model and 'bias=False' in model
                and 'nn.init.zeros_(self.projection.weight)' in model, 'Actual linear zero-initialized calibration differs')
        point = channels == 12
        row = dict(fit_item_id=fid,note_path=cards[fid]['card_path'],original_receipt=binding(receipt_path),history=binding(train/'history.json'),
            sources=sources,original_configuration=binding(project/'config.json'),original_recipe=binding(project/'train-recipe.json'),
            training_inputs=bound_inputs,teacher_manifest=binding(inputs['teacher_manifest']),retained_parameter_count=count,
            actual_training_patterns=64,development_patterns=16,temporal_scenes_used=False,optimizer_updates=1000,
            sampled_pattern_order_independently_verified=True,later_generator_asset=binding(deployed),later_asset_tensor_values_equal=True,
            method=(f'A single bias-free 3×3×3 convolution maps three velocity channels to 12 sampled point-velocity channels with {count} learned coefficients. The channels encode four destination kinds (three velocity-face types and tracer-cell points), each with xyz components. It learns negative sampled velocity; it does not itself backtrace a particle, advance a carrier, solve pressure or generate a simulation state.' if point else
                f'A single bias-free 3×3×3 convolution maps three velocity channels to three dimensionless viscous-stiffness channels with {count} learned coefficients. Targets represent the native negative symmetric-strain stiffness on free face degrees of freedom. Physical viscosity/timestep/spacing scaling and state advancement belong to the composing generator.'),
            purpose=('Replace a repeated local point-velocity sampling operation with a learned linear projection whose coefficients can be reused in displacement-based generators.' if point else
                'Calibrate the local native viscosity operator as a learned projection and freeze those coefficients in later complete generators, separating operator calibration from temporal fitting.'),
            experiment=f'Use 80 manufactured velocity patterns on the padded 13³ lattice: 64 fitting patterns and 16 development patterns, with a native-qualified teacher and explicit valid-output mask. No temporal simulation scenes or reserved scenes enter training. Zero coefficients are optimized for 1,000 CUDA SGD updates in F64, batch eight distinct fitting patterns per update, rate {rate}. PCG64 sampling uses seed {sample_seed}; its full order is independently reproduced. Training then publishes F32 coefficients. All source/recipe/teacher/output identities and complete initial/final tensors verify. The later generator asset has identical tensor bytes even though its NPZ archive encoding differs.',
            outcome=f'Original calibration passes its declared diagnostic thresholds. Recorded worst development relative error is {history["worst_development_relative_error_float64"]:.17g} in the training F64 execution and {history["worst_development_relative_error_published_float32"]:.17g} in the published F32 execution. These are retained original component diagnostics, not new forwards or complete-state accuracy. No new GPU timing or trajectory drift is measured here.',
            lesson='A tiny operator calibration error on manufactured patterns does not certify general fluid-scene accuracy or accumulated drift. Keep training F64 and published F32 arithmetic separate. A later frozen composition inherits the coefficient values and calibration cost, not a full-state training result. Equal tensors in differently encoded NPZ archives are a value-equivalence observation, not identical artifact bytes.',
            reproduction='Use the exact captured recipe/config/model/trainer/runtime, native-qualified teacher archive/manifest and its masks/split, zero initialization, private PCG64 order and F64 SGD followed by F32 publication. Preserve the original native build and qualification provenance for old-reference reproduction. The old trainer evaluates the published model and final probe on CPU; qualify GPU-only publication/restore before rerunning. Clean retraining, new-Rheon teacher regeneration and complete-generator validation remain unqualified.')
        raw = (PROJECT/row['note_path']).read_bytes()
        original = verified_review_receipt(row)
        revised,suffix = revised_card(raw,row)
        require(raw.count(original) == revised.count(original) == 1 and revised.endswith(suffix), 'Original receipt/config suffix changed')
        rows.append(row)
        notes.append(binding(PROJECT/row['note_path']))
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/n['path']) == n for n in notes)
            and 'torch' not in sys.modules, 'Read-only/no-neural invariant failed')
    value = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=2,
        original_note_bindings=notes,new_model_forwards=0,native_mutations=0,new_reference_generation=False,
        clean_reproduction_qualified=False,producer=binding(Path(__file__)),helpers=[binding(Path(__file__).with_name(n)) for n in
            ['prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(value)+b'\n')
    print('Point/viscosity calibrations reviewed; all teacher arrays, PCG64 order and inherited tensor values verified.')


if __name__ == '__main__':
    run()
