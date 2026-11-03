"""Review K2/3/4 endpoint versus retained-feature generators, without Torch."""
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
OUTPUT = DOC/'reports/captured-sharing-temporal-generator-method-reviews-01.json'


def run():
    require(not OUTPUT.exists(), 'Preserve original sharing-generator review')
    availability_path = DOC/'reports/full-generator-review-input-availability-01.json'
    candidates = [r for r in read(availability_path)['rows'] if r['original_configuration'].get('schema') == 'rheon-temporal-fit-sharing-v1']
    candidates.sort(key=lambda r:(r['declared_horizons'],r['original_configuration']['architecture']))
    require(len(candidates) == 6, 'All three original endpoint/reuse pairs required')
    prior_path = DOC/'reports/captured-joint-temporal-generator-method-reviews-01.json'
    prior = read(prior_path)
    original = prior['rows'][0]
    corpus_inputs = {v['slot']:{k:v[k] for k in ('sha256','bytes')} for v in original['training_inputs']}
    for spec in [prior['producer'],*prior['helpers']]:
        require(binding(ROOT/spec['path']) == spec, 'Prior original-corpus audit source changed')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    cards = {f['item_id']:f for f in read(index_path)['fits']}
    # Same immutable input used by six fits: hash each exact path once in this
    # serial read-only audit; retain separate original slot identities per fit.
    observed_inputs = {}
    source_sets, shared_initial, sample_orders = [], {}, []
    rows, notes = [], []
    for item in candidates:
        fid = item['fit_item_id']
        project = ROOT/item['project_path']
        receipt_path = ROOT/item['receipt']['path']
        require(binding(receipt_path)['sha256'] == item['receipt']['sha256'], 'Original fit receipt changed')
        receipt, config = read(receipt_path), read(project/'config.json')
        training = receipt_path.parent
        history = read(training/'history.json')
        k, architecture = config['horizons'], config['architecture']
        require(k in (2,3,4) and architecture in ('endpoint','reuse') and config['width'] == 12
                and config['feedback_protocol'] == 2 and config['updates'] == receipt['optimizer_updates'] == 10000
                and receipt['status'] == 'complete' and receipt['mode'] == 'train'
                and config['single_block_updates'] == 2000 and config['sample_step_range'] == [0,120]
                and config['seed'] == 2026101921 and config['sampling_seed'] == 2026101922
                and config['learning_rate'] == .001
                and config['schedule'] == [dict(update=6001,learning_rate=.0003),dict(update=9001,learning_rate=.0001)],
                'Actual paired architecture/budget/curriculum differs')
        require(binding(project/'config.json')['sha256'] == receipt['closure']['config_sha256'], 'Original config differs')
        sources = [binding(project/name) for name in sorted(receipt['closure']['sources'])]
        require(all(s['sha256'] == receipt['closure']['sources'][Path(s['path']).name] for s in sources), 'Actual source closure differs')
        source_sets.append({Path(s['path']).name:s['sha256'] for s in sources})
        manifest_project, recipe = read(project/'project.json'), read(project/'train-recipe.json')
        require(recipe['entrypoint'] == str((project/'temporal_train_sharing.py').relative_to(ROOT)), 'Actual trainer differs')
        for spec in [*manifest_project['recipes'],*recipe['sources']]:
            require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Original typed recipe source differs')
        inputs = {entry['id']:ROOT/entry['path'] for entry in manifest_project['data']}
        require(receipt['closure']['inputs'] == corpus_inputs, 'Sharing fit changed original16scene corpus/runtime slots')
        bound_inputs = []
        for slot, expected in corpus_inputs.items():
            path = inputs[slot]
            if path not in observed_inputs:
                observed_inputs[path] = dict(sha256=binding(path)['sha256'],bytes=path.stat().st_size)
            require(observed_inputs[path] == expected, 'Current original trajectory/runtime input differs')
            bound_inputs.append(dict(binding(path) if slot in ('manifest','normalization') else
                dict(path=str(path.relative_to(ROOT)),sha256=expected['sha256']),slot=slot,bytes=expected['bytes']))
        require(binding(inputs['manifest']) == original['training_manifest'], 'Sharing corpus is not the audited full-trajectory manifest')
        manifest = read(inputs['manifest'])
        for name, expected in receipt['outputs'].items():
            path = training/name
            require(dict(sha256=binding(path)['sha256'],bytes=path.stat().st_size) == expected, 'Original fit output differs')
        inventories = {}
        for filename, key in [('initial-weights.npz','initial_inventory'),('weights.npz','final_inventory')]:
            inventory = receipt[key]
            require(inventory == history[key], 'Receipt/history tensor inventory differs')
            with np.load(training/filename,allow_pickle=False) as arrays:
                require(set(arrays.files) == set(inventory), 'Complete initial/final tensors required')
                for name, spec in inventory.items():
                    value = arrays[name]
                    require(value.dtype.str == spec['dtype'] == '<f4' and list(value.shape) == spec['shape']
                            and hashlib.sha256(value.tobytes()).hexdigest() == spec['sha256'], 'Initial/final parameter bytes differ')
                heads = ('endpoint_head','intermediate_head') if architecture == 'endpoint' else ('first_head','later_head')
                if key == 'initial_inventory':
                    require(all(np.count_nonzero(arrays[f'{head}.{part}']) == 0 for head in heads for part in ('weight','bias')),
                            'Initial output heads were not zero')
            inventories[key] = inventory
        count = sum(math.prod(v['shape']) for v in inventories['final_inventory'].values())
        expected_count = 162764+152*k if architecture == 'endpoint' else 163068+36*k
        require(count == receipt['parameters'] == expected_count, 'Independent width12/GlobalMix parameter formula differs')
        core = {n:v for n,v in inventories['initial_inventory'].items() if n.startswith(('fine.','forcing.','local.','global_blocks.'))}
        if k in shared_initial:
            require(core == shared_initial[k], 'Same-K endpoint/reuse pair does not share exact seeded encoder/global tensors')
        else:
            shared_initial[k] = core
        rng = np.random.Generator(np.random.PCG64(config['sampling_seed']))
        order = np.asarray([(int(rng.integers(16)),int(rng.integers(121)),1 if u <= 2000 else 2) for u in range(1,10001)],dtype='<i8')
        require(hashlib.sha256(order.tobytes()).hexdigest() == history['sample_order_sha256'], 'Private sample/curriculum order differs')
        cells = [math.prod(n+1 for n in parent['scene']['counts']) for parent in manifest['parents']]
        work = dict(network_forwards=18000,encoder_cells=sum(cells[int(p)]*int(blocks) for p,_,blocks in order),predicted_physical_states=18000*k,
                    encoder_calls=18000,latent_updates=18000*(k-1) if architecture == 'reuse' else 0,
                    endpoint_predictions=18000 if architecture == 'endpoint' else 0,
                    intermediate_states=18000*(k-1) if architecture == 'endpoint' else 0)
        require(history['optimizer_work'] == work and history['valid_parent_count'] == 16 and history['train_only'], 'Actual optimizer forward/cell/state work differs')
        model = (project/'bundle_model_sharing.py').read_text()
        global_source = (project/'learned_global.py').read_text()
        trainer = (project/'temporal_train_sharing.py').read_text()
        require("fine=self.fine(torch.cat((state[:,1:],features[:,:10]),dim=1))" in model
                and 'self.intermediate(torch.cat((hidden,state[:,1:],endpoint[:,0],forces),dim=1))' in model
                and 'hidden=hidden+self.latent_update(torch.cat((hidden,previous,known),dim=1))' in model
                and 'generated=self.reconstruct(raw,state,counts)' in model
                and 'self.mixing=nn.Parameter(torch.randn(8,8,8,width,width)/math.sqrt(width))' in global_source
                and "BundleNet(config['horizons'],config['width'],config['architecture'])" in trainer,
                'Actual pressure-history exclusion/endpoint/retained-feature source route differs')
        card = cards[fid]
        require(not card.get('method_review_status'), 'Do not replace a published review')
        last = history['history'][-1]
        row = dict(fit_item_id=fid,note_path=card['card_path'],generator_identifier=config['identifier'],architecture=architecture,horizons=k,
            original_receipt=binding(receipt_path),history=binding(training/'history.json'),sources=sources,
            original_configuration=binding(project/'config.json'),original_recipe=binding(project/'train-recipe.json'),
            training_inputs=bound_inputs,training_manifest=binding(inputs['manifest']),retained_parameter_count=count,
            actual_train_trajectories=16,available_train_states=2064,physical_output_dt_s=1/60,block_physical_advance_s=k/60,
            optimizer_network_forwards=18000,optimizer_encoder_cells=work['encoder_cells'],optimizer_joint_outputs=18000*k,
            actual_optimizer_work=work,
            same_K_shared_seeded_core_verified=True,full_parameter_initialization_equal_across_K_not_claimed=True,
            sample_order_independently_rederived=True,fresh_common_full_generator_benchmark_complete=False,
            method=f'A {count:,}-parameter F32 width12 full-state generator for K={k}. Current velocity/tracer plus geometry/current known force enter one spatial encoder; previous pressure is deliberately excluded. All K prescribed forces enter retained features. Two learned global blocks mix up to8cosine modes per axis across12latent channels, with learned7→32→144mode/geometry-conditioned weights; these are neural mixing weights, not inverse pressure eigenvalues. '+('A fresh five-field endpoint head predicts the last state. One shared1x1decoder consumes that predicted endpoint, initial velocity/tracer, hidden features and all forces, then emits all K−1intermediate states jointly.' if architecture == 'endpoint' else 'A first head predicts the first state. Each later output updates retained latent features using its preceding predicted five-field state and current known force, then applies a shared later head; the spatial encoder/global blocks are not rerun.')+' Pressure is fresh raw output with cell-zero gauge; velocity/tracer increments always refer to the initial block state. Velocity walls are masked. Output packing upcasts pressure toF64, but prediction isF32. No numerical pressure solve or full physical integrator is called.',
            purpose=(f'Test the accurate-endpoint/cheap-intermediate architecture for {k} physical outputs per pass, with a learned predicted endpoint rather than a supplied future label.' if architecture == 'endpoint' else f'Test reusing spatial/global features for {k} outputs through smaller recurrent latent updates instead of repeating the main encoder.'),
            experiment='All six fits use the same captured16train scenes/2064states at8/16cells per metre and fixed1/60s, with16cubed,9x11x11,15x6x8 and32cubed domains. Original F32trajectory shards/normalization/runtime slots match the independently decoded joint-generator corpus. Each fit performs10,000CUDA Adam updates: batch1, model seed2026101921/sample seed2026101922, rate0.001 with drops at6001/9001. First2000updates supervise one block; remaining8000supervise a second block from reconstructed own endpoint, producing18,000network calls. Every horizon receives the original composite field/equation/divergence/mass/energy/bounds objective; no separate high-accuracy-endpoint guarantee is imposed. Same-K endpoint/reuse pairs share exact initial encoder/forcing/global tensors. ChangingKchanges parameter shapes and random-number consumption; full initialization equality acrossKis not claimed.',
            outcome=f'Original fit completed10,000updates; final logged composite objective {last["loss"]:.17g} with{last["blocks"]}supervised blocks. All original initial/final tensors, source/recipe/input/output identities and sample/work counts verify. No new complete-generator accuracy, GPU latency or own-history drift has been measured in this review. The composite loss cannot be read as pressure relative L2 or an inference-time score.',
            lesson=('This implements the proposed endpoint-conditioned intermediate strategy in the model itself. Its endpoint is predicted, so it receives no future-state leak. Shared global processing and a joint intermediate decoder can save repeated encoding, but equal-horizon training does not certify endpoint accuracy or a speed win. Keep this prototype available for common benchmarking rather than rejecting it from an old target or label.' if architecture == 'endpoint' else 'This implements retained-compute reuse with cheaper later latent updates. It remains sequential in latent predictions, but does not repeat the expensive spatial/global encoder per output. Later physical increments refer to the original block state; previous predicted state influences the latent update. Benchmark all intermediate fields and block/K; do not confuse this with Kfull integrator advances or endpoint interpolation.'),
            reproduction='Use exact original config/typed recipe/runtime/source closure, original16scene F32archive and normalization, zero output heads, private seeds/order and deployed reconstruction between supervised blocks. Preserve pressure-history exclusion, two learned global blocks and the selected endpoint/reuse branch. The older recipes perform CPU neural publication/restore probes; qualify GPU-only reproduction before rerunning. Clean retraining, new-Rheon data generation, fresh comparable performance and600-output drift remain unqualified.')
        raw = (PROJECT/card['card_path']).read_bytes()
        original_receipt = verified_review_receipt(row)
        revised, suffix = revised_card(raw,row)
        require(raw.count(original_receipt) == revised.count(original_receipt) == 1 and revised.endswith(suffix), 'Original receipt/configuration suffix changed')
        rows.append(row)
        notes.append(binding(PROJECT/card['card_path']))
        sample_orders.append(order)
    require(all(s == source_sets[0] for s in source_sets[1:]) and all(np.array_equal(o,sample_orders[0]) for o in sample_orders[1:]),
            'Endpoint/reuse family changes source implementation or sample/curriculum stream')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/n['path']) == n for n in notes), 'Current notes/index changed during source review')
    require('torch' not in sys.modules, 'Review imported a neural runtime')
    result = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=6,
        new_model_forwards=0,native_mutations=0,new_reference_generation=False,clean_reproduction_qualified=False,
        original_note_bindings=notes,prior_corpus_audit=binding(prior_path),availability=binding(availability_path),
        overview_addition='\n\n## Endpoint and retained-feature full-state prototypes\n\nSix original K2/3/4fits implement endpoint-conditioned intermediate decoding or retained-feature latent updates. They exclude previous pressure from the main encoder and share learned global modal processing. Same-K pairs preserve exact seeded core tensors and all share the original16scene/sample curriculum. These architectures do save repeated encoder work structurally, but fresh complete-generator GPU accuracy/timing and own-history tests remain required. A model name or endpoint design does not establish an accurate endpoint or target performance.\n',
        producer=binding(Path(__file__)),helpers=[binding(ROOT/'experiments/model-portfolio-migration-v1'/n) for n in
            ['prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(result)+b'\n')
    print('Six endpoint/reuse full-state fits reviewed; paired core tensors and original evidence verified; no neural execution.')


if __name__ == '__main__':
    run()
