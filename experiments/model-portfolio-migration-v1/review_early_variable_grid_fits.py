"""Review C03/C04/C05 from captured sources/data, without importing a model.

Parameter arrays and PCG64 sample order are read/derived with NumPy only. The
original CPU recipes and CUDA qualification are historical evidence; no forward,
optimizer, native simulation, clean retrain or project mutation is executed.
"""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from prepare_method_review_revisions import verified_review_receipt,revised_card

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT=ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT=DOC/'reports/captured-early-variable-grid-method-reviews-01.json'
SPECS=(('C03','runs/variable-pressure-v1-workspace-1','execution/reopened-stage-00',
        'fit-0a492f0abed283ef',10,372377,200),
       ('C04','runs/variable-pressure-v2-workspace-1','execution/stage-00/artifacts',
        'fit-9a9e8bd97302e0a5',7,371729,200),
       ('C05','runs/cuda-pressure-v1-workspace-2','execution/stage-00/artifacts',
        'fit-3955a792ed6c415a',7,371729,20))


def read(path):
    return json.loads(path.read_bytes())


def binding(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def require(value,message):
    if not value:
        raise ValueError(message)


def literal(path,name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Required captured literal missing: '+name)


def run():
    require(not OUTPUT.exists(),'Preserve the original source-review report')
    index_path=PROJECT/'notes/portfolio-index.json';index_raw=index_path.read_bytes()
    cards={r['item_id']:r for r in json.loads(index_raw)['fits']}
    scores_path=DOC/'reports/pressure-common-score-01.json';scores=read(scores_path)['models']
    rows=[];initial_inventories={};first_weights={};orders={};notes={}
    for label,folder,stage,fit,channels,parameters,updates in SPECS:
        project=ROOT/folder/'project';training=ROOT/folder/stage
        receipt_path=training/'receipt.json';receipt=read(receipt_path)
        history_path=training/'history.json';history=read(history_path)
        witness_path=project/'source-witness.json';witness=read(witness_path)
        config_path=project/'config.json';config=read(config_path)
        require(receipt['mode']=='train' and receipt['status']=='complete' and receipt['optimizer_updates']==updates
            and config['training']['updates']==updates and config['training']['batch_size']==2
            and config['training']['learning_rate']==.001 and config['training']['seed']==20261015,
            'Original fit budget/optimizer configuration differs')
        require(binding(config_path)['sha256']==receipt['closure']['experiment_sha256']
            and binding(witness_path)['sha256']==receipt['closure']['source_witness_sha256']
            and binding(project/'runtime-manifest.json')['sha256']==receipt['closure']['runtime_manifest_sha256'],
            'Captured configuration/source/runtime closure differs')
        sources=[binding(project/name) for name in sorted(witness['sources'])]
        require(all(s['sha256']==witness['sources'][Path(s['path']).name] for s in sources),
            'Original captured implementation changed')
        manifest=read(project/'project.json');recipe=read(project/'train-recipe.json')
        for entry in [*manifest['recipes'],*recipe['sources']]:
            require(binding(ROOT/entry['path'])['sha256']==entry['sha256'],'Typed recipe source changed')
        require(recipe['entrypoint']==str((project/'train.py').relative_to(ROOT))
            and recipe['arguments'][0]==dict(literal='train'),'Original training entrypoint differs')
        inputs=[]
        for entry in manifest['data']:
            path=ROOT/entry['path'];observed=binding(path)
            require(dict(sha256=observed['sha256'],bytes=path.stat().st_size)==receipt['closure']['inputs'][entry['id']],
                    'Original data input differs')
            inputs.append(dict(observed,slot=entry['id'],bytes=path.stat().st_size))
        require({s['slot'] for s in inputs}=={'dataset','splits'},'Exactly the original two input slots required')
        for name in ('checkpoint-000.npz','weights.npz','normalization.json','history.json','splits.json'):
            path=training/name;observed=binding(path)
            require(dict(sha256=observed['sha256'],bytes=path.stat().st_size)==
                    {k:receipt['outputs'][name][k] for k in ('sha256','bytes')},'Retained original output changed')
        require(receipt['outputs']['weights.npz']['sha256']==scores[label]['weights_sha256'],
                'Named checkpoint differs from the selected final fit')
        population_path=training/'splits.json';population=read(population_path)
        counts=Counter(p['entry']['split'] for p in population['parents'])
        require(counts==dict(train=4,validation=2) and all(len(p['frames'])==24 for p in population['parents']),
                'Six-trajectory physical training/validation population differs')
        geometries=[p['counts_xyz'] for p in population['parents']]
        require(geometries==[[8,8,8],[9,7,5],[12,8,6],[16,16,16],[17,13,9],[24,16,12]],
                'Original variable-grid physical dimensions differ')
        require([c['cells_per_metre'] for c in config.get('captures',population['experiment']['captures'])]==[8,16],
                'Original two physical grid densities differ')
        require(receipt['architecture']['input_channels']==channels
                and receipt['architecture']['block_normalization']==dict(affine=False,epsilon=1e-5,groups=4,
                    name='GroupNorm',position='each block convolution before SiLU'),
                'Captured actual architecture differs')
        inventory=history['checkpoints']['checkpoint-000.npz']['inventory']
        require(len(inventory)==38 and sum(math.prod(v['shape']) for v in inventory.values())==parameters,
                'Actual initialized tensor/parameter inventory differs')
        initial_inventories[label]=inventory
        with np.load(training/'checkpoint-000.npz',allow_pickle=False) as archive:
            first_weights[label]=archive['encoders.0.layers.0.weight'].copy()
        if channels==7:
            require(literal(project/'model.py','CHANNELS')==['native_residual','previous_pressure','x','y','z','rho','dt']
                and literal(project/'model.py','RETAINED_CHANNELS')==[0,1,2,3,4,8,9],
                'Seven-channel physical/input initialization contract differs')
        train_source=(project/'train.py').read_text()
        require('torch.optim.Adam(network.parameters()' in train_source
                and 'clip_grad_norm_' in train_source,'Original whole-network optimizer/gradient control differs')
        seed=config['training'].get('sampling_seed',config['training']['seed']+1)
        rng=np.random.Generator(np.random.PCG64(seed));eligible=[0,1,3,4]
        order=np.asarray([[(eligible[int(rng.integers(4))],int(rng.integers(24))) for _ in range(2)]
                          for _ in range(updates)],dtype='<i8')
        require(hashlib.sha256(order.tobytes()).hexdigest()==receipt['closure']['sample_order_sha256'],
                'Independently derived original sample order differs')
        orders[label]=order
        card=cards[fit];note_path=PROJECT/card['card_path'];raw=note_path.read_bytes();notes[note_path]=hashlib.sha256(raw).hexdigest()
        source_evidence=dict(original_receipt=binding(receipt_path),history=binding(history_path),sources=sources,
            source_witness=binding(witness_path),training_population=binding(population_path),training_inputs=inputs)
        original=verified_review_receipt(source_evidence)
        require(raw.count(original)==1,'Native card must retain the exact original receipt once')
        last=history['history'][-1];native_build=population['native_build_receipt']['binary_sha256']
        channel_text=('ten inputs: residual, previous pressure, physical XYZ coordinates, three cubic-spacing maps, rho and dt'
                      if channels==10 else 'seven inputs: residual, previous pressure, physical XYZ coordinates, rho and dt; spacing also affects the coordinates and remains scalar FV metadata')
        method=(f'An additive pressure-correction CNN with {parameters:,} float32 parameters in 38 tensors. '
            'Four encoder widths 8/16/24/32, bottleneck 48 and decoder widths 32/24/16/8; two 3x3x3 convolutions per block, '
            'affine-free four-group GroupNorm and SiLU, a bottleneck global-mean broadcast and nearest-to-skip upsampling. '
            f'The network consumes {channel_text}. Its linear 1x1x1 head is scaled to physical pressure, gauged by the '
            'first correction cell and added to the previous pressure. FV arithmetic and the pressure output remain float64. '
            'All network parameters are optimized. The captured input budget is 65,536 cells. This operation proposes '
            'pressure for one request; it does not advance the physical simulation.')
        experiment=(f'{updates} Adam updates at learning rate 0.001, batch size 2 and seed 20261015, with gradient clipping. '
            'Four training trajectories supply 96 pressure frames; two validation trajectories supply 48 frames. '
            'The shared 9,296,002-byte dataset covers 8 and 16 cells per metre. Training grids are 8x8x8, 9x7x5, '
            '16x16x16 and 17x13x9; validation grids are 12x8x6 and 24x16x12. The recorded forcing labels are steady '
            'and moving, with amplitude 0.5 m/s2 and a 1/60 s timestep. The objective equally weights normalized squared '
            'pressure-field and FV equation-residual errors. Train-only scales and the exact PCG64 draw sequence are bound. '
            'The recipe supplies a final-update candidate and does not retain an optimizer-resume state.')
        common=scores[label]['worst']
        outcome=(f'The original final logged training loss is {last["loss"]:.17g} at update {updates}. '
            f'The existing 128-request GPU comparison gives worst mean-free pressure relative L2 '
            f'{common["pressure_mean_free_relative_l2"]:.17g} and equation relative L2 '
            f'{common["common_equation_relative_l2"]:.17g}. The training objective and common field errors are different '
            'quantities. This source review adds no inference timing or own-history drift measurement.')
        if label=='C03':
            purpose='Test one pressure-correction CNN across rectangular cubic-cell grids and two physical spacings, using spatial and physical metadata channels.'
            lesson='This establishes an early variable-grid baseline with a small physical scene population and substantial common-request pressure/equation error. Shape support and metadata channels alone do not establish broad accuracy. The timestep channel was trained at one physical timestep, so wider timestep generalization remains unqualified.'
        elif label=='C04':
            purpose='Remove three redundant cubic-spacing network channels, preserving scalar spacing in the FV operator, metre-valued coordinates and the surviving seeded coefficients.'
            lesson='The dataset bytes and independently derived 200-update sample sequence are identical to C03. Removing 648 first-layer coefficients does not materially improve the common worst errors. Record the seed agreement separately from trained ancestry. Split metadata reports a different native-build witness; preserve that identity without claiming a fresh independent capture or larger training population.'
        else:
            purpose='Qualify CUDA training and typed restoration of the seven-channel CNN through a deliberately short 20-update fit.'
            lesson='This is CUDA enablement with a short fitting budget. It shares C04 initial tensors, dataset and the first 20 optimizer sample draws. The original recipe verifies CUDA gradients and Adam moments, then transfers the final model to CPU for deployment/parity inference. Its recorded maximum CPU/CUDA pressure difference is 9.715233779570773e-09 on that historical population. Different final update budgets prevent a matched-budget training comparison.'
        reproduction=('Use the captured project manifest and training recipe, exact source witness, config, runtime, normalization and two typed data slots. Preserve the initial seed inventory and final weight hash. '
            f'The population metadata retains native binary identity {native_build}. Its six capture specifications '
            'and frame hashes describe the original data. Clean training/data regeneration and conversion to the new Rheon '
            'have not been executed or qualified. '
            +('The original CPU-only contract remains historical evidence; future execution requires a qualified GPU route under the current no-CPU-neural-forward policy.' if label!='C05' else
              'The original CUDA recipe ends with CPU deployment/parity inference. Do not execute it unchanged under the current no-CPU-neural-forward policy; a clean GPU-only reproducer remains to be qualified.'))
        review=dict(fit_item_id=fit,note_path=card['card_path'],method=method,purpose=purpose,experiment=experiment,
            outcome=outcome,lesson=lesson,reproduction=reproduction,**source_evidence,
            original_configuration=binding(config_path),original_recipe=binding(project/'train-recipe.json'),
            architecture_parameters=parameters,trainable_parameter_inventory=parameters,initialized_tensor_count=38,
            physical_training_frames=96,physical_validation_frames=48,native_build_identity=native_build,
            sample_order_independently_rederived=True,recorded_execution=receipt.get('execution'),
            original_source_count=len(sources),common_score_source=binding(scores_path),
            clean_reproduction_qualified=False,new_model_forwards=0)
        rewritten,suffix=revised_card(raw,review)
        require(rewritten.count(original)==1 and rewritten.endswith(suffix),'Original receipt/configuration suffix changed')
        rows.append(review)
    require(initial_inventories['C04']==initial_inventories['C05'] and np.array_equal(orders['C04'][:20],orders['C05'])
            and np.array_equal(orders['C03'],orders['C04']),'Captured seed/sample agreement differs')
    first='encoders.0.layers.0.weight'
    require({k:v for k,v in initial_inventories['C03'].items() if k!=first}==
            {k:v for k,v in initial_inventories['C04'].items() if k!=first}
            and np.array_equal(first_weights['C03'][:,[0,1,2,3,4,8,9]],first_weights['C04']),
            'C04 did not retain the exact surviving C03seed coefficients')
    require(index_path.read_bytes()==index_raw and all(hashlib.sha256(p.read_bytes()).hexdigest()==v for p,v in notes.items()),
            'Source card/index changed during review')
    require('torch' not in sys.modules,'Source review imported a neural runtime')
    result=dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,
        producer=binding(Path(__file__)),source_reviewed_fits=3,all_initial_nonfirst_layer_tensors_equal=True,
        seven_channel_initial_tensor_inventories_exact=True,sample_order_prefixes_exact=True,
        overview_addition='\n\n## Early variable-grid and CUDA pressure experiments\n\nC03 used 372,377 parameters and ten channels. C04 removed three redundant cubic-spacing maps, retaining scalar FV spacing and 371,729 parameters. Both fit 200 updates on the same six-trajectory dataset. C05 qualified CUDA with the same seven-channel initialization and first 20 sample draws, fitting 20 updates before historical CPU deployment checks. The common 128-request worst pressure errors remain about 0.291. Native build/metadata epochs are preserved separately, and clean GPU-only reproduction remains unqualified.\n',
        new_model_forwards=0,native_mutations=0,new_reference_generation=False,clean_reproduction_qualified=False)
    OUTPUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Three original variable-grid/CUDA fits source-reviewed; no model execution or native mutation.')


if __name__=='__main__':
    run()
