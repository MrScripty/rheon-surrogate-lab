"""Review original C37/C44/C53 training from captured source, without importing it."""
import ast
from collections import Counter
import math
from pathlib import Path
import time

from prepare_method_review_revisions import encode, revised_card, verified_review_receipt
from review_detached_C37_fits import ROOT, DOC, PROJECT, binding, read, require

OUTPUT = DOC / 'reports/captured-pressure-parent-method-reviews-01.json'
SPECS = [
    ('C37','runs/water-shape-repeatability-v1/shape-one-longer/project',
     'runs/water-shape-repeatability-v1/shape-one-longer/managed-full/stage-00/artifacts',10000,1),
    ('C44','runs/water-late-residual-v1/extra-late/project',
     'runs/water-late-residual-v1/extra-late/managed-full/stage-00/artifacts',5000,2),
    ('C53','runs/pressure-current-policy-v1/current-late-project-02',
     'runs/pressure-current-policy-v1/current-late-study-03/train',5000,2),
]


def function(path,name):
    return next(node for node in ast.parse(path.read_text()).body
                if isinstance(node,ast.FunctionDef) and node.name==name)


def run():
    require(not OUTPUT.exists(),'Fresh original-parent review required')
    cards={row['item_id']:row for row in read(PROJECT/'notes/portfolio-index.json')['fits']}
    common=read(DOC/'reports/pressure-common-score-01.json')['models']
    matrix_path=DOC/'reports/pressure-correction-work-01.json'
    matrix=read(matrix_path)
    require(matrix['status']=='all_7680_retained_pressure_policy_fields_independently_rescored',
            'Independent matched-work evidence required')
    initial=None
    physical_transform=None
    rows=[]
    for label,project_name,training_name,updates,corrections in SPECS:
        project,training=ROOT/project_name,ROOT/training_name
        receipt,history=read(training/'receipt.json'),read(training/'history.json')
        config,witness=read(project/'config.json'),read(project/'source-witness.json')
        require(receipt['mode']=='train' and receipt['status']=='complete'
                and receipt['optimizer_updates']==config['training']['updates']==updates
                and config['correction']==dict(activation='linear',steps=corrections),
                'Original training budget or deployment policy differs')
        require(binding(project/'config.json')['sha256']==receipt['closure']['experiment_sha256']
                and binding(project/'source-witness.json')['sha256']==receipt['closure']['source_witness_sha256']
                and binding(project/'runtime-manifest.json')['sha256']==receipt['closure']['runtime_manifest_sha256'],
                'Original configuration, source witness or runtime declaration differs')
        sources=[binding(project/name) for name in sorted(witness['sources'])]
        require(len(sources)=={'C37':20,'C44':21,'C53':25}[label]
                and all(source['sha256']==witness['sources'][Path(source['path']).name] for source in sources),
                'Captured complete source inventory differs')
        manifest=read(project/'project.json')
        inputs={entry['id']:entry for entry in manifest['data']}
        bound_inputs=[]
        for slot,expected in receipt['closure']['inputs'].items():
            path=ROOT/inputs[slot]['path'];observed=binding(path)
            require(dict(sha256=observed['sha256'],bytes=path.stat().st_size)==expected,
                    'Captured original input differs: '+slot)
            bound_inputs.append(dict(observed,slot=slot,bytes=path.stat().st_size))
        recipe=read(project/'train-recipe.json')
        for source in [*recipe['sources'],*manifest['recipes']]:
            require(binding(ROOT/source['path'])['sha256']==source['sha256'],'Original recipe source changed')
        for name in ('weights.npz',f'checkpoint-{updates:03d}.npz','normalization.json','history.json','splits.json'):
            require(dict(sha256=binding(training/name)['sha256'],bytes=(training/name).stat().st_size)
                    == {key:receipt['outputs'][name][key] for key in ('sha256','bytes')},'Original output differs')
        require(receipt['outputs']['weights.npz']['sha256']==common[label]['weights_sha256'],
                'Named checkpoint is not the final original training output')
        inventory=history['checkpoints']['checkpoint-000.npz']['inventory']
        require(len(inventory)==19 and sum(math.prod(value['shape']) for value in inventory.values())==370016,
                'Original parameter inventory differs')
        if initial is not None:
            require(initial==inventory,'Parent fits do not share their seeded initial tensors')
        initial=inventory
        transform=ast.dump(function(project/'physical_residual.py','transform'),include_attributes=False)
        if physical_transform is not None:
            require(physical_transform==transform,'Actual activation/bias transform differs')
        physical_transform=transform
        require("module.register_parameter('bias',None)" in (project/'physical_residual.py').read_text()
                and 'if isinstance(layer,nn.SiLU):block.layers[index]=nn.Identity()' in (project/'physical_residual.py').read_text()
                and 'isinstance(layer,torch.nn.GroupNorm)' in (project/'model.py').read_text(),
                'Actual linear correction transform no longer matches the reviewed source')
        population=read(training/'splits.json')
        splits=Counter(parent['entry']['split'] for parent in population['parents'])
        parents=[parent for parent in population['parents'] if parent['entry']['split']=='train']
        require(splits==({'train':16} if label=='C53' else {'train':16,'validation':64})
                and sum(len(parent['frames']) for parent in parents)==32,
                'Original physical fitting population differs')
        banks={}
        for key,grids in [('analytical_feedback',8),('supplemental_feedback',6)]:
            if key not in history:
                continue
            value=history[key]
            require(len(value['grids'])==grids and sum(value['grid_exposures'])==updates
                    and value['additional_examples']==updates
                    and all(len(grid['bank']['samples'])==128 for grid in value['grids']),
                    'Manufactured fitting bank or actual exposure differs')
            banks[key]=dict(grids=grids,patterns_per_grid=128,distinct_grid_patterns=grids*128,
                actual_extra_optimizer_examples=value['additional_examples'],weight=value['weight'],
                grid_exposures=value['grid_exposures'],sampling_order_sha256=value['sampling_order_sha256'])
        for key,expected_cases in [('direction_feedback',144 if label=='C44' else 192),('current_feedback',144)]:
            if key not in history:
                continue
            value=history[key]
            slot='direction_manifest' if key=='direction_feedback' else 'current_manifest'
            description=read(ROOT/inputs[slot]['path'])
            require(value['cases']==expected_cases and value['additional_examples']==updates
                    and value['manifest_sha256']==binding(ROOT/inputs[slot]['path'])['sha256']
                    and description['source']['weights_sha256'], 'Captured extra direction bank differs')
            banks[key]=dict(cases=expected_cases,actual_extra_optimizer_examples=updates,weight=value['weight'],
                stage=value['stage'],source_checkpoint_sha256=description['source']['weights_sha256'],
                manifest=binding(ROOT/inputs[slot]['path']),sampling_order_sha256=value['sampling_order_sha256'])
        if label=='C53':
            require(history['optimizer_work']['network_forwards']==50000
                    and config['band_feedback']==dict(max_mode=3,signal_floor_fraction=.1,
                        stages='first-and-final',weight=.03), 'Measured work or low-mode objective differs')
        score=matrix['models'][label]
        require(score['original_default_output_parity_cases']==128,'Original default pressure replay differs')
        worst=score['summary']['captured_default']['worst']
        fit='fit-'+binding(training/'receipt.json')['sha256'][:16]
        card=cards[fit]
        require(card.get('method_review_status')!='captured_source_and_receipt_reviewed','Already reviewed parent')
        purpose={'C37':'Test longer scheduled training of a single pressure correction using real train frames and a broad manufactured grid-pattern stream.',
                 'C44':'Improve two-correction contraction on difficult residual directions while retaining primary and supplemental manufactured-shape feedback.',
                 'C53':'Add current-policy residual directions and low-spatial-mode pressure feedback while preserving a two-correction deployment.'}[label]
        experiment=(f"Train the same 370,016-parameter seeded network for {updates:,} Adam updates, batch size one, "
            f"using 16 physical training trajectories / 32 frames and {corrections} pressure correction(s) per prediction. "
            'The primary manufactured bank contains 1,024 grid-pattern pairs across eight grids, including up to 40 cubed. ')
        if label!='C37':
            experiment+='Add 768 supplemental grid-pattern pairs on six rectangular grids and late captured directions. '
        if label=='C53':
            experiment+='Also add 144 current-policy residual cases and a 0.03-weight low-mode DCT-II loss at first/final corrections; actual optimizer work records 50,000 root forwards. '
        experiment+='All parents share identical initial tensor inventories; several training factors and the C37 deployment work differ, so their ranking is not a one-factor causal ablation.'
        row=dict(fit_item_id=fit,note_path=card['card_path'],
            method='A one-channel multiscale 3D convolutional pressure correction network, encoder widths 8/16/24/32 and bottleneck48. '
                'Actual captured transforms remove GroupNorm, convolution biases and SiLU activations for the linear correction policy; '
                'the inherited activation=silu metadata does not describe those transformed layers. '
                'The F32 network consumes the scaled pressure-equation residual q=(rho/h)r with train-only RMS normalization. '
                'F64 finite-volume residual assembly and cell-zero-gauge reconstruction wrap each correction. '
                'Timestep is validated at the call boundary but is not a separate neural input channel in this physical pressure formulation.',
            purpose=purpose,experiment=experiment,
            outcome=f"{label}, at its saved {corrections}-correction default on the frozen 128 consumed requests: "
                f"worst mean-free pressure relative L2 {worst['pressure_mean_free_relative_l2']:.17g}; "
                f"equation relative L2 {worst['common_equation_relative_l2']:.17g}. "
                'The retained matrix additionally measures one/two/four/eight corrections. No fresh timing or trajectory-drift result belongs to this review.',
            lesson='Separate physical CFD frames, manufactured spatial patterns and captured residual directions when describing training diversity. '
                'These parents retain a much broader fitting stream than the later eight-pattern detached fine-tunes. '
                'C37 continuations improve only conditionally on metric and correction budget, while the C44/C53 continuations regress '
                'against their own parents at every matched tested budget. This motivates preserving broad real/directional rehearsal '
                'during future correction tuning, but does not identify narrow fitting as the sole cause. '
                'Manufactured large grids do not establish high-density or large-domain physical-trajectory accuracy.',
            reproduction='Use the exact captured project config, train-recipe.json, complete source witness, input-slot bindings and runtime manifest. '
                'Training starts from the recorded seeded checkpoint and saves the final configured update; optimizer resume is not captured. '
                'Keep the frozen old-Rheon data epoch distinct from simulations generated by the new solver. '
                'This source/receipt review and unchanged-checkpoint GPU replay do not qualify clean dataset regeneration or retraining.',
            original_receipt=binding(training/'receipt.json'),history=binding(training/'history.json'),
            source_witness=binding(project/'source-witness.json'),sources=sources,
            training_population=binding(training/'splits.json'),training_inputs=bound_inputs,
            retained_accuracy=common[label]['result'],independent_accuracy=binding(matrix_path),
            actual_train_trajectories=16,actual_train_frames=32,declared_other_splits=dict(splits),
            initialized_tensor_inventory_sha256=sha_inventory(inventory),parameter_count_from_inventory=370016,
            fitting_streams=banks,actual_counted_optimizer_forwards=history.get('optimizer_work',{}).get('network_forwards'),
            inherited_metadata_discrepancy='architecture activation=silu survives while the explicit linear transform removes SiLU')
        original=verified_review_receipt(row)
        raw=(PROJECT/row['note_path']).read_bytes()
        require(raw.count(original)==1 and original in revised_card(raw,row)[1],'Original retained receipt would change')
        rows.append(row)
    report=dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,
        classification='authored_source_and_receipt_interpretation_not_retraining',producer=binding(Path(__file__)),
        overview_addition='C37/C44/C53 share initial tensors and a transformed linear convolutional pressure core, '
            'but differ in optimizer budget, correction count and fitting streams. Broad manufactured and captured-direction '
            'training is distinct from physical scene diversity. Preserve that context when evaluating eight-pattern fine-tunes.',
        new_model_forwards=0,native_mutations=0,clean_reproduction_qualified=False)
    OUTPUT.write_bytes(encode(report))
    print('Three original parent reviews complete; source count66, original receipts preserved.')


def sha_inventory(value):
    import hashlib
    return hashlib.sha256(encode(value)).hexdigest()


if __name__=='__main__':
    run()
