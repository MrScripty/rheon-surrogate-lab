"""Source-only retrospective interpretation of four nonlinear correction fits."""
import ast
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import zipfile
from prepare_method_review_revisions import revised_card, verified_review_receipt

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/full-ai-data-diversity-v1'
PROJECT=ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT=ROOT/'docs/plans/model-portfolio-migration-v1/reports/captured-internal-residual-method-reviews-01.json'
OVERVIEW_ADDITION='''
## What the reduced-integration residual fits actually changed

Four additional cards distinguish S1/S2 AdamW from S3 physical-objective LBFGS
and S4 direct-correction supervision. All use the same10,876-parameter width8
subnet. Of148 counted training scenes,29 provide correction gradients;119
contribute fixed bypass controls. The selected S2 state equals initialization.
S4's improved direct correction loss accompanies a worse physical objective.

The N8 runner replaces eight fine advances with groups2/2/3/1 and three learned
corrections per physical output. K outputs still advance sequentially; this is
not endpoint prediction with cheap intermediate frames. Exact replay preserves
the recorded failures, so those failures cannot be dismissed as a changed
restoration path. These are source reviews, not clean retraining or new timing.
'''

def read(path):return json.loads(path.read_bytes())
def sha(raw):return hashlib.sha256(raw).hexdigest()
def binding(path):return dict(path=str(path.relative_to(ROOT)),sha256=sha(path.read_bytes()))

def checkpoint(path,expected,inventory):
    raw=path.read_bytes()
    if dict(sha256=sha(raw),bytes=len(raw))!=expected:raise ValueError('Recorded checkpoint archive changed')
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist())!={name+'.npy' for name in inventory}:raise ValueError('Checkpoint tensor membership differs')
        for name,spec in inventory.items():
            raw=archive.read(name+'.npy')
            if raw[:8]!=b'\x93NUMPY\x01\x00':raise ValueError('Expected original simple NPY v1 tensors')
            length=struct.unpack('<H',raw[8:10])[0];header=ast.literal_eval(raw[10:10+length].decode())
            if (list(header['shape'])!=spec['shape'] or header['descr']!=spec['dtype'] or header['fortran_order']
                    or sha(raw[10+length:])!=spec['sha256']):raise ValueError('Checkpoint tensor differs from original inventory')
    parameters=sum(math.prod(value['shape']) for name,value in inventory.items() if name not in ('input_rms','output_rms'))
    if parameters!=10876 or any(value['dtype']!='<f4' for value in inventory.values()):raise ValueError('Original width8 parameter inventory differs')


def run():
    if OUTPUT.exists():raise ValueError('Fresh review output required')
    manifest_path=BASE/'internal-residual-view-01/manifest.json';manifest=read(manifest_path)
    if (not manifest['all148training_scenes_in_objective'] or manifest['development_or_reserved_used']
            or manifest['expanded_reserved52_references_opened'] or manifest['future_native_labels_used_for_features']):
        raise ValueError('Expected train-only original view')
    active={r['identifier']:r for r in manifest['parts']};fixed={r['identifier']:r for r in manifest['control_rows']}
    if len(active)!=29 or len(fixed)!=119 or len(manifest['parts'])!=783 or len(manifest['control_rows'])!=1071:
        raise ValueError('Exact effective training population differs')
    generator=ROOT/'experiments/full-ai-data-diversity-v1/prepare_internal_residual_view_v1.py'
    if manifest['source_sha256'][str(generator.relative_to(ROOT))]!=sha(generator.read_bytes()):raise ValueError('Original view generator changed')
    index=read(PROJECT/'notes/portfolio-index.json');cards={fit['item_id']:fit for fit in index['fits']}
    originals=[];rows=[]
    purposes=[
        'Learn a small nonlinear correction for the difference between grouped coarse evolution and the original fine internal advances, keeping the learned macro backbone frozen.',
        'Change only training metric conditioning while preserving S1 architecture, initialization, ordering, scene population and two-epoch optimizer schedule.',
        'Continue the exact selected S1 weights using deterministic full-objective quasi-Newton optimization instead of single-sample AdamW; preserve the same correction network and physical objective.',
        'Continue the exact selected S3 weights but directly supervise the four missing correction heads, while tracking the coupled physical objective separately.'
    ]
    for stage in range(1,5):
        project=BASE/f'internal-residual-s{stage}-project-{"02" if stage==3 else "01"}'
        train=BASE/f'internal-residual-s{stage}-pair-01/train'
        receipt=read(train/'receipt.json');history=read(train/'history.json');config=read(project/'config.json')
        if receipt['status']!='complete' or receipt['mode']!='train' or receipt['parameters']!=10876 or not history['train_only']:
            raise ValueError('Original successful CUDA fit required')
        if binding(project/'config.json')['sha256']!=receipt['closure']['config_sha256']:raise ValueError('Executed configuration differs')
        sources=[binding(project/name) for name in sorted(receipt['closure']['sources'])]
        if any(row['sha256']!=receipt['closure']['sources'][Path(row['path']).name] for row in sources):raise ValueError('Captured source differs')
        checkpoint(train/'initial-weights.npz',receipt['outputs']['initial-weights.npz'],receipt['initial_inventory'])
        checkpoint(train/'weights.npz',receipt['outputs']['weights.npz'],receipt['final_inventory'])
        for key in ('initial_inventory','final_inventory','optimizer_updates','selected_full148scene_objective'):
            if history[key]!=receipt[key]:raise ValueError('Fit/history selection differs')
        if stage<3:
            if receipt['optimizer_updates']!=1566 or config['epochs']!=2:raise ValueError('Expected two complete AdamW epochs')
            if any(r['local_samples']!=783 or r['active_scenes']!=29 or r['fixed_scenes']!=119 for r in history['epoch_coverage']):raise ValueError('Epoch population differs')
            data_path=manifest_path
        else:
            data_path=BASE/f'internal-residual-s{stage}-data-01/manifest.json';compact=read(data_path)
            parent=originals[0] if stage==3 else originals[2]
            warm=base64.b64decode(compact['inline_warm_weights_npz_base64'],validate=True)
            if (receipt['initial_inventory']!=parent['final_inventory'] or sha(warm)!=parent['outputs']['weights.npz']['sha256']
                    or compact['inline_warm_restore_receipt']['final_inventory']!=parent['final_inventory']
                    or compact['original_manifest']!=str(manifest_path.relative_to(ROOT))):raise ValueError('Exact selected warm start differs')
            if (receipt['optimizer_updates'],receipt['closure_evaluations'])!=((7,18) if stage==3 else (24,51)):
                raise ValueError('Recorded quasi-Newton counter/closure evidence differs')
        source_manifest=read(project/'project.json')
        references={r['id']:r for r in source_manifest['data']}
        inputs=[]
        for slot in ('manifest','normalization','optimization_normalization','runtime_manifest'):
            path=ROOT/references[slot]['path'];observed=binding(path)
            if dict(sha256=observed['sha256'],bytes=path.stat().st_size)!=receipt['closure']['inputs'][slot]:raise ValueError('Static fit input differs')
            inputs.append(dict(slot=slot,**observed,bytes=path.stat().st_size))
        if references['manifest']['path']!=str(data_path.relative_to(ROOT)):raise ValueError('Fit used another training view')
        if stage==2 and (receipt['initial_inventory']!=originals[0]['initial_inventory'] or receipt['final_inventory']!=receipt['initial_inventory'] or receipt['selected_epoch']!=0):
            raise ValueError('Loss-conditioned arm must retain original zero-correction checkpoint')
        fit='fit-'+binding(train/'receipt.json')['sha256'][:16];card=cards[fit]
        measured=read(BASE/f'internal-residual-s{stage}-native-blocks-01/results.json')
        if measured['model_identity']['residual_weights_sha256']!=receipt['outputs']['weights.npz']['sha256']:raise ValueError('Evaluated correction checkpoint differs')
        detail=next(v for v in read(ROOT/'docs/plans/model-portfolio-migration-v1/reports/full-state-residual-accuracy-reproduction-01.json')['comparisons'] if v['variant']==f'residual-s{stage}')
        regimes={row['reference_substeps']:row for row in detail['reference_regimes']}
        outcome=f"Selected training objective {receipt['selected_full148scene_objective']:.17g}; "
        if stage<3:outcome+=f"selected epoch {receipt['selected_epoch']} after 1,566 updates. "
        else:outcome+=f"selected closure {receipt['selected_evaluation']} after {receipt['optimizer_updates']} recorded LBFGS iterations / {receipt['closure_evaluations']} closure evaluations. "
        if stage==2:outcome+='The selected complete tensor inventory exactly equals initialization; the zero-output heads remain selected despite executing the training updates. '
        if stage==4:
            selected=next(row for row in history['history'] if row['evaluation']==receipt['selected_evaluation'])
            outcome+=f"Its separately recorded physical objective is {selected['physical_objective']:.17g}; the direct-delta objective cannot be compared numerically with S1/S3 physical objectives. "
        outcome+=f"The retained complete-generator replay passes all {regimes[1]['scored_outputs']} N1 outputs and fails {regimes[8]['diagnostic_failures']} of {regimes[8]['scored_outputs']} N8 outputs at the diagnostic1e-5 gate. These are original consumed cases, not independent validation or qualified new latency."
        experiment=('Two CUDA AdamW epochs consume all783 active scene/start/group samples each, cosine learning rate0.003→0.0003, epsilon1e-12, zero weight decay and gradient clip1. Select the minimum complete training objective including initialization; no development/reserved selection.' if stage<3 else
                    f"CUDA full-batch LBFGS, {8 if stage==3 else 24} maximum iterations and {18 if stage==3 else 60} maximum closures. Cache the same teacher-derived inputs/targets on GPU; select the best complete closure including the exact warm start. Recorded internal iteration counters are not minibatch update counts.")
        if stage==2:experiment+=' Training metric weights are equation0.3686505980, pressure0.7769568476, velocity13044.41817 and tracer3922.417979; physical acceptance metrics remain unchanged.'
        if stage==4:experiment+=' Objective is per-head output-RMS-normalized correction MSE; physical metrics are diagnostics, not the selection objective.'
        review=dict(fit_item_id=fit,note_path=card['card_path'],original_receipt=binding(train/'receipt.json'),history=binding(train/'history.json'),sources=sources,
            training_inputs=inputs,training_configuration=binding(project/'config.json'),parameters=10876,trainable_parameters=10876,
            optimizer_updates=receipt['optimizer_updates'],training_scenes=148,active_gradient_scenes=29,fixed_bypass_scenes=119,
            training_archive_declaration=receipt['closure']['inputs']['part_00'],large_training_archive_rehashed=False,
            method='A10,876-parameter float32 CUDA correction CNN:42 cell-lattice input channels, two width8 clamped3×3×3 convolutions with SiLU, three staggered transposed-convolution velocity heads and one tracer head. Output heads initialize to zero. The quiescence gate uses current/coarse-change/known-force channels. Pressure correction is supplied through the frozen learned coupled projection; this subnet does not directly emit pressure. Inputs include spacing100h, group duration480dt, group width and ordered known forces; future fields are labels only.',
            purpose=purposes[stage-1],experiment=experiment+' All148 training scenes are counted, but only29N8 scenes provide parameter-dependent gradients:9 starts×3 groups=783 local samples. The119N1 scenes contribute1,071 fixed control windows with zero parameter derivative. Active grids span eight shapes through64³, at8/10/16/25/50/100cells per metre; the complete fixed/control population includes19shapes through128³.',
            outcome=outcome,
            lesson='Optimization improvements on local teacher-state corrections do not establish accurate own-history reduced integration. The N8 runner advances groups2/2/3/1 with three correction calls per physical1/60s output; K1–K4 outputs still advance sequentially. It does not predict a distant endpoint with cheap temporal intermediate frames. N1 bypasses this correction and preserves the original branch. Keep effective gradient scenes, fixed controls, loss choice, trained checkpoint and complete execution variant separate.',
            reproduction='Use the exact captured trainer/configuration/static inputs and retained selected/initial inventories shown here and in the original facts below. S1/S2 share initialization; S3 starts from selected S1, S4 from selected S3. The physical-state view is prepared by prepare_internal_residual_view_v1.py; compact views regenerate teacher labels with the captured teacher-code archive and learned assets. Static inputs and checkpoint tensors were freshly checked; the9.77GB full /0.671GB compact archives were not reread, labels were not decoded and clean retraining/new-Rheon regeneration remain unqualified. CPU restore receipts copy the CUDA probe without a CPU model forward.')
        verified_review_receipt(review)
        revised,suffix=revised_card((PROJECT/card['card_path']).read_bytes(),review)
        if not revised.endswith(suffix):raise ValueError('Original retained facts changed')
        rows.append(review);originals.append(receipt)
    result=dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),classification='authored_retrospective_interpretation_original_evidence_preserved',rows=rows,
        producer=binding(Path(__file__)),overview_addition=OVERVIEW_ADDITION,
        training_population=dict(manifest=binding(manifest_path),active_scenes=29,fixed_scenes=119,local_samples=783,fixed_windows=1071,
            all_scene_densities=dict(Counter(r['cells_per_metre'] for r in [*active.values(),*fixed.values()])),active_scene_densities=dict(Counter(r['cells_per_metre'] for r in active.values()))),
        scope='Four source-only reviews. All captured trainer sources, static inputs, original output archives/tensor inventories/history and selected warm starts checked. Large training archives not reread or decoded. Existing reviewed-card consumer exercised before output. No native project mutation, model import/forward, simulation, clean retraining or fresh latency claim.',
        new_neural_forwards=0,native_mutations=0,clean_reproduction_qualified=False)
    OUTPUT.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status='four_residual_fit_source_reviews_verified',fits=[r['fit_item_id'] for r in rows],native_adopted=False)))

if __name__=='__main__':run()
