"""Bind authored C106/C107 explanations to original source/receipt evidence.

No model import, parameter update, simulation, native mutation or payload copy.
The resulting five-card set supersedes the unadopted three-component proposal.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from prepare_method_review_revisions import verified_review_receipt

ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'docs/plans/model-portfolio-migration-v1'
RUN=ROOT/'runs/model-portfolio-migration-v1'
OUTPUT=D/'reports/captured-model-method-reviews-02.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def binding(path):return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))


def run():
    if OUTPUT.exists():raise ValueError('Fresh authored review report required')
    original=json.loads((D/'reports/core-component-method-reviews-01.json').read_bytes())
    original_index=json.loads((RUN/'native-composition-primary-01/root/project/notes/portfolio-index.json').read_bytes())
    cards={r['item_id']:r for r in original_index['fits']}
    scores=json.loads((D/'reports/pressure-common-score-01.json').read_bytes())
    snapshots={};review_rows=[]
    arms=[('C106','v4-student50-distill','fit-c1bd1512d1e225d7',1.0),
          ('C107','v4-student50-control','fit-af2d2e2f93da1faa',0.0)]
    for label,arm,fit_id,weight in arms:
        project=ROOT/'runs/pressure-methods-v4'/(arm+'-project-01')
        train=ROOT/'runs/pressure-methods-v4'/(arm+'-study-01')/'train'
        receipt=json.loads((train/'receipt.json').read_bytes());witness=json.loads((project/'source-witness.json').read_bytes())
        config=json.loads((project/'config.json').read_bytes());fine=json.loads((project/'fine-tuning.json').read_bytes())
        distill=json.loads((project/'distillation.json').read_bytes());splits=json.loads((train/'splits.json').read_bytes())
        manifest=json.loads((project/'project.json').read_bytes())
        if (sha(project/'source-witness.json')!=receipt['closure']['source_witness_sha256']
                or sha(project/'config.json')!=receipt['closure']['experiment_sha256']
                or sha(project/'runtime-manifest.json')!=receipt['closure']['runtime_manifest_sha256']
                or receipt['closure']['inputs']!=witness['inputs']):
            raise ValueError('Original training closure differs')
        for entry in manifest['recipes']:
            if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Source recipe differs from captured manifest')
        recipe=json.loads((project/'train-recipe.json').read_bytes())
        for entry in recipe['sources']:
            if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Captured recipe source differs')
        sources=[binding(project/name) for name in sorted(witness['sources'])]
        if len(sources)!=45 or any(v['sha256']!=witness['sources'][Path(v['path']).name] for v in sources):
            raise ValueError('Full original source witness population differs')
        parameters=sum(math.prod(v['shape']) for v in fine['initial_inventory'].values())
        if parameters!=93758 or distill['weight']!=weight or fine['teacher_weight']!=0:
            raise ValueError('Captured student/teacher policy differs')
        if receipt['optimizer_updates']!=2400 or receipt['heldout_evaluations']!=0:
            raise ValueError('Original training/evaluation budget differs')
        parents=splits['parents']
        if len(parents)!=34 or any(p['split']!='train' or p['dt_s']!=1/60 or p['density_kg_m3']!=1000 for p in parents):
            raise ValueError('Original train-only population differs')
        frames=sum(len(p['frames']) for p in parents)
        if frames!=68:raise ValueError('Original selected frame population differs')
        stats=receipt['fine_tuning_statistics']
        if stats['distillation_teacher_forwards']!=(4800 if weight else 0):
            raise ValueError('Recorded teacher work differs')
        worst=scores['models'][label]['worst']
        if scores['models'][label]['weights_sha256']!=receipt['outputs']['weights.npz']['sha256']:
            raise ValueError('Fresh benchmark refers to another checkpoint')
        base=dict(fit_item_id=fit_id,note_path=cards[fit_id]['card_path'],original_receipt=binding(train/'receipt.json'),
            history=binding(train/'history.json'),source_witness=binding(project/'source-witness.json'),
            training_population=binding(train/'splits.json'),sources=sources,parameters=parameters,trainable_parameters=parameters,
            optimizer_updates=2400,train_only=True,training_parents=34,selected_training_frames=68,
            training_densities_cells_per_metre=dict(Counter(str(1/p['cell_size_m']) for p in parents)),
            shape_counts_xyz=dict(Counter(str(p['counts_xyz']) for p in parents)),
            runtime_contract=binding(project/'runtime-manifest.json'),
            reproduction_recipes=[binding(project/'train-recipe.json'),binding(project/'restore-recipe.json')],
            common_pressure_evidence=binding(D/'reports/pressure-common-score-01.json'),
            clean_reproduction_qualified=False,fresh_complete_generator_benchmark_qualified=False,
            declared_head_lr_multiplier=receipt['architecture']['shape_informed_coarse']['head_learning_rate_multiplier'],
            executed_head_lr_multiplier=10/3)
        base['method']=('A pressure-only residual predictor with 93,758 trained parameters: 92,560 fine-network coefficients, '
            '63 learned coarse logits and a 1,135-parameter grid-shape head. The fine network has four encoder widths '
            '[4,8,12,16], bottleneck 24 and reversed decoder widths, average pooling, nearest upsampling, skip '
            'concatenation and a global pooled addition. Captured selection removes GroupNorm, SiLU and convolution '
            'biases from this one-channel fine path. Its input is the normalized physical residual q=(rho/h)r; '
            'rho is fluid density and h is metres per cell. dt is validated metadata, already reflected in the '
            'externally assembled RHS, rather than an independent neural input channel. Before each of two '
            'requested internal correction passes, up to 63 non-DC cosine modes receive positive learned gains '
            'conditioned by a 3→16 tanh→63 shape head. A fine-equation residual gate may reject that coarse '
            'proposal, then the float32 CNN correction is reconstructed and accumulated in float64. The gate '
            'does not certify the final CNN output. These are pressure iterations, not two physical simulation '
            'steps; this model does not itself advance velocity or tracer.')
        base['purpose']=('Test whether the half-width student can retain useful pressure accuracy when supervised by the frozen '
            '371,214-parameter teacher’s fine corrections, reducing the fine-network width while retaining the global branch.'
            if weight else 'Provide the matched half-width control without the teacher-correction loss, so the distillation '
            'intervention can be assessed against the same initial weights, data, update budget and source implementation. '
            'C107 is a sibling experiment of C106; it was not trained by continuing the completed C106 checkpoint.')
        base['experiment']=('Both arms start from checkpoint dd6688cf203e9b65aceaeedfd123492aeb0c8200840cc40813956aba89993bce '
            'and use the same sampling order, 2,400 CUDA Adam updates, batch size one, seed 2026101905 and fixed original '
            'C53 normalization. The fine learning rate is 3e-4, then 3e-5 at update 1801 and 1e-5 at 2251. The executed '
            'head_optimizer.py multiplies the coarse/shape-head rate by 10/3; the inherited architecture field saying '
            '1000 is stale and is preserved as an original observation. Training uses 34 captured train-only parents '
            'and 68 selected frames: 15 parents at 8 cells/m, 17 at 16 and two at 32. Domains include rectangular grids '
            'and 32³,64³,128³ cubes; dt=1/60 s and fluid density=1000 kg/m³ throughout. These are parent/frame counts, '
            'not proof of 34 independent scenes or held-out generalization. The physical final/intermediate equation '
            'feedback is shared; the old eight-pass pressure-teacher term has weight zero in both arms. '
            +('C106 separately adds weight-one fine-correction distillation: 4,800 actual teacher forwards were recorded.'
              if weight else 'C107’s fine-correction distillation weight is zero and records zero teacher forwards.'))
        base['outcome']=(f'Training completed at the declared budget. On the fresh common set of 128 original consumed '
            f'pressure requests, this exact {label} checkpoint has worst mean-free pressure relative L2 '
            f'{worst["pressure_mean_free_relative_l2"]:.17g} and equation relative L2 '
            f'{worst["common_equation_relative_l2"]:.17g}. C107’s corresponding tails '
            'are about 0.00815329/0.00102944 versus C106’s 0.0155048/0.00678757. The common requests cover only '
            '8 and 16 cells/m, small grids and 1/60 s pressure solves. Both receipts record zero held-out '
            'evaluations. These results establish this controlled diagnostic comparison, not large-grid, '
            '50/100-density, full-state, multistep or drift accuracy. No new qualified latency follows from this review.')
        base['lesson']=('This matched run favors the physical-loss control on the retained diagnostic tails; adding a teacher '
            'loss does not guarantee an improvement. It is not evidence that distillation is generally harmful: '
            'replicated seeds and held-out scene/time/grid tests are still needed. Parameter count and original '
            'architecture labels alone omit active preprocessing, global gating, internal correction work and '
            'optimizer overrides. Compare these pressure-only scores separately from the nearly exact full-state '
            'component anchors and preserve the old reference epoch.')
        base['reproduction']=('The exact original receipt/history, source witness, all 45 witnessed files, fixed normalization, '
            'training population, original train/restore recipes and runtime manifest are bound below or in the review '
            'report. Reproduction requires the witness’s twelve exact input blobs, shared initial/teacher weights and '
            'the pinned original solver/data contract, then an explicitly admitted CUDA training execution with the '
            'captured configuration. Opening a project must not execute it. Clean retraining and exact final-checkpoint '
            'reproduction have not been exercised; new Rheon-generated data must be a separate physics epoch.')
        verified_review_receipt(base)
        review_rows.append(base)
        snapshots[label]=dict(receipt=receipt,witness=witness,config=config,fine=fine,distill=distill,splits=splits)
    a,b=snapshots['C106'],snapshots['C107']
    shared=('inputs','normalization_sha256','runtime_manifest_sha256','sample_order_sha256')
    if any(a['receipt']['closure'][k]!=b['receipt']['closure'][k] for k in shared):
        raise ValueError('Pair does not share the declared controlled closure')
    if {k:v for k,v in a['config'].items() if k!='id'}!={k:v for k,v in b['config'].items() if k!='id'}:
        raise ValueError('Scientific configurations differ beyond their arm ID')
    if {k:v for k,v in a['distill'].items() if k!='weight'}!={k:v for k,v in b['distill'].items() if k!='weight'}:
        raise ValueError('Distillation specifications differ beyond the tested weight')
    differences={k for k,v in a['witness']['sources'].items() if b['witness']['sources'].get(k)!=v}
    if differences!={'config.json','distillation.json','fine-tuning.json'}:
        raise ValueError('Captured executable source differs between the paired arms')
    overview='''
## Source-reviewed methods and distinct comparison scopes

Five of 205 fit cards now have retrospective source reviews prepared: the shared
spectral pressure response, the local transport sampler, the fixed-viscosity
parent and the matched C106/C107 pressure students. Original receipts and
configurations remain exact below each authored explanation. Clean retraining
and full portfolio acceptance remain pending.

The spectral pressure response has four learned float64 coefficients fitted on
148 training scenes. The prescribed local sampler has 48 learned float64
parameters fitted on 16 scenes. The fixed-viscosity parent records 100,456 total
parameters and 6,600 trainable parameters, with a frozen 243-coefficient viscosity
kernel. A later composition can replace components and cannot inherit the
parent’s entire active parameter count or training coverage.

C106/C107 each have 93,758 parameters and two internal pressure correction
passes; they do not generate two physical simulation steps. They are matched
siblings starting from the same student initialization. The teacher-loss arm
records 4,800 teacher forwards; the control records zero. The captured optimizer
uses a 10/3 head learning-rate multiplier despite inherited metadata saying
1000. The common 128-request diagnostic favors C107’s tails, with no independent
held-out or freshly qualified speed claim.

Pressure relative L2, full-state RMS errors, equation residual and fitting losses
are distinct metrics. The N8-labelled cell-coordinate anchor actually takes its
N1 branch on the quoted case. Existing K-output generators still advance each
physical output in sequence; the reviewed components do not establish endpoint
prediction with cheap intermediate frames. All model and source identities,
parameter populations, integration work and reference epochs remain separate.
'''
    result=dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),
        classification='authored_retrospective_interpretation_original_evidence_preserved',rows=original['rows']+review_rows,
        overview_addition=overview,source_component_review=binding(D/'reports/core-component-method-reviews-01.json'),
        paired_source_differences=sorted(differences),all45sources_verified_per_arm=True,
        both_original_histories_and_population_bytes_verified=True,models_executed=0,parameter_updates=0,
        reference_simulations_executed=0,original_receipts_modified=False,runner_sha256=sha(Path(__file__)))
    with OUTPUT.open('x') as stream:json.dump(result,stream,indent=2,ensure_ascii=False);stream.write('\n')
    print(json.dumps(dict(status='five_source_reviews_bound_not_adopted',new_reviews=2,total_reviews=5,
                         paired_parameters=93758,source_files_verified=90,report=str(OUTPUT.relative_to(ROOT)))))


if __name__=='__main__':run()
