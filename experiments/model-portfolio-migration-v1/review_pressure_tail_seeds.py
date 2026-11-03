"""Audit three captured pressure-tail fits without running their model code.

Bind the active head-only recipe, all original sources/static inputs, frozen
inventories, teacher geometry partition and retained common-pressure results.
No inference, fitting, simulation, native mutation or dataset copy is performed.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from prepare_method_review_revisions import verified_review_receipt

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/plans/model-portfolio-migration-v1'
BASE=ROOT/'runs/full-ai-next-priorities-v1'
OUTPUT=DOC/'reports/captured-model-method-reviews-03.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_bytes())
def binding(path):return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
def head(name):return name=='tail_logits' or name.startswith('tail_shape_head.')


def run():
    if OUTPUT.exists():raise ValueError('Fresh source-audited review report required')
    previous=read(DOC/'reports/captured-model-method-reviews-02.json')
    index=read(ROOT/'runs/model-portfolio-migration-v1/native-composition-primary-01/root/project/notes/portfolio-index.json')
    cards={r['item_id']:r for r in index['fits']}
    scores=read(DOC/'reports/pressure-common-score-01.json')
    teacher_manifest=BASE/'tail-teacher-bank-01/manifest.json';teacher=read(teacher_manifest)
    archive=BASE/'tail-teacher-bank-01/teacher.npz'
    if sha(archive)!=teacher['archive_sha256'] or archive.stat().st_size!=teacher['archive_bytes']:
        raise ValueError('Original teacher archive differs')
    generator=ROOT/'experiments/full-ai-next-priorities-v1/prepare_tail_teacher.py'
    if sha(generator)!=teacher['source_sha256']:raise ValueError('Original teacher generator differs')
    # This is the small offline label dataset, never a neural checkpoint.
    with np.load(archive,allow_pickle=False) as saved:
        arrays={name:saved[name] for name in saved.files}
    if set(arrays)!=set(teacher['inventory']):raise ValueError('Teacher array membership differs')
    for name,value in arrays.items():
        declared=teacher['inventory'][name]
        if (value.dtype.str!=declared['dtype'] or list(value.shape)!=declared['shape']
                or hashlib.sha256(value.tobytes()).hexdigest()!=declared['sha256']):
            raise ValueError('Original teacher array differs: '+name)
    counts=arrays['counts_zyx'];split=arrays['split']
    geometry={tuple(int(n) for n in shape):('development' if int(part) else 'train')
              for shape,part in zip(counts,split)}
    if (len(geometry)!=1728 or int(counts.min())!=5 or int(counts.max())!=16
            or Counter(geometry.values())!={'train':1526,'development':202}
            or not bool((arrays['tier_small']==1).all())):
        raise ValueError('Original whole-geometry training partition differs')
    shared_parent=None;rows=[];source_sets=[]
    arms=[('C128','a-tail8','fit-be632c4368575d8f'),
          ('C129','a-tail8-s2','fit-6f787c8050986aeb'),
          ('C130','a-tail8-s3','fit-516f4d627c6425fd')]
    for label,arm,fit in arms:
        project=BASE/(arm+'-project-01');train=BASE/(arm+'-pair-01/train')
        receipt=read(train/'receipt.json');history=read(train/'history.json')
        witness=read(project/'source-witness.json');manifest=read(project/'project.json')
        config=read(project/'config.json');calibration=read(project/'calibration.json')
        recipe=read(project/'train-recipe.json');restore=read(BASE/(arm+'-pair-01/restore/receipt.json'))
        if (sha(project/'source-witness.json')!=receipt['closure']['source_witness_sha256']
                or sha(project/'config.json')!=receipt['closure']['experiment_sha256']
                or sha(project/'calibration.json')!=receipt['closure']['calibration_sha256']
                or witness['inputs']!=receipt['closure']['inputs']):
            raise ValueError('Captured fit closure differs')
        sources=[binding(project/name) for name in sorted(witness['sources'])]
        if len(sources)!=57 or any(s['sha256']!=witness['sources'][Path(s['path']).name] for s in sources):
            raise ValueError('Complete source witness differs')
        python_sources={Path(s['path']).name:s['sha256'] for s in sources if Path(s['path']).suffix=='.py'}
        if len(python_sources)!=47 or python_sources!=receipt['closure']['sources']:
            raise ValueError('Receipt directory source inventory differs')
        for entry in manifest['recipes']:
            if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Captured recipe differs')
        for entry in recipe['sources']:
            if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Captured recipe source differs')
        if (len(recipe['sources'])!=60 or recipe['entrypoint']!=str((project/'head_train.py').relative_to(ROOT))
                or recipe['arguments'][0]!={'literal':'train'}
                or {'source':str((project/'calibration.json').relative_to(ROOT))} not in recipe['arguments']):
            raise ValueError('Actual recipe does not select head-only calibration')
        inputs=[dict(slot=e['id'],**binding(ROOT/e['path']),bytes=(ROOT/e['path']).stat().st_size)
                for e in manifest['data']]
        if len(inputs)!=6 or {e['slot'] for e in inputs}!=set(receipt['closure']['inputs']):
            raise ValueError('Actual static training input population differs')
        initial,final=receipt['initial_inventory'],receipt['final_inventory']
        if initial!=history['initial_inventory'] or final!=history['final_inventory'] or set(initial)!=set(final):
            raise ValueError('Original complete parameter inventories differ')
        total=sum(math.prod(v['shape']) for v in initial.values())
        tail=sum(math.prod(v['shape']) for k,v in initial.items() if head(k))
        frozen={k:v for k,v in initial.items() if not head(k)}
        if (total!=128938 or tail!=33982 or len([k for k in initial if head(k)])!=5
                or frozen!={k:v for k,v in final.items() if not head(k)}
                or not receipt['fine_tensors_bitwise_unchanged']):
            raise ValueError('Frozen parent or trained-tail scope differs')
        if shared_parent is not None and shared_parent!=frozen:raise ValueError('Seed arms do not share frozen parent tensors')
        shared_parent=frozen
        if (receipt['parameters']!=total or receipt['trainable_parameters']!=tail or receipt['optimizer_updates']!=10000
                or history['optimizer_work']!={'head_forwards':10000,'fine_spatial_forwards_during_fitting':0,'teacher_examples':2560000}
                or history['physical_reference_fields_used'] or calibration['batch_size']!=256
                or calibration!=receipt['architecture']['tail_calibration']):
            raise ValueError('Executed calibration budget differs')
        if (restore['optimizer_updates']!=0 or not restore['full_cpu_probe_bitwise_verified']
                or restore['outputs']['weights.npz']!=receipt['outputs']['weights.npz']):
            raise ValueError('Historical restore does not retain the exact training checkpoint')
        score=scores['models'][label]
        if score['weights_sha256']!=receipt['outputs']['weights.npz']['sha256']:
            raise ValueError('Pressure diagnostic refers to another checkpoint')
        result_path=ROOT/score['result']['path']
        if sha(result_path)!=score['result']['sha256']:raise ValueError('Retained pressure evidence differs')
        case_rows=read(result_path)['rows']
        partition=Counter(geometry[tuple(reversed(r['counts_xyz']))] for r in case_rows)
        if len(case_rows)!=128:raise ValueError('Common pressure population differs')
        row=dict(fit_item_id=fit,note_path=cards[fit]['card_path'],original_receipt=binding(train/'receipt.json'),
            history=binding(train/'history.json'),source_witness=binding(project/'source-witness.json'),sources=sources,
            training_inputs=inputs,data_generation_source=dict(**binding(generator),manifest_slot='teacher_manifest'),
            parameters=total,trainable_parameters=tail,frozen_parameters=total-tail,optimizer_updates=10000,
            training_geometries=1526,development_geometries=202,physical_training_scenes=0,
            common_pressure_teacher_geometry_partition=dict(partition),
            seed=calibration['seed'],sampling_seed=calibration['sampling_seed'],
            sampled_geometry_order_sha256=history['sampled_geometry_order_sha256'],
            calibration=binding(project/'calibration.json'),runtime_contract=next(i for i in inputs if i['slot']=='runtime_manifest'),
            reproduction_recipes=[binding(project/'train-recipe.json'),binding(project/'restore-recipe.json')],
            common_pressure_evidence=binding(result_path),historical_restore=binding(BASE/(arm+'-pair-01/restore/receipt.json')),
            clean_reproduction_qualified=False,fresh_complete_generator_benchmark_qualified=False)
        row['method']=('A pressure-only routed C107 successor with 128,938 total parameters. Its unchanged 94,956-parameter '
            'parent makes two fine CNN corrections with learned coarse blocks. For domains with every axis at most 16 cells, '
            'a new 33,982-parameter tail predicts 511 positive gains for non-DC cosine modes 0..7 from log(Nz/16), '
            'log(Ny/16), log(Nx/16), using 3→64 tanh→511 plus 511 logits. Float64 restriction/prolongation applies the '
            'tail to the remaining physical residual; an actual fine-equation norm check rejects a worsening proposal. '
            'The inference tail evaluates no numerical inverse. Above 16 cells on any axis, it returns the parent '
            'prediction unchanged. Two pressure corrections and three learned global blocks are internal solver work, '
            'not multiple simulation outputs; this component does not advance velocity or tracer.')
        row['purpose']=('Reduce the small-grid pressure-error tail by fitting a final global correction while preserving '
            'the selected parent, its CNN and both routed coarse heads. '
            +('This is the first retained tail calibration in the three-seed comparison.' if label=='C128' else
              'This is a separate initialization/sampling-seed replicate of the tail calibration, not continuation of the completed earlier fit.'))
        row['experiment']=(f'The executed recipe runs head_train.py with calibration.json: 10,000 CUDA Adam updates, '
            f'batch 256 with replacement, initialization seed {calibration["seed"]} and sampling seed {calibration["sampling_seed"]}. '
            'The rate is 0.001, then 0.0003 at update 6001 and 0.0001 at 9001; gradients are clipped to norm one. '
            'Only five tail tensors train; all parent inventories remain byte-identical. The objective is mean valid-mode '
            'squared log-gain-fraction error plus 0.1 times each example’s worst valid-mode error. Offline inverse-mode '
            'labels cover all 1,728 rectangular geometries with axes 5..16: 1,526 train and 202 development geometries. '
            'The history records 2,560,000 sampled teacher examples and zero spatial CNN forwards during fitting. '
            'No physical trajectory fields train this tail. Inherited scene-feedback/batch-one context fields remain '
            'in the original config; they do not replace this separately selected calibration policy.')
        tail_score=score['worst'];last=history['history'][-1]
        row['outcome']=(f'The recorded final teacher-development objective is {last["teacher_development_loss"]:.17g}; '
            f'it is a modal calibration loss. On the fresh shared 128 consumed pressure requests, this exact {label} '
            f'checkpoint has worst mean-free relative pressure L2 {tail_score["pressure_mean_free_relative_l2"]:.17g} '
            f'and equation relative L2 {tail_score["common_equation_relative_l2"]:.17g}. '
            f'{partition["train"]} requests use teacher-training geometries and {partition["development"]} use '
            'teacher-development geometries. Those partitions describe grid shape, not independent scene selection. '
            'The common pressure requests cover 8/16 cells per metre, small grids and 1/60-second steps. These results '
            'provide no new qualified latency, larger-domain, full-state, multistep or rollout-drift result.')
        row['lesson']=('The three seeds yield similar small-grid pressure tails, but the lower teacher-development loss '
            'of the later replicates does not produce a lower worst pressure-field error on this common diagnostic. '
            'Evaluate modal fitting, pressure-field error and equation residual separately. The grid-count bypass '
            'makes this improvement conditional; it cannot be extrapolated to 32³ or larger domains. This calibration '
            'also supplies no new training coverage for timestep variation, forcing schedules or temporal extrapolation. '
            'Total parameter count, inherited context fields and historical CPU probes do not describe the active '
            'training scope or establish a complete fluid-generator frontier.')
        row['reproduction']=('Retain the exact six typed inputs, all witnessed files, calibration.json and original '
            'train/restore recipes. The teacher manifest binds prepare_tail_teacher.py and all eight label-array '
            'inventories; the generator is an offline sealed-Neumann geometry teacher, separate from physical '
            'Rheon scene generation. Recreate a fresh admitted output rather than overwrite the guarded bank. '
            'The original recipe includes a historical CPU deployment probe on two grids; this source review runs '
            'no model and future reproduction must respect the current GPU-only inference policy. A changed recipe '
            'needs a separately identified execution. Clean managed retraining and complete-generator reproduction '
            'remain unverified; new Rheon physical data belongs to a separate reference epoch.')
        verified_review_receipt(row);rows.append(row);source_sets.append(witness['sources'])
    differences={name for name in source_sets[0] if any(s[name]!=source_sets[0][name] for s in source_sets[1:])}
    if differences!={'config.json','calibration.json'}:raise ValueError('Replicate executable source differs')
    overview=previous['overview_addition'].replace(
        'Five of 205 fit cards now have retrospective source reviews prepared: the shared\n'
        'spectral pressure response, the local transport sampler, the fixed-viscosity\n'
        'parent and the matched C106/C107 pressure students.',
        'Eight of 205 fit cards have retrospective source reviews prepared: the shared\n'
        'spectral pressure response, the local transport sampler, the fixed-viscosity\n'
        'parent, C106/C107 and the three C128/C129/C130 pressure-tail calibrations.')+'''

The C128/C129/C130 pressure tails each retain 94,956 frozen parent
parameters and fit 33,982 new tail parameters on 1,526 training geometries,
with 202 teacher-development geometries. The actual head-only recipe supersedes
inherited scene-training context for this fit. The tail is bypassed above 16
cells on any axis. Its small-grid pressure scores therefore remain a distinct
scope from complete generators, larger domains and rollout drift.
'''
    output=dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),
        classification='authored_retrospective_interpretation_original_evidence_preserved',
        rows=previous['rows']+rows,overview_addition=overview,teacher_array_inventories_verified=8,
        replicate_executable_sources_equal=True,replicate_source_differences=sorted(differences),
        new_model_forwards=0,new_parameter_updates=0,native_project_written=False,payload_copies=0)
    OUTPUT.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='eight_source_reviews_ready_for_guarded_note_preparation',
        new_fits=[r['fit_item_id'] for r in rows],teacher_geometry_partition=rows[0]['common_pressure_teacher_geometry_partition'])))


if __name__=='__main__':run()
