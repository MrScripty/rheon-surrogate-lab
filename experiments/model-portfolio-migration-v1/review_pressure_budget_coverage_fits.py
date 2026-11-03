"""Review C38--C40 budget, manufactured coverage and intermediate objectives."""
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from captured_pressure_review import load_fit
from prepare_method_review_revisions import revised_card, verified_review_receipt
from review_water_learning_sequence import binding,canonical,population_identity,read,require

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT=ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT=DOC/'reports/captured-pressure-budget-coverage-method-reviews-01.json'


def audit_bank(facts):
    spec=facts['config']['analytical_feedback'];saved=facts['history']['analytical_feedback'];grids=spec['grids'];updates=facts['config']['training']['updates']
    require(spec['patterns']==32 and spec['seed']==2026101507 and spec['weight']==.1
            and spec['sampling']=='uniform-balanced-grid' and len(saved['grids'])==len(grids), 'Captured manufactured policy differs')
    rng=np.random.Generator(np.random.PCG64(spec['seed']))
    grid_order=np.concatenate([rng.permutation(len(grids)) for _ in range(math.ceil(updates/len(grids)))])[:updates]
    order=(grid_order*128+rng.integers(128,size=updates,dtype=np.int64)).astype('<i8')
    require(hashlib.sha256(order.tobytes()).hexdigest()==saved['sampling_order_sha256']
            and saved['grid_exposures']==np.bincount(grid_order,minlength=len(grids)).tolist()
            and saved['additional_examples']==updates, 'Manufactured proposal order/exposure differs')
    amplitude=saved['amplitude_rms_pa']
    for grid,evidence in zip(grids,saved['grids'],strict=True):
        shape=tuple(reversed(grid['counts_xyz']));bank=evidence['bank'];high=tuple(n-1 for n in shape)
        require(evidence['grid']==grid and tuple(bank['shape_zyx'])==shape, 'Manufactured grid counts/density differ')
        fixed=[(0,0,1),(0,1,0),(1,0,0),(0,1,2),(3,2,1),(4,4,4),(0,0,high[2]),high]
        fixed=[tuple(min(k,n-1) for k,n in zip(f,shape)) for f in fixed]
        rng=np.random.Generator(np.random.PCG64(spec['seed']));descriptions=[]
        for index in range(32):
            if index<8:components=[(fixed[index],1.)]
            else:
                components=[]
                for _ in range(1 if index%2 else 3):
                    frequencies=tuple(int(rng.integers(n)) for n in shape)
                    if not any(frequencies):frequencies=(0,0,1)
                    components.append((frequencies,float(rng.uniform(.5,1.5))))
            descriptions.extend(dict(components_zyx=components,amplitude_pa=amplitude*factor) for factor in (-2.,-.5,.5,2.))
        within=rng.integers(128,size=1,dtype=np.int64).astype('<i8')
        require(canonical(descriptions)==canonical(bank['samples'])
                and hashlib.sha256(canonical(descriptions)).hexdigest()==bank['bank_description_sha256']
                and hashlib.sha256(within.tobytes()).hexdigest()==bank['sampling_order_sha256'], 'Manufactured descriptors or per-bank draw differs')
    projected=128*(32*sum(math.prod(g['counts_xyz']) for g in grids)+24*len(grids))
    require(saved['resident_tensor_bytes']==projected<=spec['cache_max_bytes']==805306368
            and hashlib.sha256(canonical(saved['grids'])).hexdigest()==saved['bank_description_sha256'], 'Manufactured cache budget/bank identity differs')
    return dict(actual_manufactured_grids=len(grids),actual_manufactured_examples=128*len(grids),
        additional_manufactured_proposals=updates,actual_grid_exposures=saved['grid_exposures'],
        manufactured_descriptors_and_sampling_orders_rederived=True)


def run():
    require(not OUTPUT.exists(),'Preserve original source-review report')
    index_path=PROJECT/'notes/portfolio-index.json';index_raw=index_path.read_bytes();index=read(index_path)
    work=read(ROOT/'runs/model-portfolio-migration-v1/discovery/model-worklist-05.json')
    material={r['id']:r for r in read(ROOT/'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')['models']}
    score=read(DOC/'reports/pressure-common-score-01.json')['models'];timing_path=DOC/'reports/pressure-portfolio-GPU-timing-01.json';timing=read(timing_path)
    baseline_report=DOC/'reports/captured-multigrid-recurrence-method-reviews-01.json'
    baseline=next(row for row in read(baseline_report)['rows'] if row['named_checkpoint']=='C33')
    for spec in (baseline['history'],baseline['training_population'],baseline['original_receipt']):
        require(binding(ROOT/spec['path'])==spec,'Original reviewed C33 baseline differs')
    initial=read(ROOT/baseline['history']['path'])['checkpoints']['checkpoint-000.npz']['inventory']
    population=population_identity(read(ROOT/baseline['training_population']['path'])['parents'])
    purposes={
        'C38':'Extend the two-correction cohort policy to 10,000 optimizer updates, moving learning-rate drops later while keeping eight manufactured grids.',
        'C39':'Add three mixed odd/even rectangular manufactured grids at 16 cells/metre to the 5,000-update two-correction cohort policy.',
        'C40':'Train the first of two pressure corrections for residual contraction in addition to the final canonical objective, without changing deployed correction count.'}
    lessons={
        'C38':'Both common worst errors improve over C33, but training budget and learning-rate schedule both change. More optimizer work does not imply more inference corrections or extra physical trajectories.',
        'C39':'Adding manufactured shapes does not improve these common worst errors over C33. Eleven grids redistribute 5,000 proposals into 454/455 exposures each instead of 625; this is not an equal-exposure pure shape comparison. Grid capacity 16 is a validation bound, not 16 executed grids or new CFD scenes.',
        'C40':'Intermediate contraction improves both common worst metrics over C33 at the same physical/manufactured sample budget and two-correction deployment. The larger logged training loss includes an extra term; final validation remains canonical. This does not establish complete-step or rollout improvement.'}
    rows=[];notes=[];orders={}
    for number in (38,39,40):
        name=f'C{number}';f=load_fit(name,work=work,material=material,index=index,score=score);meta=f['metadata'];config=f['config'];history=f['history']
        require((meta['actual_train_trajectories'],meta['actual_train_frames'],meta['actual_validation_trajectories'],meta['actual_validation_frames'])==(16,32,64,128)
                and meta['actual_parameter_count']==370016 and meta['initialized_tensor_count']==19
                and f['inventory']==initial and population_identity(f['split']['parents'])==population,
                'Matched physical population/linear initial state differs')
        require(config['conditioning']=='physical-residual' and config['normalization']=='none'
                and config['correction']==dict(activation='linear',steps=2) and config['output_scale']=='previous-pressure-rms'
                and config['training']['seed']==20261015 and config['training']['sampling_seed']==2026102501,
                'Captured two-correction physical policy differs')
        require(config['training']['updates']==(10000 if number==38 else 5000)
                and config['training']['schedule']==[dict(learning_rate=.0001,update=6000 if number==38 else 3000),dict(learning_rate=.00001,update=9000 if number==38 else 4500)],
                'Actual optimizer budget/schedule differs')
        bank=audit_bank(f);require(bank['actual_manufactured_grids']==(11 if number==39 else 8),'Actual manufactured grid population differs')
        model_source=(f['project']/'model.py').read_text();physical_source=(f['project']/'physical_residual.py').read_text();trainer=(f['project']/'train.py').read_text()
        require("q=residual*(rho/h).reshape(-1,1,1,1,1)" in physical_source
                and "module.register_parameter('bias',None)" in physical_source, 'Executed physical/linear input transform differs')
        if number==40:
            require(config['objective']=='canonical-intermediate-contraction'
                    and 'INTERMEDIATE_CONTRACTION_WEIGHT=.1' in physical_source
                    and "initial-owner.full_operator(pressure-sample['base']" in physical_source
                    and 'physical_residual.intermediate_contraction(pressures[:-1],sample,owner)' in model_source
                    and 'return final+physical_residual.INTERMEDIATE_CONTRACTION_WEIGHT*intermediate' in model_source
                    and 'extra=model.training_loss(network,analytical_sample,normalization,' in trainer,
                    'Intermediate contraction must apply to actual and manufactured fitting, not inference')
        else:require(config['objective']=='canonical','Unexpected training-only objective')
        common=score[name]['worst'];perf=timing['models'][name]['summary']
        experiment=(f'{config["training"]["updates"]:,} whole-network Adam updates, batch 1, initial rate 0.001, schedule {config["training"]["schedule"]}, '
            'model seed 20261015, physical sample seed 2026102501, clipping 1.0. The same 16 physical trajectories / 32 frames at 8/16 cells per metre '
            'supply training; axes are at most 16 cells and physical dt is 1/60 s. Validation uses 64 trajectories / 128 frames. '
            f'Manufactured feedback has {bank["actual_manufactured_examples"]} fields on {bank["actual_manufactured_grids"]} grids, '
            f'{bank["additional_manufactured_proposals"]:,} additional proposals at weight 0.1, train-cohort correction-RMS amplitude, and balanced grid exposures {bank["actual_grid_exposures"]}. '
            'Base manufactured grids include 16/24/32/40-cubed and 24x16x12 at 16 cells/metre, plus 17x13x9, 33x17x9 and 9x25x13 at 8 cells/metre. '
            + ('Three additions are 16x23x25 and its cyclic axis permutations at 16 cells/metre. ' if number==39 else '')
            + ('Training adds 0.1 times the mean relative squared nonfinal equation residual to both actual and manufactured final losses; with two corrections this supervises the first correction. ' if number==40 else '')
            + 'Manufactured examples are pressure-operator fields, not new fluid trajectories.')
        row=dict(meta,purpose=purposes[name],experiment=experiment,lesson=lessons[name],
            method='A one-channel 370,016-parameter linear pressure-correction network with 19 weight tensors, no GroupNorm/SiLU/biases despite inherited cubic-groupnorm-v3 naming. '
                'F64 equation residual is physically scaled by rho/h and train-only q RMS before F32 convolutions. Two shared-weight corrections recompute the F64 operator residual; '
                'previous-pressure RMS scales cell-zero-gauged corrections before F64 addition. Physical dt is validated but not a separate CNN channel. '
                'The sealed uniform cubic-voxel Neumann policy returns pressure only, with no obstacle/free-surface mask or physical simulation advancement.',
            outcome=f'Last logged training objective {history["history"][-1]["loss"]:.17g}; retained final validation objective {history["validation"]["final"]:.17g}. '
                f'Common 128-request worst mean-free pressure relative L2 {common["pressure_mean_free_relative_l2"]:.17g}; equation relative L2 {common["common_equation_relative_l2"]:.17g}. '
                f'Complete GPU pressure-call median of case medians {perf["median_case_median_ms"]:.17g} ms; worst case median {perf["worst_case_median_ms"]:.17g} ms. '
                'The interval includes CPU input validation/packing, transfers, pressure work, independent CPU output and synchronization. It excludes setup/scoring and is not a physical-step or drift measurement.',
            reproduction='Use the original captured typed recipe/config/runtime, source witness, input slots, physical split/order, train-only normalization and seeded initial inventory. '
                'Recreate the manufactured fields from explicit shapes/densities, train-cohort correction RMS and private RNG seed, preserving proposal counts and objective. '
                'The historical trainer includes CPU deployment/parity forwards; qualify a GPU-only reproducer before rerunning under current policy. Clean retraining/optimizer resume/new-Rheon regeneration are not established here.',
            retained_timing=binding(timing_path),configured_pressure_corrections=2,matched_C33_initial_weights_and_physical_population=True,
            actual_train_cells_per_metre=[8,16],actual_validation_cells_per_metre=[8,16],**bank)
        receipt=verified_review_receipt(row);note=PROJECT/row['note_path'];raw=note.read_bytes();new,suffix=revised_card(raw,row)
        require(raw.count(receipt)==new.count(receipt)==1 and new.endswith(suffix),'Original receipt/configuration suffix changed')
        rows.append(row);notes.append(binding(note));orders[name]=f['sample_order']
    require(np.array_equal(orders['C39'],orders['C40']) and np.array_equal(orders['C38'][:5000],orders['C39']),
            'Claimed matching physical sample-order prefixes differ')
    require(index_path.read_bytes()==index_raw and all(binding(ROOT/n['path'])==n for n in notes),'Native notes/index changed during source review')
    require('torch' not in sys.modules,'Source review imported a neural runtime')
    value=dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=3,
        new_model_forwards=0,native_mutations=0,new_reference_generation=False,original_note_bindings=notes,
        clean_reproduction_qualified=False,baseline=binding(baseline_report),
        overview_addition='\n\n## Longer fitting, manufactured coverage and intermediate contraction\n\nC38–C40 keep the same16-trajectory physical cohort and two-correction pressure policy while varying update/schedule budget, manufactured domain coverage, or nonfinal residual supervision. Larger manufactured banks redistribute fixed sample exposure; an extra training objective changes logged loss semantics. These are pressure-only comparisons, not new physical trajectory or rollout qualification.\n',
        producer=binding(Path(__file__)),helpers=[binding(ROOT/'experiments/model-portfolio-migration-v1/captured_pressure_review.py'),binding(ROOT/'experiments/model-portfolio-migration-v1/review_water_learning_sequence.py')])
    OUTPUT.write_bytes(canonical(value)+b'\n');print('Three budget/coverage/intermediate-objective fits source-reviewed; no native mutation or neural execution.')


if __name__=='__main__':run()
