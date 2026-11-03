"""Consolidate completed bounded experiments; refuse incomplete worker custody."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha

ARMS=['domain-small-control','domain-expanded','teacher-coarse','teacher-half','teacher-quarter',
      'numerical-coarse','sparse-anchors','learned-coarse','spacing-data-control','spacing-zero-feature',
      'spacing-conditioned','coarse-head-rate']


def report():
    base=ROOT/'runs/pressure-methods-v1'
    docs=ROOT/'docs/plans/pressure-methods-v1'
    if (docs/'results.json').exists()or(docs/'results.md').exists():
        raise ValueError('Final evidence already published; inspect rather than overwrite')
    controllers=[base/name/'execution.json'for name in ['teacher-spacing-sequence-01','hybrid-spacing-sequence-01',
        'fit-evaluation-sequence-01','extra-evaluation-sequence-01','trained-native-sequence-01',
        'coarse-head-rate-controller-01','frozen-native-followups-01','quiet-timing-01']]
    controllers.append(docs/'drift-execution.json')
    for path in controllers:
        result=read(path)
        if result['status']!='completed' or result.get('active'):
            raise ValueError('Required controller has not completed: '+str(path))
        if any(not row.get('reaped',True)or row.get('returncode',0)!=0 for row in result.get('workers',[])):
            raise ValueError('Worker failure or unresolved cessation: '+str(path))
    quiet_start=read(base/'quiet-timing-01/quiet-start.json')
    if (base/'frozen-native-followups-01/execution.json').stat().st_mtime>quiet_start['unix_time']:
        raise ValueError('Additional native followups overlap the quiet timing; repeat timing after cessation')
    fits=[]
    evaluations={}
    for arm in ARMS:
        study=base/('domain-expanded-study-02'if arm=='domain-expanded'else arm+'-study-01')
        path=study/'receipt.json'
        result=read(path)
        if (result['status']!='completed'or result['optimizer_updates']!=1600
                or len(result['artifact_parity'])!=5 or not all(result['artifact_parity'].values())
                or result['training_receipt']['optimizer_updates']!=1600
                or result['restore_receipt']['optimizer_updates']!=0
                or result['restore_receipt']['execution']['cuda_initialized']
                or sha(study/'restore/weights.npz')!=result['weights_sha256']):
            raise ValueError('Managed training/restore custody differs: '+arm)
        evaluation_path=base/'evaluation'/(arm+'-01/results.json')
        evaluation=read(evaluation_path)
        if (evaluation['status']!='completed'or evaluation['optimizer_updates']!=0
                or evaluation['cuda_initialized']or evaluation['threads']!=1 or len(evaluation['rows'])!=700
                or evaluation['training_receipt_sha256']!=sha(path)):
            raise ValueError('Complete 140-request evaluation required: '+arm)
        for population in ['consumed128','domain32','domain64','domain-odd']:
            numerical=next(r for r in evaluation['summary']if r['population']==population and r['policy']=='full-numerical')
            if numerical['qualified']!=numerical['cases']:
                raise ValueError('Independent numerical equation control failed')
        evaluations[arm]=evaluation
        fits.append(dict(arm=arm,optimizer_updates=1600,training_run_id=result['training_run_id'],restore_run_id=result['restore_run_id'],
            weights_sha256=result['weights_sha256'],artifact_parity=result['artifact_parity'],
            receipt=str(path.relative_to(ROOT)),receipt_sha256=sha(path),
            evaluation=str(evaluation_path.relative_to(ROOT)),evaluation_sha256=sha(evaluation_path),summary=evaluation['summary']))
    native=[]
    paths=list((base/'drift-matrix-01').glob('*/results.json'))
    paths+=list((base/'trained-native-sequence-01').glob('*/results.json'))
    paths+=list((base/'frozen-native-followups-01').glob('*/results.json'))
    paths+=list((base/'coarse-head-rate-controller-01').glob('native-*/results.json'))
    paths+=[base/'before-six-followup-01/results.json',base/'before-six-long-followup-01/results.json']
    for path in paths:
        result=read(path)
        if result['status']not in ['completed','pressure_rejection']or not result['native_process']['reaped']:
            raise ValueError('Native result has unresolved custody: '+str(path))
        maxima={key:max((p[key]for p in result['points']),default=None)for key in result['thresholds']}
        native.append(dict(name=str(path.parent.relative_to(base)),status=result['status'],kind=result['kind'],
            checkpoint=result['checkpoint'],passes=result['passes'],interval=result['interval'],scene=result['scene'],
            accepted_steps=result['completed_post_warmup_steps'],maxima=maxima,first_crossings=result['first_crossings'],
            network_calls=result['network_calls'],numerical_corrections=result['numerical_corrections'],
            elapsed_s=result['elapsed_s'],candidate_track_s=result['summed_candidate_track_s'],
            results=str(path.relative_to(ROOT)),results_sha256=sha(path)))
    six_long=read(base/'before-six-long-followup-01/results.json')
    if six_long['status']!='completed'or six_long['completed_post_warmup_steps']!=3585:
        raise ValueError('The promising six-pass policy still needs its declared long followup')
    quiet=[]
    for name in ['C53','C44','C37','numerical-coarse','coarse-head-rate']:
        path=base/'quiet-timing-01'/(name+'.json')
        result=read(path)
        if result['status']!='completed'or result['threads']!=1 or result['cuda_initialized']:
            raise ValueError('Quiet comparison incomplete')
        if any(len(row['timing_samples_s'])!=3 for row in result['rows']):
            raise ValueError('Three retained matched timing samples required')
        for row in result['summary']:
            if row['ordering']=='full-numerical'and name!='C53':continue
            quiet.append(dict(model=name,**row,source=str(path.relative_to(ROOT)),source_sha256=sha(path)))
    frontier=[row for row in quiet if not any(q['median_policy_ms']<=row['median_policy_ms']
        and q['worst_relative_equation_residual']<=row['worst_relative_equation_residual']
        and(q['median_policy_ms']<row['median_policy_ms']or q['worst_relative_equation_residual']<row['worst_relative_equation_residual'])for q in quiet)]
    spacing={}
    for name in ['C53',*ARMS]:
        path=(base/'coarse-head-rate-controller-01/spacing-followup.json'if name=='coarse-head-rate'
              else base/'extra-evaluation-sequence-01'/(name+'-spacing.json'))
        result=read(path)
        if result['status']!='completed'or len(result['rows'])!=96:
            raise ValueError('All 24 independent spacing requests and four pass counts required')
        spacing[name]=dict(summary=result['summary'],source=str(path.relative_to(ROOT)),sha256=sha(path))
    def selected(arm,population='consumed128',passes=8):
        return next(r for r in evaluations[arm]['summary']if r['population']==population and r['policy']=='neural-'+str(passes))
    head=selected('coarse-head-rate')
    baseline_head=selected('learned-coarse')
    head_result=('Improved standard-request worst residual'if head['worst_relative_equation_residual']<baseline_head['worst_relative_equation_residual']
                 else 'Did not improve standard-request worst residual')
    cohorts=[
        dict(id=1,name='larger-training-domains',outcome='Mixed: strong 32/64 cube gains; small and larger-rectangular regressions',
             controls=['domain-small-control','domain-expanded'],followup='Independent 32-cube native trajectory: expanded completes 113 post-warmup steps; small-only rejects first request.'),
        dict(id=2,name='frozen-model-numerical-correction',outcome='Promising: C53 before-six, C44 before/after-seven and C37 before/after-ten qualify all standard requests',
             followup='Intermediate-pass sweeps, separate native checks, six-pass 3600-step trajectory, matched quiet timing.'),
        dict(id=3,name='finer-teacher-timesteps',outcome='Small matched improvement at half-step; no monotonic quarter-step improvement and no gain over original C53 at eight passes',
             controls=['teacher-coarse','teacher-half','teacher-quarter'],limitation='Fixed-equation pressure requests from finer teacher states; not a learned fine-integrated coarse full-state transition.'),
        dict(id=4,name='numerical-coarse-neural-fine',outcome='Promising: 103/128 at four passes, 128/128 at five; eight-pass worst residual about 9.07e-10',
             controls=['domain-expanded','numerical-coarse'],followup='Intermediate pass sweep and native budget tests; transfers, coarse solve and all fine stages timed.'),
        dict(id=5,name='sparse-numerical-anchors',outcome='No general advantage from this exact-anchor construction; standard and large-grid regressions remain',
             controls=['domain-expanded','sparse-anchors'],limitation='Anchors require a global transform/all-mode solve and selected inverse; their full cost is included.'),
        dict(id=6,name='revised-learned-coarse-fine',outcome='Positive gated low-mode branch is finite and trainable; '+head_result,
             controls=['domain-expanded','learned-coarse','coarse-head-rate'],followup='Separate 1000x rate for the 63 newly added logits; same fine rates, loss, data, initialization and inference.'),
        dict(id=7,name='spacing-training-and-conditioning',outcome='Explicit spacing does not improve independent-spacing worst error over its matched zero-feature control',
             controls=['spacing-data-control','spacing-zero-feature','spacing-conditioned'],followup='Twelve independently reserved validation trajectories, 24 requests at 4/8/16/32 cells per metre.'),
        dict(id=8,name='periodic-numerical-correction',outcome='Eight-pass periodic policies remain below declared drift limits over the tested horizons; periodic correction does not rescue rejected four-pass requests',
             followup='Three development scenes at 512 steps; first-scene native/full-numerical/neural/periodic controls extended to 3600 steps.')]
    result=dict(schema='rheon-pressure-methods-campaign-results-v1',status='completed_bounded_experiments',cohorts=cohorts,
        managed_fits=fits,total_optimizer_updates=1600*len(fits),native=native,quiet_timing=quiet,
        quiet_frontier_including_full_numerical=frontier,spacing=spacing,
        controller_evidence={str(p.relative_to(ROOT)):sha(p)for p in controllers},
        declared_equation_tolerance=1e-6,declared_drift_relative_l2_tolerances=dict(pressure=.01,velocity=.01,tracer=.01),
        scope='Development research on sealed uniform cubic water grids. No original reserved-test scenes, model promotion or production deployment.',
        runtime_scope='Quiet prepared-pressure timing includes all numerical/neural stages and final residual. Native times are instrumented dual-track diagnostic costs, not single-track production timings.',
        original_chart='Historical chart and its different timing scope remain unchanged.',
        limits=['Single training seed per new arm; these are bounded controlled comparisons, not exhaustive optimization.',
                'Matched optimizer-update budgets; tier sampling changes per-parent exposure and forward-cell work when larger domains are added.',
                '64-cube model requests evaluated offline; coupled larger-domain followup uses 32-cube. Production 256/512/1024 deferred.',
                'Arbitrary geometry, obstacles, free surfaces and variable mass density are outside the current model/operator contract.'],
        source_sha256=sha(__file__),model_promotion=False,untouched_scenes_consumed=0)
    lines=['# Eight-cohort pressure-method results','',
        'All eight bounded cohorts and their selected follow-ups are complete. Twelve managed fits each ran 1,600 CUDA updates and a fresh CPU restore, with exact parity of the five retained artifacts. C53 remains the baseline; no model was promoted.','',
        'The full spectral numerical control is the strongest prepared-pressure baseline on these sealed, uniform grids. The neural/hybrid candidates below are useful comparisons against the existing learned policies. These timings include numerical work and final residual evaluation; the historical chart uses a different scope and was left unchanged.','',
        '| Cohort | Result |','|---|---|']
    for row in cohorts:lines.append('| '+str(row['id'])+'. '+row['name']+' | '+row['outcome']+' |')
    lines+=['','Eight-pass accuracy on the common populations (qualified requests / total, equation tolerance 1e-6):','',
        '| Training arm | Standard 128 | New 32³ | New 64³ | Larger rectangular | Worst standard residual |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        stats=[selected(arm,p)for p in ['consumed128','domain32','domain64','domain-odd']]
        lines.append('| '+arm+' | '+' | '.join(f"{r['qualified']}/{r['cases']}"for r in stats)+' | '+f"{stats[0]['worst_relative_equation_residual']:.3g}"+' |')
    lines+=['','Selected quiet comparisons: same CPU, three repetitions per request, after owned campaign workers ceased.','',
        '| Model/policy | Neural passes | Qualified / 128 | Worst equation residual | Median pressure-policy ms |','|---|---:|---:|---:|---:|']
    chosen=[]
    for row in quiet:
        if ((row['model']=='C53'and(row['ordering']=='full-numerical'or row['ordering']=='none'and row['passes']in[7,8]
             or row['ordering']=='before'and row['passes']in[6,7]))
            or row['model']=='C44'and row['ordering']in['before','after']and row['passes']==7
            or row['model']=='C37'and row['ordering']=='before'and row['passes']==10
            or row['model']=='numerical-coarse'and row['passes']in[5,6]):chosen.append(row)
    head_options=[row for row in quiet if row['model']=='coarse-head-rate'and row['ordering']=='none'and row['qualified']==128]
    if head_options:chosen.append(min(head_options,key=lambda r:r['median_policy_ms']))
    for row in chosen:
        lines.append(f"| {row['model']} / {row['ordering']} | {row['passes']} | {row['qualified']}/128 | {row['worst_relative_equation_residual']:.3g} | {row['median_policy_ms']:.3f} |")
    maxima={key:max(p[key]for p in six_long['points'])for key in six_long['thresholds']}
    lines+=['',f"Frozen C53 before-six completed all 3,585 post-warmup steps in a 3,600-step simulation (60 seconds total). Its maximum pressure/velocity/tracer relative L2 drift was {maxima['pressure_trajectory_relative_l2']:.3g} / {maxima['velocity_trajectory_relative_l2']:.3g} / {maxima['tracer_trajectory_relative_l2']:.3g}. No declared 1% drift threshold was crossed. This establishes a tested lower bound, not a prediction of eventual failure time.",'',
        'The finer-teacher experiment preserves a fixed model pressure equation and uses matched coarse-history initial guesses. Its rescaled instantaneous pressure requests are not a claim of exact fine-integrated coarse flow labels. The coarse-control fit exactly reproduces the small-only control weights.','',
        'The spacing ablation varies cells per metre separately from cell counts; mass density remains 1000 kg/m³. Both two-channel arms have identical initialization, with the added convolution column zeroed. All independent-spacing scenes are excluded from fitting.','',
        'These are controlled development results from one training seed per new arm. The larger-data arms use balanced spatial tiers under the same optimizer-update budget, changing per-parent exposure and total forward-cell work. The full numerical control uses separable dense cosine transforms; no production FFT scaling claim is made. Coupled larger-domain testing here uses 32³, while the 64³ pressure requests and training gradients were tested offline. Arbitrary geometry and production sizes 256³–1024³ remain future work. Native runtime includes two tracks, diagnostic reference work and IPC; a single-track application benchmark is still needed before a deployment speed claim.','',
        '[Full results and hashes](results.json) · [Execution ledger](execution-ledger.md) · [Domain comparison](domain-training-results.md) · [Drift execution](drift-execution.json)']
    write(docs/'results.json',result)
    (docs/'results.md').write_text('\n'.join(lines)+'\n')
    campaign=read(docs/'campaign.json')
    campaign.update(status='completed_bounded_experiments',results='results.json',summary='results.md',next_required_work=None)
    for row in campaign['cohorts']:
        row.update(status='completed_and_tested',outcome=cohorts[row['id']-1]['outcome'])
    write(docs/'campaign.json',campaign)
    print('Consolidated all eight completed and tested bounded cohorts',flush=True)


if __name__=='__main__':report()
