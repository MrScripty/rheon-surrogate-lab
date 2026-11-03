"""Audit original study requirements without rewriting the frozen registration."""
import time
from study_support import ROOT, HERE, BASE, DOCS, read, write, sha


def run():
    registration = read(DOCS/'registration.json')
    rows = {row['id']:dict(id=row['id'],requirement=row['requirement'],status='pending',evidence=[]) for row in registration['requirements']}
    def record(identifier,status,note,*paths):
        files=[]
        for relative in paths:
            path=ROOT/relative
            if not path.is_file():
                rows[identifier].update(status='pending',note='Evidence not yet available: '+relative)
                return
            files.append(dict(path=relative,sha256=sha(path)))
        rows[identifier].update(status=status,note=note,evidence=files)
    b='runs/full-ai-next-priorities-v1/'
    d='docs/plans/full-ai-next-priorities-v1/'
    record('A-restore','passed','Exact original checkpoint reproduced on all128requests within registered numerical reproduction tolerance',b+'pressure-baseline-03/summary.json',d+'baseline-evidence.json')
    record('A-timing','passed','Five balanced serial repeats, one thread and recorded warmup',b+'pressure-baseline-03/summary.json')
    record('A-profile','passed','Clean final profile plus residual-stage and coarse-gate diagnostics',b+'pressure-profile-03/profile.json',b+'C107-residual-diagnostics-01.json',b+'C107-coarse-gate-diagnostics-01.json')
    record('A-execution','passed','Cache/buffer/layout and qualified compiled comparison retained, including negative timing outcomes',b+'pressure-baseline-03/summary.json',b+'pressure-compiled-timing-01/summary.json',b+'C107-compiled-qualification-01.json')
    record('A-four-fits','passed','All four1200-update managed fits,600/1200snapshots and exact CPU restores complete',
           b+'pressure-evaluation-01/a-control/checkpoint-1200.json',b+'pressure-evaluation-01/a-head/checkpoint-1200.json',
           b+'pressure-evaluation-01/a-tail/checkpoint-1200.json',b+'pressure-evaluation-01/a-combined/checkpoint-1200.json',d+'a2-screen-decision.json')
    record('A-tail','passed','Train-only initial ranking, capped multiplier4 and preserved tiers',b+'pressure-profile-02/train-ranking.json')
    record('A-selection','passed','Three replicated learned heads qualified and frozen before independent pressure capture',d+'pressure-finalist-selection.json',d+'routed-head-registration.json')
    record('A-architecture','passed','Both bounded architecture fits completed; neither improved accuracy',b+'architecture-training-batch-01/execution.json',b+'pressure-architecture-evaluation-01/a-redistributed/summary.json',b+'pressure-architecture-evaluation-01/a-one-stage/summary.json')
    record('A-seeds','passed','Three actual learned-head fits and CPU restores; continuation variance from fixed C107',b+'a-head-calibration-pair-01/receipt.json',b+'a-head-calibration-s2-pair-01/receipt.json',b+'a-head-calibration-s3-pair-01/receipt.json')
    record('A-fresh','passed','All16independent scenes,1024requests and unchanged-gate native attempts; native rejection retained',b+'pressure-reserved-batch-01/execution.json',b+'pressure-fresh-data-01/bank.json',b+'pressure-fresh-timing-01/summary.json')
    record('A-target','pending','Read separate historical/fresh outcomes and actual latency; native acceptance is a distinct unmet criterion',b+'pressure-fresh-timing-01/summary.json')
    fresh=BASE/'pressure-fresh-timing-01/summary.json'
    if fresh.exists():
        selection=read(DOCS/'pressure-finalist-selection.json')
        timing=next(row for row in read(fresh)['rows'] if row['identifier']==selection['primary_model']and row['population']=='small')
        evaluated=read(BASE/'pressure-fresh-evaluation-01'/selection['primary_model']/'summary.json')
        errors=next(row for row in evaluated['prepared_requests']if row['population']=='small')
        rows['A-target'].update(status='accuracy_pass_latency_reported' if errors['worst_equation_error']<=7e-4 else 'failed',
            note=f"Frozen primary fresh worst {errors['worst_equation_error']:.9g}; median CPU {timing['median_ms']:.4f}ms. Strict4ms comparison={timing['median_ms']<=4}; original wording approximately4ms. Native application acceptance remains unmet.")
    confirmation=BASE/'pressure-confirmation-timing-01/summary.json'
    if confirmation.exists():
        primary='a-head-calibration-s2-routed'
        timing=next(row for row in read(confirmation)['rows']if row['identifier']==primary and row['population']=='small')
        path=BASE/'pressure-confirmation-evaluation-01'/primary/'summary.json'
        errors=next(row for row in read(path)['prepared_requests']if row['population']=='small')
        record('A-target','passed'if timing['median_ms']<=4 and errors['worst_equation_error']<=7e-4 else'failed_independent_confirmation',
            f"Compiled median {timing['median_ms']:.4f}ms; independent worst {errors['worst_equation_error']:.9g}, {errors['passing_7e_4']}/768 pass. No combined target claim; native acceptance remains unmet.",
            str(confirmation.relative_to(ROOT)),str(path.relative_to(ROOT)),d+'pressure-confirmation-v1/selection.json')
        record('A-fresh','passed','Both initial and independent compiled-execution confirmation banks evaluated; accuracy outliers and native rejection retained',
            b+'pressure-reserved-batch-01/execution.json',b+'pressure-confirmation-batch-01/execution.json',
            d+'pressure-confirmation-v1/selection.json')
        record('A-execution','passed','Complete-graph and fine-only traces, baseline cache/buffer/layout comparisons; all1152consumed and1024new confirmation outputs checked',
            b+'pressure-whole-qualification-01/a-head-calibration-s2-routed.json',b+'pressure-whole-timing-01/summary.json',
            b+'pressure-confirmation-timing-01/summary.json',b+'pressure-confirmation-evaluation-01/a-head-calibration-s2-routed/summary.json')
    precise=DOCS/'precise-routed-pressure-results.json'
    if precise.exists():
        value=read(precise)
        timing=next(row for row in value['timing']if row['variant']=='materialized_graph'and row['population']=='fresh_small')
        record('A-target','development_target_pass_confirmation_pending',
            f"Precise learned execution passes consumed targets: median{timing['median_ms']:.4f}ms, worst5.5613e-4. New primary fixed before capture; independent confirmation remains outstanding. Earlier independent failure remains retained.",
            d+'precise-routed-pressure-results.json',d+'pressure-confirmation-v2/primary-model-decision.json',
            b+'pressure-confirmation-evaluation-01/a-head-calibration-s2-routed/summary.json')
        record('A-execution','passed','Qualified precise learned routing, all2176consumed requests, five paired repeats; failed float32/mixed attempts and cold/memory costs retained',
            b+'precise-routed-pressure-qualification-01/a-tail8.json',b+'precise-routed-pressure-timing-01/summary.json',
            d+'precise-routed-pressure-results.json')
        record('A-fresh','partial','Previous two pressure banks completed; new learned-tail/precise primary has16new scenes reserved but not captured or confirmed yet',
            d+'pressure-confirmation-v2/pressure-scene-reservation.json',d+'pressure-confirmation-v2/primary-model-decision.json')
        replicas=BASE/'tail8-replication-batch-01/execution.json'
        if replicas.exists():
            value=read(replicas)
            if value['status']=='completed':
                record('A-seeds','passed','Original learned-head recipe and new final-tail recipe each have three managed calibration/continuation seeds and exact CPU restores; all negative outcomes retained',
                    b+'head-replication-batch-01/execution.json',b+'tail8-replication-batch-01/execution.json',
                    b+'a-tail8-pair-01/receipt.json',b+'a-tail8-s2-pair-01/receipt.json',b+'a-tail8-s3-pair-01/receipt.json')
            else:
                record('A-seeds','in_progress','Three original learned-head seeds complete; two registered final-tail continuation seeds triggered, with batch terminal state tracked',
                    b+'head-replication-batch-01/execution.json',b+'tail8-replication-batch-01/execution.json',d+'tail-head-replication-decision.json')
    independent_precise=DOCS/'pressure-confirmation-v2/results.json'
    if independent_precise.exists():
        value=read(independent_precise)
        if value['status']=='completed'and value['primary_independent_target_pass']:
            record('A-target','passed','Fixed primary independently confirms2.133msmedian and4.6963e-4worst on768smallrequests; both continuation seeds also pass.16cubedand32cubedlatency, coldcosts and unchanged native rejection remain separately reported',
                d+'pressure-confirmation-v2/results.json',b+'pressure-precise-confirmation-timing-01/summary.json',
                b+'pressure-precise-confirmation-evaluation-01/a-tail8/summary.json')
            record('A-fresh','passed','New16scene reservation,1024pressure requests and all4models evaluated after source/weight/primary freeze; native attempts retained without fallback',
                d+'pressure-confirmation-v2/selection.json',b+'pressure-precise-confirmation-data-01/bank.json',
                d+'pressure-confirmation-v2/results.json')
            record('A-selection','passed','Original tail seed fixed as primary before capture; completed continuation seeds characterize robustness, with no post-confirmation reselection',
                d+'pressure-confirmation-v2/primary-model-decision.json',d+'pressure-confirmation-v2/selection.json',
                d+'tail-head-replication-results.json')
    record('B-contract','passed','Causal full staggered physical state, known prescribed forcing and explicit physical time',b+'temporal-data-qualification-01/qualification.json')
    record('B-restart','passed','At least eight subsequent native steps reproduced bitwise after restore',b+'temporal-data-qualification-01/qualification.json',b+'temporal-build-02/build-receipt.json')
    record('B-bank','partial','32scenes split before capture;16train+8dev captured,8reserved remain unopened pending qualifying temporal finalists',d+'temporal-scene-reservation.json',b+'temporal-data-01/bank.json')
    record('B-frames','partial','Train/dev initial+128consecutive states; dev extended to240. Reserved240captures deliberately outstanding',b+'temporal-data-01/bank.json',b+'temporal-development-240-01/bank.json')
    record('B-causality','passed','Whole-scene split; no future reference input. Known forcing is available input, not reference state',d+'temporal-scene-reservation.json',b+'temporal-training-view-02/manifest.json',b+'temporal-objective-cuda-04.json')
    record('B-four-models','passed','Corrected primaryK1/2/3/4fits and exact CPU restores completed; all intermediate outputs retained',b+'temporal-training-batch-02/execution.json',d+'temporal-primary-screen-decision.json')
    record('B-loss','passed','Original registered equal-horizon physical objective and train-only normalization; loss followup separately registered',d+'temporal-training-registration-v2.json',b+'temporal-training-view-02/normalization.json')
    record('B-continuation','passed','All four primaries trained on two own-history blocks with deployment-equivalent differentiable encoding',b+'temporal-objective-cuda-04.json',b+'temporal-training-batch-02/execution.json')
    record('B-seeds','conditional_not_triggered','No accuracy-qualified bundled recipe; no confirmed advantage claim; do not replicate the failed screen',d+'temporal-primary-screen-decision.json')
    record('B-rollout','passed','All8development scenes, every output horizon and own-history12/48/240intervals',d+'temporal-primary-screen-decision.json',b+'temporal-development-evaluation-01/b4/summary.json')
    record('B-physical','passed','Equation, field relative/absolute errors and physical diagnostics independently qualified',b+'temporal-metrics-qualification-01.json',b+'temporal-development-evaluation-01/b1/summary.json')
    record('B-timing','passed','Complete block/perstep/tensor timings and separate native/event application cost. Rejections confer no throughput claim',b+'temporal-timing-02/summary.json',b+'native-controls-01/summary.json',b+'temporal-events-01/summary.json')
    record('B-controls','passed','SequentialK1, numerical nativePCG/spectral and C107 actual native controls',b+'temporal-timing-02/summary.json',b+'native-controls-01/summary.json')
    record('B-events','passed','Step13unexpectedforcing invalidates K-1future states; counted discarded work and explicit structural research transport',b+'temporal-events-01/summary.json')
    record('B-target','failed','Small-grid latency passes, but allK2/3/4equation/field and own-history accuracy targets fail.32cubedlatency also fails',b+'temporal-timing-02/summary.json',d+'temporal-primary-screen-decision.json')
    sharing_results=DOCS/'temporal-sharing-results.json'
    if sharing_results.exists():
        sharing=read(sharing_results)
        if sharing['status']=='completed':
            record('B-rollout','passed','All primaries and both endpoint/reuseK2/3/4families evaluated on all8development scenes, every output horizon and own-history12/48/240intervals; all failures retained',
                d+'temporal-primary-screen-decision.json',d+'temporal-sharing-results.json',b+'temporal-sharing-comparison-audit-01.json')
            record('B-continuation','passed','Primary and six sharing fits train through two deployed own-history blocks with matched sampling and audited state/encoder counts',
                b+'temporal-training-batch-02/execution.json',b+'temporal-sharing-comparison-audit-01.json')
            record('B-timing','passed','Two completed five-repeat CPU campaigns; new sharing models include pairedK1/sequential controls, full-block and tensor scopes, timing spread and reaped workers. Native/event controls remain separate',
                b+'temporal-timing-02/summary.json',b+'temporal-sharing-timing-01/summary.json',b+'temporal-sharing-timing-01/execution.json',
                b+'native-controls-01/summary.json',b+'temporal-events-01/summary.json')
            if not sharing['small_grid_finalists']:
                record('B-seeds','conditional_not_triggered','No accuracy-qualified primary or endpoint/reuse bundled recipe; conditioning follow-up remains separately registered. Reserved scenes are not consumed',
                    d+'temporal-primary-screen-decision.json',d+'temporal-sharing-results.json',d+'temporal-step-scaled-registration-01.json')
                record('B-target','failed','All six endpoint/reuse models pass small-grid median speed at0.85..1.47ms/step but fail equation/field and own-history accuracy. Individual latency tails and32cubed also exceed budgets. Conditioning follow-up remains unproven',
                    d+'temporal-sharing-results.json',b+'temporal-sharing-timing-01/summary.json',d+'temporal-step-scaled-registration-01.json')
    conditioning_results=DOCS/'temporal-step-scaled-results.json'
    if conditioning_results.exists():
        conditioning=read(conditioning_results)
        if conditioning['status']=='completed'and not conditioning['accuracy_qualified_small_grid_recipes']:
            record('B-seeds','conditional_not_triggered','No accuracy-qualified primary, sharing or conditioning recipe; matched MSE supervision follow-up active. Reserved scenes remain unopened',
                d+'temporal-sharing-results.json',d+'temporal-step-scaled-results.json',d+'temporal-increment-supervision-registration-01.json')
            record('B-target','failed','All six sharing and both conditioning models fail full-state/own-history accuracy. Sharing small-grid medians pass speed; conditioning CPU latency unmeasured. Matched supervision follow-up does not change evaluation targets',
                d+'temporal-sharing-results.json',d+'temporal-step-scaled-results.json',d+'temporal-increment-supervision-registration-01.json')
    increment_results=DOCS/'temporal-increment-supervision-results.json'
    if increment_results.exists():
        value=read(increment_results)
        if value['status']=='completed'and not value['accuracy_qualified_small_grid_recipes']:
            record('B-seeds','conditional_not_triggered','No qualified primary/sharing/conditioning/MSE recipe; generative curl-readout trial active. Reserved scenes remain unopened',
                d+'temporal-increment-supervision-results.json',d+'temporal-curl-registration-01.json')
            record('B-target','failed','Increment-MSE improves fitting but worst development velocity0.8224 and equation1.1538 still fail; own240tracer2.1956. New weights have no measured CPU latency; curl constraint qualification supplies no accuracy claim',
                d+'temporal-increment-supervision-results.json',d+'temporal-curl-registration-01.json')
    curl_results=DOCS/'temporal-curl-results.json'
    if curl_results.exists():
        value=read(curl_results)
        if value['status']=='completed'and not value['accuracy_qualified_small_grid_recipe']:
            record('B-seeds','conditional_not_triggered','No accuracy-qualified full-state recipe including the completed curl representation. Frozen pressure-expert port is component preparation only; reserved scenes remain unopened',
                d+'temporal-curl-results.json',d+'temporal-pressure-expert-reuse-plan.md')
            record('B-target','failed','Curl reduces divergence but worsens worst single-block velocity to1.2467 and own240tracer to2.3915. No new CPU latency measured; full-state accuracy remains unmet',
                d+'temporal-curl-results.json',d+'temporal-pressure-expert-reuse-plan.md')
    expert_results=DOCS/'temporal-pressure-expert-results.json'
    if expert_results.exists():
        value=read(expert_results)
        if value['status']=='completed'and not value['accuracy_qualified_small_grid_recipe']:
            record('B-target','failed','Causal full-state pressure-expert follow-up completes the bounded fit, exact CPU restore and complete evaluation but misses accuracy. New weights have no measured CPU latency; inherited training is explicit',
                d+'temporal-pressure-expert-results.json')
            record('B-seeds','conditional_not_triggered','No full-state recipe including the completed pressure-expert follow-up qualifies on development accuracy; reserved scenes remain unopened',
                d+'temporal-pressure-expert-results.json')
    decoder_results=DOCS/'temporal-decoder-results.json'
    if decoder_results.exists():
        value=read(decoder_results)
        if value['status']=='completed'and not value['accuracy_qualified_small_grid_recipe']:
            record('B-target','failed','Combined lead-time/gauge/passive-tracer decoder completes bounded fit, exact CPU restore and every-horizon/own240 evaluation but misses accuracy. New weights have no measured CPU latency',
                d+'temporal-decoder-results.json')
            record('B-seeds','conditional_not_triggered','No full-state finalist including the completed combined decoder qualifies on development accuracy; reserved scenes remain unopened',
                d+'temporal-decoder-results.json')
    coupled_results=DOCS/'temporal-coupled-results.json'
    if coupled_results.exists():
        value=read(coupled_results)
        if value['status']=='completed'and not value['accuracy_qualified_small_grid_recipe']:
            record('B-target','failed','Coupled K2 improves single-block velocity to0.005251 but pressure/equation/tracer and own240 accuracy still fail; all8scenes remain finite and complete. No new CPU latency claim',
                d+'temporal-coupled-results.json',b+'temporal-coupled-evaluation-01/b2-coupled/summary.json')
            record('B-seeds','conditional_not_triggered','No accuracy-qualified coupled finalist; diagnose warm pressure fitting and transport before extensions, replication or reserved capture',
                d+'temporal-coupled-results.json',b+'temporal-coupled-fitting-diagnosis-01.json')
            record('B-rollout','passed','Original requested K1/2/3/4 and both sharing families remain evaluated; coupled K2 adds all8development scenes, every horizon and complete own12/48/240 intervals',
                d+'temporal-sharing-results.json',d+'temporal-coupled-results.json',b+'temporal-coupled-evaluation-01/b2-coupled/summary.json')
    provisional_results=DOCS/'temporal-provisional-results.json'
    if provisional_results.exists():
        value=read(provisional_results)
        if value['status']=='completed'and not value['accuracy_qualified_small_grid_recipes']:
            record('B-target','failed','Both matched4000-update continuations, exact CPU restores and every-horizon/own240 evaluations complete; auxiliary worst small velocity0.005072 still fails and pressure/equation/tracer/long-rollout accuracy also fail. CPU cost measurement is separate',
                d+'temporal-provisional-results.json',b+'temporal-provisional-batch-01/execution.json')
            record('B-seeds','conditional_not_triggered','No qualifying recipe after matched provisional supervision; reserved scenes remain unopened. Native-displacement and reference-velocity scalar controls are excluded from candidate claims',
                d+'temporal-provisional-results.json',d+'tracer-representation-results.json',d+'tracer-reference-velocity-replay-results.json')
            record('B-rollout','passed','OriginalK1/2/3/4and sharing families remain evaluated; both matched coupled continuations add all8development scenes and complete own12/48/240 evaluation',
                d+'temporal-sharing-results.json',d+'temporal-provisional-results.json')
    cost_results=DOCS/'temporal-provisional-cost-results.json'
    if cost_results.exists():
        record('B-timing','passed','Additional complete five-repeat CPU campaign measures parent/control/aux weights on64requests per scope, full blocks and tensors.1920timed requests, separate32cube, per-grid costs, worker preparation/RSS and terminal reaping; failed0-repeat logging setup retained',
            d+'temporal-provisional-cost-results.json',b+'temporal-provisional-timing-02/summary.json',b+'temporal-provisional-timing-02/execution.json',
            b+'temporal-provisional-timing-batch-02/execution.json',b+'temporal-provisional-timing-01/execution.json')
        record('B-target','failed','Both matched continuations fail full-state/own240 accuracy and the mixed small-grid median CPU budget: completeK2blocks5.59..5.75ms, or2.79..2.88ms/step.32cube and tails are slower; no app throughput or promotion claim',
            d+'temporal-provisional-results.json',d+'temporal-provisional-cost-results.json')
    displacement_results=DOCS/'temporal-displacement-results.json'
    displacement_cost=DOCS/'temporal-displacement-cost-results.json'
    if displacement_results.exists()and displacement_cost.exists():
        value=read(displacement_results)
        if value['status']=='completed'and not value['accuracy_qualified_small_grid_recipe']:
            record('B-target','failed','Displacement K2 greatly improves full-state error but still fails: small single-block worst equation0.01642 and own240equation0.01992. Complete CPU blocks6.350ms or3.175ms/step also exceed budget. All8scenes complete; no app throughput/promotion claim',
                d+'temporal-displacement-results.json',d+'temporal-displacement-cost-results.json')
            record('B-seeds','conditional_not_triggered','No accuracy/speed-qualified displacement finalist. Train-only controls identify the neural viscosity head as the next fluid limitation; native controls remain excluded and reserved scenes unopened',
                d+'temporal-displacement-results.json',d+'temporal-displacement-cost-results.json',d+'displacement-component-results.json')
            record('B-timing','passed','Displacement/coupled/auxiliary comparison retains five balanced serial CPU repeats,1920timed requests, full/tensor scopes, exact restored weights, per-grid/setup/RSS evidence and terminal worker reaping',
                d+'temporal-displacement-cost-results.json',b+'temporal-displacement-timing-01/execution.json')
    neural_cost=DOCS/'neural-viscosity-cost-results.json'
    if neural_cost.exists():
        value=read(neural_cost)
        point=next(p for p in value['points']if p['identifier']=='b2-displacement-viscosity-fused'and p['regime']=='small'and p['interval']=='rollout_240')
        if not point['accuracy_pass']:raise ValueError('Expected qualified neural viscosity K2 accuracy')
        record('B-target','partial','Repaired all-neural K2 passes all8development scenes, every field/horizon and own240: worst0.0033941. Exact fused execution preserves3856physical outputs, but median block4.7793ms still misses4ms. Corrected-family K3/K4, replicas and reserved240 remain open',
            d+'neural-viscosity-results.json',d+'neural-viscosity-cost-results.json',b+'fused-neural-viscosity-model-qualification-01.json')
        record('B-seeds','conditional_not_triggered','First K2 accuracy-qualified recipe still misses complete CPU budget; no combined finalist selected and eight reserved scenes remain unopened',d+'neural-viscosity-cost-results.json')
        record('B-timing','passed','Repaired and fused models plus unchanged parent measured in five balanced serial CPU repeats:1920timed blocks, complete/tensor scopes, exact3856-output proof, per-grid/tails/setup/RSS and terminal worker reaping',
            d+'neural-viscosity-cost-results.json',b+'neural-viscosity-timing-01/execution.json',b+'neural-viscosity-timing-batch-01/execution.json')
    fast_cost=DOCS/'fast-preparation-cost-results.json'
    if fast_cost.exists():
        value=read(fast_cost)
        point=next(p for p in value['points']if p['regime']=='small'and p['interval']=='rollout_240')
        record('B-target','partial',f'First all-neural K2 development accuracy/own240 pass: worst{point["worst_error"]:.7g}. Qualified forcing/codec execution preserves3856physical outputs; five-repeat block{point["full_block_ms"]:.4f}ms, combined small-grid target={point["target_met"]}. Corrected-family K3/K4, replicas, reserved240 and application/event cost remain open',
            d+'fast-preparation-cost-results.json',b+'fast-preparation-model-qualification-01.json',d+'fixed-viscosity-family-plan.md')
        record('B-seeds','pending','Current K2 recipe still needs confirmed seed evidence; corrected-family K1/2/3/4 qualification and fitting precede recipe selection/replication. Eight reserved scenes remain unopened',d+'fixed-viscosity-family-plan.md')
        record('B-timing','passed','Qualified forcing/codec execution and original fused control measured in five balanced serial repeats,1280timed complete/tensor blocks; all workers and parent reaped, per-grid/tails/setup/RSS retained',
            d+'fast-preparation-cost-results.json',b+'fast-preparation-timing-01/execution.json',b+'fast-preparation-timing-batch-01/execution.json')
    family_cost=DOCS/'fixed-viscosity-cost-results.json'
    if family_cost.exists():
        value=read(family_cost)
        qualified=value['small_grid_combined_qualified']
        record('B-target','partial',f'All corrected K1/2/3/4 models pass every original accuracy criterion and own240 across8development scenes. Five-repeat full blocks: K2=4.1240ms, K3=5.5742ms, K4=7.4335ms; combined median qualifiers={qualified}. K2precision follow-up, replicas, reserved scenes and application/event measurements remain open',
            d+'fixed-viscosity-family-results.json',d+'fixed-viscosity-cost-results.json')
        record('B-seeds','pending','Corrected K3/K4 now qualify on development accuracy and combined median speed. Finish actual corrected K2precision comparison, then select up to2recipes and replicate each plus B1with2additional seeds. Reserved scenes remain unopened',
            d+'fixed-viscosity-cost-results.json',d+'next-decisions.md')
        record('B-timing','passed','Actual four-model family and sequential2/3/4B1controls complete five balanced serial repeats:3520timed blocks/55responses, exact17360physical outputs, full/tensor scopes, per-grid/tails/setup/RSS and all worker reaping. Sustained application costs remain separate',
            d+'fixed-viscosity-cost-results.json',b+'fixed-viscosity-timing-01/execution.json')
    precision=DOCS/'fixed-viscosity-k2-precision-results.json'
    if precision.exists():
        value=read(precision)
        point=next(p for p in value['points']if p['policy']=='fast-float32-map'and p['regime']=='small'and p['interval']=='rollout_240')
        record('B-target','partial',f'All three requested K now meet development all-field/every-output/own240 accuracy and combined small-grid median speed: corrected K2float32 block={point["full_block_ms"]:.4f}ms, per state={point["median_ms"]:.4f}ms, worst={point["worst_error"]:.7g}; K3/K4float64 blocks5.5742/7.4335ms. Seed confirmations, untouched reserved scenes and sustained application/event measurements remain open; per-grid/tails and32cubedare separate',
            d+'fixed-viscosity-k2-precision-results.json',d+'fixed-viscosity-cost-results.json')
    selection=DOCS/'fixed-viscosity-replication-selection-01.json'
    if selection.exists():
        value=read(selection)
        record('B-seeds','pending',f'Concrete bundled horizons{value["recipe_horizons"]}selected before added-seed training, plus replicated B1control. Two new initialization/sampling seeds each, shared frozen pretraining and unchanged36000physical-target budgets. Original primary seed retained; no reserved capture yet',
            d+'fixed-viscosity-replication-selection-01.json',d+'fixed-viscosity-replication-qualification-protocol-01.json')
    mechanics=BASE/'forecast-application-qualification-01/receipt.json'
    event_bank=BASE/'application-event-reference-01/bank.json'
    if mechanics.exists()and event_bank.exists():
        value=read(mechanics);events=read(event_bank)
        if value['status']!='passed'or len(value['rows'])!=12 or len(events['trajectories'])!=8:
            raise ValueError('Expected complete application preparation evidence')
        record('B-events','passed','Original event comparison retained; new bounded transport/restart and12real neural/native mechanics cases pass. Unused pre-event states are invalidated, unsupported boundary stops before14. These are K2checkpoint diagnostic horizon views, not independent K fits or application-speed evidence. Eight consumed-development event references captured; reserved scenes untouched',
            b+'forecast-application-qualification-01/receipt.json',b+'bounded-native-client-qualification-01/receipt.json',b+'application-event-reference-01/bank.json',d+'application-evaluation-plan.md')
    record('P-custody','passed','Finite managed fits, source/data closure, bound training artifacts and bitwise CPU restores. Routed assembly honestly local with zero optimizer updates',b+'temporal-training-batch-02/execution.json',b+'head-replication-batch-01/execution.json',d+'routed-head-registration.json')
    record('P-report','interim','Separate frontiers retain all negative comparisons and the first complete K2 accuracy/median-speed pass; five-repeat spreads, exact execution proofs and corrected-family gates are explicit; whole-plan goal remains active',d+'frontiers.json',d+'results.md',d+'next-decisions.md')
    record('P-immutable','passed','Original plan copy unchanged; qualified original model sources/artifacts retained; no candidate numerical inverse/projection. Original-user dirty files were outside study edits',d+'requested-scope.md',d+'baseline-evidence.json',b+'a-head-calibration-routed-qualification-01.json')
    if sha(DOCS/'plan.md')!=registration['plan_sha256'] or sha(DOCS/'requested-scope.md')!=registration['requested_scope_sha256']:
        raise ValueError('Original requested scope or plan changed')
    write(DOCS/'requirements-audit.json',dict(status='active',original_goal_complete=False,updated_unix=time.time(),
          registration_sha256=sha(DOCS/'registration.json'),requirements=list(rows.values()),source_sha256=sha(__file__),
          outstanding=['Temporal finalist seed confirmation, replicated control and their exact execution/CPU costs','Frozen temporal finalists and reserved240-step evidence','Application/event accounting and final consolidated report']))
    lines=['# Original requirement audit','','The frozen registration is unchanged. Experiment completion and target attainment are separate.','',
           '| ID | Status | Finding |','| --- | --- | --- |']
    lines.extend('| '+row['id']+' | '+row['status']+' | '+row.get('note','')+' |' for row in rows.values())
    lines.extend(['','[Machine-readable evidence and hashes](requirements-audit.json). The original whole-plan goal remains active.',''])
    (DOCS/'requirements-audit.md').write_text('\n'.join(lines))
    print('Audited all29original requirements without changing registration',flush=True)


if __name__=='__main__':run()
