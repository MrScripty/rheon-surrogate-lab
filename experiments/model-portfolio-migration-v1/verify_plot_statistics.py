"""Check scalar export against all accepted aggregates; retain lead-level views."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from portfolio_statistics import distribution
from retain_portfolio_scalar_statistics import error_values


def run():
    report_path = DOC/'reports/plot-statistics-qualification-01.json'
    lead_path = DOC/'reports/multistep-lead-statistics-01.json'
    require(not report_path.exists() and not lead_path.exists(), 'Preserve first qualification')
    source_path = DOC/'reports/plot-statistics-01.json'
    view = read(source_path)
    spec = view['scalar_case_payload']
    require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Scalar payload changed')
    aggregate_path = DOC/'reports/portfolio-scalar-statistics-01.json'
    accepted = {m['fit_item_id']:m for m in read(aggregate_path)['models']}
    values = defaultdict(lambda:defaultdict(list))
    leads = defaultdict(list)
    counts = Counter()
    metadata = {m['subject_id']:m for m in view['models']}
    guidance_example = defaultdict(list)
    per_subject_calls = Counter()
    with gzip.open(ROOT/spec['path'],'rt') as stream:
        for line in stream:
            case = json.loads(line)
            subject = case['subject_id']
            dims = case['dimensions']
            per_subject_calls[subject] += 1
            for output in case['outputs']:
                metrics = dict(output['plot_metrics'])
                if metadata[subject]['comparison_group'] == 'pressure_request':
                    # The independent common-score report verified this equality
                    # on every named request; original restores use its scorer.
                    if 'equation_relative_l2' in metrics:
                        metrics['common_equation_relative_l2'] = metrics['equation_relative_l2']
                metrics.update(error_values(output.get('destination_metrics',{}),'destination.'))
                if subject in accepted:
                    for metric,value in metrics.items():
                        values[subject][metric].append(value)
                if 'lead' in output:
                    for metric,value in metrics.items():
                        key = (subject,dims.get('K'),dims.get('output_dt_s'),output['lead'],
                               output.get('output_role'),metric)
                        leads[key].append((value,dict(dims,lead=output['lead'],case_index=case['case_index'])))
                    if subject == 'fit-d18b434d52b73791' and dims['identifier'] == 'long-v1-irregular_pulse-reflected_line-r0-cpm100' \
                            and dims['start_fine_step'] == 4768 and dims['output_steps_per_s'] == 30 \
                            and output['lead'] == 1:
                        guidance_example[dims['K']].append(metrics['pressure_relative_l2'])
                    counts['per_lead_outputs'] += 1
                if 'GPU_timing_ms' in case:
                    require('GPU_timing_ms' not in output and 'latency_ms' not in output,
                        'Joint-call cost must not be presented as a separately measured lead latency')
            if 'GPU_timing_ms' in case:
                counts['timed_calls'] += 1
    metric_checks = 0
    scalar_checks = 0
    for subject,model in accepted.items():
        for metric,summary in model['accuracy']['metrics'].items():
            observed = distribution(values[subject][metric])
            for field in ('samples','best','worst','mean','median','p95'):
                require(observed[field] == summary[field], f'Aggregate differs: {subject}/{metric}/{field}')
                scalar_checks += 1
            metric_checks += 1
    require(set(guidance_example) == {1,2,3,4} and all(len(v) == 1 for v in guidance_example.values()),
        'The same physical first target must survive independently at every K')
    require(guidance_example[1][0] < .005 and guidance_example[4][0] > .4,
        'Known inaccurate intermediate frame must not be hidden by endpoint/block aggregation')
    rows = []
    for key,population in sorted(leads.items()):
        subject,K,dt,lead,role,metric = key
        result = distribution(v for v,_ in population)
        result.update(best_case=min(population,key=lambda item:item[0])[1],
                      worst_case=max(population,key=lambda item:item[0])[1])
        rows.append(dict(subject_id=subject,K=K,output_dt_s=dt,block_horizon_s=K*dt,
            lead=lead,physical_lead_dt_s=lead*dt,output_role=role,metric=metric,statistics=result))
    save(lead_path,dict(status='every_retained_physical_lead_summarized_separately',
        producer=binding(Path(__file__)),source=binding(source_path),rows=rows,
        same_target_cross_K_example=dict(subject_id='fit-d18b434d52b73791',
            identifier='long-v1-irregular_pulse-reflected_line-r0-cpm100',start_fine_step=4768,
            output_steps_per_s=30,lead=1,pressure_relative_l2_by_K=dict(guidance_example)),
        timing_attribution='Full block and block/K remain call statistics; no invented per-lead costs',
        aggregation='Per-output errors over original consumed diagnostic cases, not independent holdout/drift',
        deletion_paused=True,new_model_forwards=0,prediction_fields_opened=0))
    save(report_path,dict(status='all203fit_accuracy_distributions_exact_and_every_physical_lead_preserved',
        created_unix=time.time(),producer=binding(Path(__file__)),source=binding(source_path),
        accepted_aggregates=binding(aggregate_path),lead_statistics=binding(lead_path),
        fit_populations_exact=len(accepted),metric_populations_exact=metric_checks,scalar_aggregate_checks=scalar_checks,
        calls_by_subject=dict(per_subject_calls),counts=dict(counts),
        same_target_cross_K_control_passed=True,joint_latency_not_falsely_assigned_to_leads=True,
        deletion_paused=True,prediction_fields_opened=0,prediction_files_deleted=0,new_model_forwards=0,
        full_portfolio_complete=False))
    print(json.dumps(dict(fits=len(accepted),metric_populations=metric_checks,scalar_checks=scalar_checks,
        lead_groups=len(rows),same_target_first_lead_pressure_by_K=dict(guidance_example)),sort_keys=True),flush=True)


if __name__ == '__main__':
    run()
