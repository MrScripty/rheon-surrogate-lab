"""Publish small paired-replay evidence; raw captures remain under runs/."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(run, destination):
    plan = read(run/'plan.json')
    summary = read(run/'summary.json')
    records = []
    source_receipts = {}
    for model in summary['models']:
        path = run/model['id']/'receipt.json'
        receipt = read(path)
        source_receipts[model['id']] = dict(path=str(path),sha256=sha(path))
        by_step = {r['step']:r for r in receipt['rows']}
        contrasts = []
        for row in receipt['rows']:
            if row['detected_peak'] and row['step']-1 in by_step and row['step']+1 in by_step:
                contrasts.append({kind:row[kind]['absolute_rms_pa']/statistics.mean(
                    by_step[row['step']+offset][kind]['absolute_rms_pa'] for offset in (-1,1))
                    for kind in ('baseline','extra')})
        records.append({**model,
            'weights_sha256':next(j['weights_sha256'] for j in plan['jobs'] if j['id']==model['id']),
            'original_trajectory_exact':receipt['original_trajectory_exact'],
            'cuda_initialized':receipt['cuda_initialized'],'torch_threads':receipt['torch_threads'],
            'peak_child_rss_bytes':receipt['peak_child_rss_bytes'],'cpu':receipt['cpu'],
            'peak_to_adjacent_error_ratio':{kind:statistics.median(r[kind] for r in contrasts) for kind in ('baseline','extra')},
            'rows':receipt['rows']})
    total = sum(len(r['rows']) for r in records)
    peak_count = sum(r['detected_peak'] for m in records for r in m['rows'])
    improved = sum(r['absolute_error_ratio']<1 for m in records for r in m['rows'])
    free_improved = sum(r['mean_free_error_ratio']<1 for m in records for r in m['rows'])
    peak_improved = sum(r['absolute_error_ratio']<1 for m in records for r in m['rows'] if r['detected_peak'])
    peak_reductions = ', '.join(f"{m['id']}: {100*(1-m['groups']['peaks']['median_absolute_error_ratio']):.1f}%" for m in records)
    neighbor_reductions = ', '.join(f"{m['id']}: {100*(1-m['groups']['neighbors']['median_absolute_error_ratio']):.1f}%" for m in records)
    timing = [t for m in records for r in m['rows'] for t in r['timing_samples']]
    report = dict(schema='rheon-pressure-spike-extra-pass-results-v1',status='completed',
        recorded_utc=datetime.now(timezone.utc).isoformat(),configuration=plan['config'],
        experiment_source_sha256=sha(Path(__file__).with_name('replay.py')),report_source_sha256=sha(Path(__file__)),
        plan_sha256=sha(run/'plan.json'),source_receipts=source_receipts,models=records,
        cases=total,absolute_improved=improved,mean_free_improved=free_improved,
        selected_drift_peak_cases=peak_count,
        median_measured_incremental_cost_fraction=statistics.median(s['extra_pass_s']/s['baseline_s'] for s in timing),
        longest_concurrent_worker_s=max(r['elapsed_s'] for r in records),
        sum_worker_peak_rss_bytes=sum(r['peak_rss_bytes'] for r in records),
        baseline_physical_steps=summary['baseline_simulation_steps'],counterfactual_simulation_steps=0,optimizer_updates=0,
        method=dict(selection=f"Highest {plan['config']['peaks_by_excursion']} local excursions and highest {plan['config']['peaks_by_absolute_error']} absolute errors among detected peaks; union with worst-error step and immediate neighbors, per cohort. Selection frozen before execution.",
            inputs=f"{len(records)} CPU episodes of {plan['scene']['steps']} steps, {plan['scene']['warmup_steps']} native warm-up steps each. Original physical frames reproduced exactly, ignoring clocks and request UUID hashes. Selected input tensors and baseline predictions retained as NPZ.",
            counterfactual='Recompute remaining residual from the original input and baseline pressure; execute exactly one learned pass, using the existing corrector. Do not advance the simulation with that pressure.',
            oracle='Existing separable spectral pseudoinverse for the sealed Neumann operator, checked against the independent explicit-neighbor action at relative defect < 1e-9. Reference is the solution for this exact AI request, not the independently evolving reference trajectory in the drift plot.',
            precision='Original float32 neural weights and forward pass; unchanged existing float64 physical residual/reconstruction/oracle arithmetic. CPU only.',
            timing=f"One discarded warm-up repetition plus {plan['config']['timing_repetitions']} measured pairs per capture. Baseline is model correction and stopping checks; incremental extra includes residual reconstruction, one neural pass and final stopping-quality calculation. Excludes IPC and simulator advancement. Independent workers pinned to CPUs {[r['cpu'] for r in records]}, each one Torch thread.",
            resource_policy=f"Immutable model packages in separate processes; {plan['config']['minimum_available_memory_bytes']} bytes available-memory reserve; native deadline {plan['config']['native_timeout_seconds']} s and worker deadline {plan['config']['worker_timeout_seconds']} s. Existing rollout owns native termination/reaping; worker cancellation raises through that owner."),
        findings=[
            f'{improved}/{total} tested inputs improved in absolute error; {free_improved}/{total} improved in mean-free same-request error. Selected drift-peak steps: {peak_count}.',
            'Median same-request pressure error reduction at selected peak steps: '+peak_reductions+'.',
            'Median reduction in neighboring/context inputs: '+neighbor_reductions+'. The comparison shrinks current-solve errors but does not establish disappearance of the phase-dependent drift pattern.',
            'Some peaks in independently evolved trajectory error are not peaks in same-request pressure error. Accumulated state differences and pressure-gauge effects remain distinct from current-solve error.',
            ('The model can still contract these small residuals after the existing stopping rule accepts them; this is not an irreducible current-solve error floor in these cases.'
             if improved == total else 'One extra pass did not improve every selected input; review individual cases before changing stopping.'),
            'This paired test does not establish improved full-trajectory drift, deployment readiness, or a general model ranking. No weights, training, acceptance thresholds or deployed solver changed.',
        ],next_step='Run a separately labelled closed-loop comparison using adaptive stop plus exactly one extra pass, with unchanged scene inputs and physical limits. Measure drift, worst-step accuracy and complete step runtime before adopting the policy or selecting new training feedback.')
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    cards = []
    table = []
    for model in records:
        group=model['groups']['peaks']
        table.append(f"<tr><th>{model['id']}</th><td>{len(model['rows'])}</td><td>{group['cases']}</td><td>{100*(1-group['median_absolute_error_ratio']):.1f}%</td><td>{model['baseline_median_s']*1000:.2f} ms</td><td>+{model['extra_pass_median_s']*1000:.2f} ms</td></tr>")
        rows=[]
        for row in model['rows']:
            rows.append(f"<tr><th>{row['step']}</th><td>{'Drift peak' if row['detected_peak'] else 'Neighbor / worst context'}</td><td>{row['baseline']['absolute_rms_pa']*1e6:.4f}</td><td>{row['extra']['absolute_rms_pa']*1e6:.4f}</td><td>{100*(1-row['absolute_error_ratio']):.1f}%</td><td>{row['baseline_network_passes']} → {row['baseline_network_passes']+1}</td></tr>")
        cards.append(f"<details><summary>{model['id']} · individual paired measurements</summary><div class='scroll'><table><thead><tr><th>Physical step</th><th>Selection</th><th>Baseline µPa RMS</th><th>Extra-pass µPa RMS</th><th>Reduction</th><th>Passes</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></details>")
    prose=''.join('<li>'+html.escape(f)+'</li>' for f in report['findings'])
    method=''.join('<p><strong>'+html.escape(k.capitalize())+':</strong> '+html.escape(v)+'</p>' for k,v in report['method'].items())
    (destination/'results.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pressure spike · one extra pass</title><style>body{margin:0;background:#0b141c;color:#e8f0f4;font:16px/1.6 system-ui}main{max-width:1050px;margin:auto;padding:36px 24px}h1{line-height:1.2}p,li{color:#b7c6d0}a{color:#60dfc3}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;text-align:left;font-size:14px;white-space:nowrap}td,th{padding:12px;border-bottom:1px solid #263744}thead th{color:#9cbdcb}details{margin-top:24px;padding:16px;background:#111e29;border-radius:8px}summary{cursor:pointer}strong{color:#e8f0f4}.banner{padding:14px;border-left:3px solid #60dfc3;background:#111e29}h2{margin-top:32px}</style><main><p>Rheon surrogate lab · CPU diagnostic</p><h1>One additional learned pressure correction</h1>'''+f'''<p class="banner">{improved}/{total} paired inputs improved; {peak_improved}/{peak_count} selected drift-peak steps improved. No model training or deployment change.</p><p>This measures pressure accuracy on identical requests. It does not measure the drift of a trajectory advanced using the extra-pass policy.</p><div class="scroll"><table><thead><tr><th>Model</th><th>Inputs</th><th>Peak steps</th><th>Median peak error reduction</th><th>Baseline correction</th><th>Extra pass</th></tr></thead><tbody>{''.join(table)}</tbody></table></div><p>Errors use a qualified spectral solution of the same request. Timing excludes simulation and IPC. All three unchanged baseline trajectories reproduced exactly. Workers ran concurrently; the longest took {report['longest_concurrent_worker_s']:.1f} seconds. The largest worker peak RSS was {max(r['peak_rss_bytes'] for r in records)/2**20:.1f} MiB.</p><h2>What the result means</h2><ul>{prose}</ul>{''.join(cards)}<details><summary>Method, precision and timing</summary>{method}</details><h2>Next experiment</h2><p>{html.escape(report['next_step'])}</p><p><a href="results.json">Complete measurements and method</a> · <a href="../cohort-progress.html">Cohort progress dashboard</a></p></main></html>''')
    print(json.dumps({k:report[k] for k in ['cases','absolute_improved','mean_free_improved','longest_concurrent_worker_s','median_measured_incremental_cost_fraction']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.run.resolve(),args.output.resolve())
