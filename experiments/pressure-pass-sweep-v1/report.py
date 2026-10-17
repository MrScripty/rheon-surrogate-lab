"""Validate completed receipts and retain a compact, reproducible study report."""
import argparse
from datetime import datetime,timezone
import html
import json
from pathlib import Path
import re
import statistics
import subprocess
from material import read,sha


def embedded(path):
    return json.loads(re.search(r'<script id="chart-data" type="application/json">(.*?)</script>',path.read_text(),re.S)[1])


def report(material_path,destination):
    material=read(material_path);root=Path(material['root']);run=material_path.parent
    chart=read(destination/'chart-data.json')
    if chart['status']!='completed':raise ValueError('Both numerical and isolated timing campaigns must finish')
    snapshot_path=run/'cohort-progress-selection-snapshot.html'
    if sha(snapshot_path)!=material['chart_sha256']:raise ValueError('Frozen selection chart changed')
    prior=embedded(snapshot_path)
    current=embedded(root/material['config']['chart'])
    changed=[key for key in prior if prior[key]!=current.get(key) and key not in ['history_refreshed_utc','pass_sweep']]
    if changed:raise ValueError('Historical chart records changed: '+str(changed))
    rows=[];reproductions=[];all_sources_unchanged=True
    for model in material['models']:
        for path,expected in model['identities'].items():
            if sha(root/path)!=expected:raise ValueError('Changed captured model artifact '+path)
        snapshot=read(run/model['id']/'snapshots.json')
        rows.extend(snapshot['rows'])
        if any(row['network_calls']!=row['passes']for row in snapshot['rows']):
            raise ValueError('Measured forward count differs from requested count')
        config=read(root/model['project']/'config.json')
        original_n=(config.get('correction')or{}).get('steps',1)
        original_error=max(row['relative_residual']for row in snapshot['rows']if row['passes']==original_n)
        difference=abs(original_error-model['original_worst_error'])
        if difference>1e-12:raise ValueError('Original proposal error not reproduced: '+model['id'])
        reproductions.append(dict(model=model['id'],original_passes=original_n,absolute_difference=difference))
    if len(rows)!=len(material['models'])*len(material['config']['passes'])*128:
        raise ValueError('Missing static validation rows')
    for name,expected in material['validation_sources'].items():
        if sha(root/material['config']['validation_inputs']/name)!=expected:
            raise ValueError('Changed validation input '+name)
    if sha(root/material['config']['binary'])!=material['binary_sha256']:
        raise ValueError('Changed native binary')
    full_steps=material['config']['scene']['steps']-material['config']['scene']['warmup_steps']
    for trace in chart['trajectories']:
        if not trace['native_reaped']:raise ValueError('Native process not reaped')
        if trace['mean_actual_network_calls']!=trace['passes']:raise ValueError('Trajectory call count mismatch')
        if trace['completed']:
            if len(trace['points'])!=full_steps or trace['rejected']:raise ValueError('Incomplete full-window trace')
        elif not trace['rejected']:raise ValueError('Missing scientific rejection')
    if any(worker['status']!='passed'for worker in chart['workers']):raise ValueError('Worker failed')
    if any(read(run/model['id']/'receipt.json')['cuda_initialized']for model in material['models']):
        raise ValueError('CPU campaign initialized CUDA')
    timing_config=read(root/'docs/plans/pressure-pass-sweep-v1/timing-config.json')
    for model in material['models']:
        receipt=read(run/'isolated'/model['id']/'receipt.json')
        if (receipt['status']!='passed' or receipt['cuda_initialized'] or receipt['torch_threads']!=1
            or receipt['cpu']!=timing_config['cpu'] or receipt['material_sha256']!=sha(material_path)):
            raise ValueError('Isolated CPU timing contract differs: '+model['id'])
    per_pass=[]
    for n in chart['passes']:
        points=[p for p in chart['points']if p['passes']==n]
        traces=[t for t in chart['trajectories']if t['passes']==n]
        complete=[t for t in traces if t['completed']]
        best=min(points,key=lambda p:p['worst_error'])
        per_pass.append(dict(passes=n,completed_models=[t['id']for t in complete],
            rejected_models=[t['id']for t in traces if not t['completed']],
            all_128_equation_checks_pass_models=[p['model']for p in points if p['equation_passes']==128],
            best_static_model=best['model'],best_static_worst_error=best['worst_error'],
            best_static_inference_median_ms=best['inference_median_ms'],
            median_model_inference_ms=statistics.median(p['inference_median_ms']for p in points),
            complete_window_worst_relative_pressure_error={t['id']:max(p[2]for p in t['points'])for t in complete}))
    execution=read(run/'execution.json');timing=read(run/'isolated/execution.json')
    lab=root/material['config']['source_lab']
    lab_commit=subprocess.run(['git','-C',str(lab),'rev-parse','HEAD'],check=True,text=True,capture_output=True).stdout.strip()
    lab_status=subprocess.run(['git','-C',str(lab),'status','--porcelain'],check=True,text=True,capture_output=True).stdout
    if lab_status:raise ValueError('Captured integration checkout has ongoing edits')
    def archive(name):
        path=run/name
        return str(path.relative_to(root))if path.exists()else None
    interruption=read(run/'interruption-01.json')if(run/'interruption-01.json').exists()else None
    runtime=read(root/material['config']['runtime'])
    browser=read(run/'browser/browser-receipt.json');focused=read(run/'focused-verification.json')
    if (browser['status']!='passed' or browser['html_sha256']!=sha(root/material['config']['chart'])
        or focused['status']!='passed'):
        raise ValueError('Browser or focused verification is missing or stale')
    verification=dict(status='passed',historical_data_keys_preserved=len(set(prior)-{'history_refreshed_utc','pass_sweep'}),
        historical_refresh_timestamp_exempted=True,static_rows=len(rows),
        static_forward_counts_exact=True,trajectory_forward_counts_exact=True,
        original_proposals_reproduced=reproductions,
        model_artifacts_unchanged=all_sources_unchanged,validation_inputs_unchanged=True,
        binary_unchanged=True,cuda_initialized=False,
        all_primary_children_reaped=execution['all_owned_children_reaped'],
        all_timing_children_reaped=all(w['reaped']and w['returncode']==0 for w in timing['workers']),
        all_native_children_reaped=True,
        isolated_inputs_and_outputs_match_primary_exactly=True,
        preflight_fixtures=[read(path)for path in sorted(run.glob('*-fixture/receipt.json'))],
        browser=browser,focused_checks=focused,
        standards_snapshot='snapshot:v1:ffcfc3f8-b52a-434e-b57f-d259a0355711')
    result=dict(schema='rheon-inference-pass-sweep-results-v1',status='completed',
        completed_utc=datetime.now(timezone.utc).isoformat(),models=chart['models'],passes=chart['passes'],
        combinations=chart['expected_combinations'],validation_cases=128,static_rows=len(rows),
        full_window_trajectories=sum(t['completed']for t in chart['trajectories']),
        rejected_trajectories=sum(not t['completed']for t in chart['trajectories']),
        scene=chart['scene'],physical_contract=material['support'],per_pass=per_pass,timing_scope=chart['timing_scope'],drift_scope=chart['drift_scope'],
        execution=execution,isolated_timing=timing,interruption=interruption,
        checkpoint_workers=chart['workers'],
        elapsed_accounting='execution.elapsed_s covers the final parallel supervisor invocation, including any reused completed settings. Per-worker and per-trajectory measured durations are retained separately. isolated_timing.elapsed_s covers the subsequent serial latency campaign. When interrupted, the first supervisor has no retained total wall-time receipt; no uninterrupted total is inferred.',
        runtime={**{key:runtime[key]for key in ['profile','python','python_version','site_packages','packages','assumption']},
            'source':material['config']['runtime'],'sha256':sha(root/material['config']['runtime'])},
        peak_checkpoint_rss_bytes=max(w['peak_rss_bytes']for w in chart['workers']),
        peak_native_child_rss_bytes=max(w['peak_child_rss_bytes']for w in chart['workers']),
        provenance=dict(material=str(material_path.relative_to(root)),material_sha256=sha(material_path),
            binary_sha256=material['binary_sha256'],selection_chart_sha256=material['chart_sha256'],
            integration_lab_commit=lab_commit,integration_lab_clean=True,
            study_sources={path.name:sha(path)for path in Path(__file__).parent.iterdir()if path.suffix in ['.py','.cjs']},
            original_owner_source=archive('executed-source-attempt-01/run_all.py'),
            resumed_owner_source=archive('executed-source-attempt-02/run_all.py'),
            timing_owner_source=archive('executed-timing-supervisor-01.py'),
            owner_note='Where present, source archives identify code loaded at launch. This study repaired supervisor cancellation draining during the campaign; its focused lifecycle test passes. Numerical timing workers load the final per-case warm-up implementation.'),
        commands=[
            'python3 experiments/pressure-pass-sweep-v1/material.py --config docs/plans/pressure-pass-sweep-v1/config.json --output runs/pressure-pass-sweep-v1/NEW-RUN',
            'python3 experiments/pressure-pass-sweep-v1/run_all.py --material runs/pressure-pass-sweep-v1/NEW-RUN/material.json',
            'python3 experiments/pressure-pass-sweep-v1/retime.py --material runs/pressure-pass-sweep-v1/NEW-RUN/material.json --config docs/plans/pressure-pass-sweep-v1/timing-config.json --wait',
            'python3 experiments/pressure-pass-sweep-v1/collect.py --material runs/pressure-pass-sweep-v1/NEW-RUN/material.json --output docs/plans/pressure-pass-sweep-v1',
            'python3 docs/plans/refresh-cohort-progress.py',
            'node experiments/pressure-pass-sweep-v1/browser.cjs CHROMIUM-EXECUTABLE runs/pressure-pass-sweep-v1/NEW-RUN/browser',
            'python3 experiments/pressure-pass-sweep-v1/report.py --material runs/pressure-pass-sweep-v1/NEW-RUN/material.json --output docs/plans/pressure-pass-sweep-v1',
            'PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/pressure-pass-sweep-v1 -p "test_*.py" -v'],
        limits=['Previously consumed validation inputs; no untouched holdout.',
            'Independent drift is measured on one configured physical scene.',
            'Full-window completion keeps native physics gates, but does not establish the deployed adaptive pressure-bound certificate.',
            'Static equation residual and independently evolving pressure-field error measure different quantities.',
            'More passes are measured rather than assumed to be monotonically beneficial.',
            'Fixed research policy; original weights, trained proposals and deployed adaptive policy remain unchanged.'])
    for name,value in [('results.json',result),('verification.json',verification)]:
        (destination/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>'for v in [p['passes'],len(p['completed_models']),len(p['rejected_models']),p['best_static_model'],format(p['best_static_worst_error'],'.3g'),format(p['best_static_inference_median_ms'],'.3f')])+'</tr>'for p in per_pass)
    (destination/'results.html').write_text('''<!doctype html><meta charset="utf-8"><title>Fixed-pass CPU study</title>
<style>body{max-width:1100px;margin:40px auto;padding:0 20px;font:17px system-ui;background:#111925;color:#e2eafa}a{color:#8cceff}table{border-collapse:collapse}td,th{padding:10px;text-align:left;border-bottom:1px solid #465367}</style>
<h1>Fixed-pass CPU inference study</h1><p>21 checkpoints, eight fixed neural pass counts, identical 128 validation inputs and independent 512-step simulations with 15 native warm-up steps.</p>
<p><a href="../cohort-progress.html#passsweep">Open the connected error/time and drift charts</a> · <a href="results.json">Full results and timings</a> · <a href="verification.json">Verification receipt</a></p>
<p>Rejected simulations stop without fallback. Completion below refers to native physics checks; the deployed adaptive adapter's additional pressure-bound certificate is not applied by this research policy. Snapshot equation error differs from pressure-field drift.</p>
<table><thead><tr><th>Neural passes</th><th>Full trajectories</th><th>Rejected</th><th>Best static error model</th><th>Worst equation error</th><th>Its median inference, ms</th></tr></thead><tbody>'''+table+'</tbody></table><p>Methods, interruption recovery, elapsed times, CPU environment and source hashes are retained in the linked JSON. Historical chart results are preserved. No training or deployment changed.</p>\n')
    print(json.dumps(dict(status='passed',combinations=result['combinations'],completed=result['full_window_trajectories'],rejected=result['rejected_trajectories'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--material',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report(args.material.resolve(),args.output.resolve())
