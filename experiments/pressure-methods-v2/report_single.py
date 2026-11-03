"""Summarize completed single-track repeats without hiding failed trajectories."""
import statistics
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def report():
    base=ROOT/'runs/pressure-methods-v2'
    execution=read(base/'single-timing-controller-02/execution.json')
    if execution['status']!='completed' or len(execution['completed'])!=48 or not execution['owned_child_reaped']:
        raise ValueError('All 48 terminal, reaped commands required')
    records=[];groups={}
    for path in sorted((base/'single-timing-02').glob('*/results.json')):
        result=read(path)
        if not result['native_process']['reaped'] or result['native_process']['returncode']!=0:
            raise ValueError('Native custody incomplete')
        records.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
        key=(tuple(result['scene']['counts']),result['policy'],result['passes'])
        groups.setdefault(key,[]).append(result)
    if len(records)!=48:raise ValueError('All policy results required')
    rows=[]
    for (counts,policy,passes),results in sorted(groups.items()):
        if len(results)!=3:raise ValueError('Three repeats per policy required')
        active=[r['post_warmup_median_ms'] for r in results if r['post_warmup_median_ms']]
        rows.append(dict(counts_xyz=list(counts),policy=policy,passes=passes,
            repetitions=3,completed=sum(r['status']=='completed' for r in results),
            accepted_steps=[r['completed_steps'] for r in results],
            median_frame_ms=statistics.median(r['frame_seconds'] for r in active) if active else None,
            median_pressure_ms=statistics.median(r['pressure_seconds'] for r in active) if active else None,
            median_model_load_seconds=statistics.median(r['model_load_seconds'] for r in results),
            median_native_setup_seconds=statistics.median(r['native_setup']['setup_seconds'] for r in results),
            median_simulation_wall_seconds=statistics.median(r['simulation_wall_seconds'] for r in results),
            median_worker_total_seconds=statistics.median(r['worker_total_seconds'] for r in results),
            peak_worker_rss_kib=max(r['peak_worker_rss_kib'] for r in results),
            peak_native_rss_kib=max(r['peak_native_rss_kib'] for r in results)))
    destination=ROOT/'docs/plans/pressure-methods-v2'
    write(destination/'application-results.json',dict(status='completed_bounded_single_track',rows=rows,
        measurements=records,execution_sha256=sha(base/'single-timing-controller-02/execution.json'),
        source_sha256=sha(__file__),build_receipt_sha256=sha(base/'single-build-01/build-receipt.json'),
        limitations=['64 steps per scene, two cube sizes, one force seed; not production geometry qualification',
                    'Python/JSON request transfer counted for all external numerical/neural policies; native PCG is in-process',
                    'Native-loose uses 1e-6 throughout; other native warmup steps use 1e-8',
                    'Median frame times for rejected policies describe accepted prefixes, not completed simulations']))
    lines=['# Single-track application comparison','',
        'Three serial repetitions per policy on CPU 23, one thread, 64 steps including 15 native warmup steps. '
        'Every candidate frame uses one fluid track; there are no reference or same-request oracle solves. '
        'Fifteen Rust diagnostic tests and Clippy passed, including exact native pressure/velocity/tracer parity '
        'and rejection without advancing history. Original selected Rheon sources were not modified.','',
        '| Grid | Policy | Completed repeats | Accepted steps | Median frame ms | Median pressure ms |',
        '| --- | --- | ---: | --- | ---: | ---: |']
    for row in rows:
        name=row['policy']+(f" ({row['passes']} passes)" if row['policy'] not in ['native','native-loose','spectral'] else '')
        frame='—' if row['median_frame_ms'] is None else f"{row['median_frame_ms']:.3f}"
        pressure='—' if row['median_pressure_ms'] is None else f"{row['median_pressure_ms']:.3f}"
        lines.append(f"| {row['counts_xyz'][0]}³ | {name} | {row['completed']}/3 | {row['accepted_steps']} | {frame} | {pressure} |")
    lines+=['','Frame time includes advection, forcing, viscosity, pressure request/reply conversion and IPC, '
        'projection and host physical gates. Simulation wall time additionally includes native startup/setup and output IO. '
        'The JSON retains cold model loading, setup, total worker time, per-frame measurements and peak memory. '
        'No model loading or setup cost is amortized into the frame medians.','',
        'The spectral policy is fastest on these two sealed uniform scenes. The trained coarse hybrid improves '
        'over the retained neural policy but does not beat native PCG with this CPU/Python integration. '
        'These results motivate numerical baselines and integration work alongside training. They do not establish '
        'performance for an embedded implementation, GPU inference, arbitrary geometry or production grid sizes.','',
        'Rejected-policy timings cover accepted prefixes only; they are ineligible as successful speed comparisons. '
        'Native-loose uses a 1e-6 solver tolerance for all steps, while the other native prefixes use 1e-8. '
        'All candidate admission gates retain the existing relative 1e-6 residual and physical checks.','',
        '[Full measurements and hashes](application-results.json)']
    (destination/'application-results.md').write_text('\n'.join(lines)+'\n')
    campaign=read(destination/'campaign.json');campaign['directions'][0]['status']='completed_bounded_single_track'
    campaign['directions'][0]['evidence']='application-results.json'
    write(destination/'campaign.json',campaign)
    print('Recorded all 48 single-track results, including rejected policies',flush=True)

if __name__=='__main__':report()
