"""Render retained evidence without conflating data populations or timing scopes."""
import html
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>'for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>'for v in row)+'</tr>'for row in rows)+'</tbody></table>'

def build():
    base=ROOT/'runs/pressure-methods-v3';docs=ROOT/'docs/plans/pressure-methods-v3'
    sections=[];payload=dict(status='in_progress',source_sha256=sha(__file__),promotion=False,drift_thresholds='Diagnostic only; application acceptance remains deferred')
    frozen=[]
    for directory in ['frozen-full','frozen-screen']:
        for path in sorted((base/directory).glob('*.json')):
            d=read(path)
            for s in d['summary']:
                frozen.append([d['checkpoint'],d['width'],d['parameters'],s['policy'],f"{s['qualified']}/{s['cases']}",s['median_network_calls'],f"{s['worst_pressure_relative_l2']:.3g}"])
    sections.append('<h2>Frozen named checkpoints and channel slicing</h2><p>Full-population rows use 156 prior development requests. Screens use 16 predeclared requests. Half-width networks have 92,560 parameters; all original named checkpoints have 370,016. Frozen pruning and subsequent training are separate experiments.</p>'+table(['Checkpoint','Width','Parameters','Coarse method','Physical gates passed','Median CNN calls','Worst pressure rel. L2'],frozen))
    evaluations=[];rows=[];old=[]
    for path in sorted((base/'evaluation').glob('*.json')):
        d=read(path);evaluations.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),**d))
        for s in d['summary']:
            if s['population']=='native-development':
                rows.append([d['entry']['id'],d['kind'],s['max_passes'],f"{s['qualified']}/{s['cases']}",f"{s['worst_relative_residual']:.3g}",f"{s['worst_pressure_relative_l2']:.3g}",s['median_network_calls'],f"{s['median_ms']:.2f}"])
        group=[r for r in d['rows']if r['population']!='native-development'and r['max_passes']==8]
        if group:old.append([d['entry']['id'],len(group),sum(r['qualified']for r in group),f"{max(r['pressure_relative_l2']for r in group):.3g}"])
    payload['evaluations']=evaluations
    sections.append('<h2>New native cold/warm development requests</h2><p>Eight same-request tests: 64³ at 8/16 cells per metre, and 128³ at 16/32 cells per metre, each at cold frame 0 and warm frame 15. Physical extents pair at 8 m and 4 m. Mass density is fixed at 1,000 kg/m³. The model timestep stays 1/60 s. Gates require both the original pressure residual and predicted divergence limits. These are solver diagnostics, not application acceptance limits.</p>'+table(['Model','Coarse','Pass ceiling','Gates passed','Worst residual','Worst pressure rel. L2','Median CNN calls','Exploratory median ms'],rows))
    if old:sections.append('<h2>Retained prior development coverage</h2>'+table(['Model','Prior cases','Gates passed','Worst pressure rel. L2'],old))
    training=[];fits=[]
    for arm in ['v3-replay','v3-native','v3-c37half','v3-c44half']:
        generation=3 if arm=='v3-replay'else 7;p=base/f'{arm}-study-{generation:02d}/receipt.json'
        if not p.exists():training.append([arm,'Pending','','','','']);continue
        d=read(p);t=read(p.parent/'train/receipt.json');history=read(p.parent/'train/history.json')
        q=read(base/f'{arm}-qualification-{generation:02d}.json')
        work=history.get('optimizer_work',{})
        training.append([arm,d['status'],t['optimizer_updates'],json.dumps(q['sampling']['tier_exposures']),work.get('forward_cells'),all(d['artifact_parity'].values())])
        fits.append(dict(arm=arm,path=str(p.relative_to(ROOT)),sha256=sha(p),status=d['status'],updates=t['optimizer_updates'],optimizer_work=work,artifact_parity=d['artifact_parity']))
    payload['fits']=fits
    sections.append('<h2>Actual managed training and fresh CPU restore</h2><p>All arms preserve the exact original 1,600-small-sample stream. The control adds 800 prior 32/64-domain samples; the native fits balance 800 larger samples across 32/64/128 tiers. Reported cell work is measured. No equal-cell-budget or isolated initialization effect is claimed between the full and reduced architectures.</p>'+table(['Arm','Status','Updates','Size-tier exposures','Actual optimizer-forward cells','Restore artifact parity'],training))
    sections.append('<p>Two earlier native attempts finished 2,400 updates without publishing usable artifacts. The first hit an inherited 4 GiB process guard; the second exceeded the 8 MiB evidence-file cap with a 23.3 MB diagnostic prediction file. The final Project pins a 12 GiB guard within its 16 GiB reservation. Full-resolution training, evaluation and exact full-field CPU restore hashes are retained; only the diagnostic NPZ exports a declared sampled view. Both unsuccessful attempts remain separate from successful fits.</p>')
    profiling=[];profile_evidence=[]
    for label,stem in [('Baseline','base'),('Shared coordinates','shared')]:
        results=[]
        for rep in range(3):
            candidates=[base/f'profile-quiet-{stem}-{rep}/receipt.json',base/f'profile-shared-{stem}-{rep}/receipt.json']
            path=next((p for p in candidates if p.exists()),None)
            if path:
                d=read(path);frames=d['frames'][15:];results.append((path,d,frames))
        if not results:continue
        values={k:statistics.median(statistics.median(f[k]for f in fs)for _,_,fs in results)for k in ['frame_seconds','preparation_seconds','pressure_seconds']}
        velocity=statistics.median(statistics.median(f['preparation_profile']['advect_velocity']for f in fs)for _,_,fs in results)
        profiling.append([label,len(results),*[f'{values[k]*1000:.3f}'for k in ['frame_seconds','preparation_seconds','pressure_seconds']],f'{velocity*1000:.3f}'])
        profile_evidence.extend(dict(path=str(p.relative_to(ROOT)),sha256=sha(p))for p,_,_ in results)
    payload['profile_evidence']=profile_evidence
    sections.append('<h2>Measured remaining frame cost</h2><p>32³, one thread, CPU 23, 32 native frames; first 15 omitted from timing summaries. Values are medians of three per-run medians. Shared-coordinate sampling targets velocity advection. Native request and spectral pressure hashes match bitwise across baseline/modified captures. Earlier inlining-only trials did not establish a speed improvement. Timing status and isolation are retained in the final audit.</p>'+table(['Sampler','Repeats','Frame ms','Preparation ms','Pressure + IPC ms','Velocity advection ms'],profiling))
    aligned=[]
    for label,stem in [('Matched baseline','base'),('Shared coordinates + exact-sample shortcut','fast')]:
        groups=[]
        for rep in range(3):
            path=base/f'profile-aligned-{stem}-{rep}/receipt.json'
            if path.exists():groups.append(read(path)['frames'][15:]);profile_evidence.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
        if groups:
            frame_values=[1000*statistics.median(f['frame_seconds']for f in frames)for frames in groups]
            velocity_values=[1000*statistics.median(f['preparation_profile']['advect_velocity']for f in frames)for frames in groups]
            aligned.append([label,len(groups),f'{statistics.median(frame_values):.3f}',f'{statistics.median(velocity_values):.3f}',', '.join(f'{v:.2f}'for v in frame_values)])
    if aligned:sections.append('<h2>Exact-sample sampler follow-up</h2><p>A direct lookup is used only when all interpolation fractions are exactly zero. The original accumulator’s signed-zero behavior is preserved. Forty Rust tests include exact face/cell centres, extreme finite values and subnormals; 32³ and 128³ native request/reference hashes match the baseline. The three paired trials use AB, BA, AB order. The gain is small: report the measured medians and run variation, not a large or statistically established speedup. All frame timings here mean headless fluid steps; GUI/rendering and model loading are excluded.</p>'+table(['Sampler','Repeats','Median fluid step ms','Velocity advection ms','Per-run fluid step medians ms'],aligned))
    payload['profile_evidence']=profile_evidence
    large_profiles=[]
    for size in [64,128]:
        frames=[]
        for path in (base/'native-data').glob('v3-train-*/receipt.json'):
            d=read(path)
            if d['scene']['counts']==[size]*3:frames.extend(d['frames'][1:])
        if frames:
            large_profiles.append([f'{size}³',len(frames),f"{1000*statistics.median(f['frame_seconds']for f in frames):.1f}",
                f"{1000*statistics.median(f['preparation_profile']['advect_velocity']for f in frames):.1f}",
                f"{1000*statistics.median(f['preparation_profile']['viscosity']for f in frames):.1f}",
                f"{1000*statistics.median(f['preparation_profile']['advect_tracer']for f in frames):.1f}",
                f"{1000*statistics.median(f['pressure_seconds']for f in frames):.1f}"])
    if large_profiles:sections.append('<p>The same phase instrumentation on native training captures also identifies velocity advection as the largest phase at 64³ and 128³. These exploratory measurements ran concurrently with other campaign work and are not isolated speed comparisons. Frame 0 is omitted; phase medians need not sum to the frame median.</p>'+table(['Grid','Frames','Full frame ms','Velocity advection ms','Viscosity ms','Tracer advection ms','Spectral pressure + IPC ms'],large_profiles))
    coupled=[]
    for path in sorted((base/'coupled').glob('*/receipt.json')):
        d=read(path);coupled.append([(d.get('entry')or{}).get('id','spectral'),str(d['grid_dimensions']),d['status'],sum(f['advanced']for f in d['frames']),d['scene']['steps'],d.get('max_passes',0),f"{1000*statistics.median(f['frame_seconds']for f in d['frames']):.2f}",statistics.median(c.get('work',{}).get('network_calls',0)for c in d['callbacks'])])
    if coupled:sections.append('<h2>Coupled host checks</h2><p>Actual surrogate pressures drive the native fluid track on the same CPU 18 and reserved forcing scenes. A rejected pressure never advances history; no fallback solver is substituted. Four-step checks are diagnostic, not long-horizon drift acceptance. Timing for a rejected track does not represent a usable solver.</p>'+table(['Model','Grid','Result','Advanced frames','Requested frames','Pass ceiling','Median full frame ms','Median CNN calls'],coupled))
    blind=[];blind_evidence=[]
    for path in sorted((base/'blind-evaluation').glob('*.json')):
        d=read(path);blind_evidence.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),**d))
        for s in d['summary']:blind.append([d['entry']['id'],s['max_passes'],f"{s['qualified']}/{s['cases']}",f"{s['worst_relative_residual']:.3g}",f"{s['worst_pressure_relative_l2']:.3g}"])
    payload['blind_evaluations']=blind_evidence
    if blind:sections.append('<h2>Reserved blind native requests</h2><p>Whole scenes generated only after development selection. Four independent grid/spacing trajectories share one new forcing profile; their cold/warm snapshots give eight diagnostic requests. This is a small blind comparison, not broad application certification.</p>'+table(['Model','Pass ceiling','Gates passed','Worst residual','Worst pressure rel. L2'],blind))
    quiet=[]
    for path in sorted((base/'quiet').glob('*.json')):
        d=read(path);quiet.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),**d))
    payload['quiet_benchmarks']=quiet
    if (docs/'figures/historical-frontier.svg').exists():
        sections.append('<h2>Common-population accuracy and inference cost</h2><p>One thread on CPU 23, one complete warmup and three repeats per case. These are the same 128 historical requests on 5–16-cell grids, not 128³ domains. They measure retention and small-grid cost; larger native accuracy remains in the separate tables above.</p><img src="figures/historical-frontier.svg" alt="Common historical population pressure error versus inference time" style="width:100%;max-width:1100px">')
    if (docs/'figures/native-coverage.svg').exists():
        sections.append('<h2>Native development and blind coverage</h2><img src="figures/native-coverage.svg" alt="Native cold and warm request qualification by model and correction ceiling" style="width:100%;max-width:1100px">')
    audit=docs/'final-audit.json'
    if audit.exists():payload['status']=read(audit)['status'];payload['final_audit_sha256']=sha(audit)
    completion=docs/'completion-audit.json'
    if completion.exists():
        payload['status']=read(completion)['status'];payload['completion_audit_sha256']=sha(completion)
    findings='<p>The native 64³/128³ fit passed all eight reserved blind pressure requests using at most five CNN corrections. On the four-step 128³ coupled scene, median fluid-step time fell from 21.43 s for the previous learned method to 11.43 s; spectral pressure remained faster at 3.37 s. The new fit did not improve the historical small-grid accuracy/speed frontier.</p><p>Both genuinely reduced-width fits failed the native pressure gates. That result applies to this pruning initialization and training budget, and does not establish a minimum required network size. Velocity advection was the largest measured remaining preparation phase; a sampler change reduced the matched median fluid-step time by 2.1%, a modest signal across three paired trials. The native training recipe used about 10.45 times the replay control’s optimizer-forward cells. Four-step checks leave long-horizon drift and 256³-plus domains unqualified.</p>'
    body='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Pressure methods v3</title><style>body{font:16px system-ui;max-width:1400px;margin:40px auto;padding:0 24px;color:#192936;background:#f8fafb}h1,h2{color:#123e54}table{border-collapse:collapse;width:100%;margin:20px 0 36px;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ccd6dc;text-align:left}th{background:#e4edf2}p{max-width:1000px;line-height:1.5}code{overflow-wrap:anywhere}</style><h1>Pressure methods v3</h1>'+findings+'<p>Retained experimental evidence. Original C37/C44/C53 captures remain immutable. No model promotion. Drift thresholds remain diagnostic. Prepared-request timing and coupled fluid timing are reported separately.</p>'+''.join(sections)+'<p>Machine-readable evidence: <a href="results.json">results.json</a>. Audits: <a href="final-audit.json">custody and evidence</a>, <a href="completion-audit.json">objective completion</a>. Prior campaign: <a href="../pressure-methods-v2/results.html">v2</a>.</p></html>'
    (docs/'results.html').write_text(body);write(docs/'results.json',payload)
    print('Rendered v3 evidence report',len(evaluations),'evaluated checkpoints,',len(fits),'managed fits',flush=True)

if __name__=='__main__':build()
