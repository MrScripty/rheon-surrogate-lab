"""Publish only completed, controlled precision measurements into the dashboard."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
from build import read,sha


COLUMNS=['step','time_s','pressure_rms_pa','pressure_relative_l2','velocity_rms_m_s','velocity_relative_l2',
    'tracer_rms','tracer_relative_l2','advection_cast_rms','force_cast_rms','viscosity_cast_rms','projection_cast_rms',
    'tracer_cast_rms','projection_div_before','projection_div_after','projection_div_cast','projection_cast_max',
    'projection_cast_bound','pressure64_rms_pa','velocity64_rms_m_s','native32_equation_residual','native64_equation_residual']


def compact(value):return float(format(value,'.9g'))if isinstance(value,float)else value


def collect(build,run,output,historical_reference=None):
    material=read(build/'material.json');root=Path(material['root']);execution=read(run/'execution.json')
    if execution['status']!='completed'or not execution['all_owned_children_reaped']:raise ValueError('Precision workers have not completed and drained')
    sources={};traces=[];summaries=[]
    def source(path):
        key=os.path.relpath(path,root/'docs/plans');sources[key]=sha(path);return key
    source(build/'material.json');source(build/'build-receipt.json');source(run/'execution.json')
    for row in material['scenes']:
        folder=run/row['id'];receipt=read(folder/'receipt.json');path=folder/'frames.jsonl'
        if receipt['frames_sha256']!=sha(path):raise ValueError('Precision frame source changed')
        values=[json.loads(line)for line in path.read_bytes().splitlines()];frames=values[:-1];complete=values[-1]
        if complete['kind']!='complete'or len(frames)!=row['scene']['steps']:raise ValueError('Missing precision steps')
        points=[]
        for frame in frames:
            if not(frame['canonical_f32_exact']and frame['f64_repeat_exact']and frame['f32_quality']['qualified']and frame['f64_quality']['qualified']):
                raise ValueError('Native/precision/null qualification failed')
            casts=frame['casts']
            if any(casts[name]['cast_mismatches']for name in ['advection','tracer_advection','force','viscosity','projection']):
                raise ValueError('Local cast did not reproduce native result')
            points.append([frame['step'],frame['time_s'],frame['pressure']['rms'],frame['pressure']['relative_l2'],
                frame['velocity']['rms'],frame['velocity']['relative_l2'],frame['tracer']['rms'],frame['tracer']['relative_l2'],
                casts['advection']['rms'],casts['force']['rms'],casts['viscosity']['rms'],casts['projection']['rms'],casts['tracer_advection']['rms'],
                casts['projection_divergence_before_cast'],casts['projection_divergence_after_cast'],casts['projection_cast_divergence'],
                casts['projection']['maximum'],casts['projection']['maximum_bound'],frame['pressure']['reference_rms'],frame['velocity']['reference_rms'],
                frame['f32_quality']['relative_residual'],frame['f64_quality']['relative_residual']])
        reference_match=None
        if row['id']=='matched16'and historical_reference is not None:
            path_original=historical_reference
            original=[json.loads(line)for line in path_original.read_bytes().splitlines()]
            if len(original)!=len(frames)or any(a['step']!=b['step']or a['reference_quality']!=b['f32_quality']for a,b in zip(original,frames)):
                raise ValueError('Float32 native result differs from historical baseline')
            reference_match=True;source(path_original)
        summary=dict(id=row['id'],label=row['label'],steps=len(frames),elapsed_s=receipt['elapsed_s'],
            peak_rss=complete['max_rss_kib'],verified_cast_values=complete['verified_cast_values'],
            worst_pressure_rms_pa=max(f['pressure']['rms']for f in frames),
            worst_velocity_rms_m_s=max(f['velocity']['rms']for f in frames),
            worst_pressure_relative_l2=max((f['pressure']['relative_l2']for f in frames if f['pressure']['relative_l2']is not None),default=None),
            worst_projection_cast_divergence=max(f['casts']['projection_cast_divergence']for f in frames),
            worst_projection_cast_velocity=max(f['casts']['projection']['maximum']for f in frames),
            canonical_f32_exact_steps=complete['canonical_f32_exact_steps'],f64_repeat_exact_steps=complete['f64_repeat_exact_steps'],
            historical_native_quality_exact=reference_match,receipt_source=source(folder/'receipt.json'),frames_source=source(folder/'frames.jsonl'))
        traces.append(dict(id=row['id'],label=row['label'],scene=row['scene'],points=[[compact(v)for v in p]for p in points],summary=summary))
        summaries.append(summary)
    if any(s['id']=='zero'and(s['worst_pressure_rms_pa']!=0 or s['worst_velocity_rms_m_s']!=0)for s in summaries):
        raise ValueError('Zero-forcing control acquired artificial drift')
    result=dict(schema='rheon-storage-precision-chart-v1',status='completed',updated_utc=datetime.now(timezone.utc).isoformat(),
        point_columns=COLUMNS,traces=traces,summaries=summaries,execution={k:execution[k]for k in ['status','elapsed_s','maximum_workers','all_owned_children_reaped']},
        verified_cast_values=sum(s['verified_cast_values']for s in summaries),sources=sources,
        method='Same Rheon equations, pressure solver, solver tolerance and initial state. Pressure/intermediate arithmetic remains float64. Only velocity/tracer storage changes between float32 and float64, beginning at the first step; no shared native warm-up is imposed. No neural network is used. Local cast measurements replay each stage on identical float32 inputs, including identical pressure for the projection.',
        limits='The independent trajectory difference measures the effect of storage precision under this fixed numerical algorithm and stopping policy. Float64 is a comparison, not exact real arithmetic. Changed precision may change iteration paths. This does not measure network float32-versus-float64 inference or establish a universal float32 error floor.',
        fixed=dict(density_kg_m3=1000,dynamic_viscosity_pa_s=.001,dt_s=1/60,cells='cubic',boundaries='impermeable stationary free-slip walls',pressure_dtype='float64',pressure_relative_tolerance=1e-8,pressure_absolute_tolerance=1e-12))
    output.mkdir(parents=True,exist_ok=True)
    (output/'chart-data.json').write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
    summary=dict(schema=result['schema'],status='completed',method=result['method'],limits=result['limits'],fixed=result['fixed'],
        summaries=summaries,verified_cast_values=result['verified_cast_values'],execution=execution,build=read(build/'build-receipt.json'),sources=sources,
        selection_chart_sha256=material['selection_chart_sha256'],commands=[
            ['python3','experiments/pressure-precision-drift-v1/build.py','--config','experiments/pressure-precision-drift-v1/config.json','--output',str(build.relative_to(root))],
            ['python3','experiments/pressure-precision-drift-v1/run.py','--build',str(build.relative_to(root)),'--output',str(run.relative_to(root))],
            ['python3','experiments/pressure-precision-drift-v1/collect.py','--build',str(build.relative_to(root)),'--run',str(run.relative_to(root)),'--output',str(output.relative_to(root))]
            + (['--historical-reference',str(historical_reference.relative_to(root))]if historical_reference is not None else [])])
    (output/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed',scenes=len(traces),steps=sum(len(t['points'])for t in traces),verified_cast_values=result['verified_cast_values'])))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--historical-reference',type=Path)
    args=p.parse_args();collect(args.build.resolve(),args.run.resolve(),args.output.resolve(),args.historical_reference.resolve()if args.historical_reference else None)
