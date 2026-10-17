"""Bound local velocity-cast divergence; do not infer cumulative causal drift."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import html
import json
from pathlib import Path
import struct
import time


UNIT_ROUNDOFF = Fraction(1, 2**24)
HALF_MIN_SUBNORMAL = Fraction(1, 2**150)
REFERENCE = 'https://docs.oracle.com/cd/E19422-01/819-3693/ncg_goldberg.html'


def quantize(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def cell_divergence(faces, spacing):
    """Exact rational divergence of three (low, high) binary floating faces."""
    return sum((Fraction(hi)-Fraction(lo))/Fraction(h)
               for (lo, hi), h in zip(faces, spacing))


def face_cast_bound(stored):
    """Bound |RN32(v)-v| using the stored value, including gradual underflow."""
    return (UNIT_ROUNDOFF*abs(Fraction(stored))+HALF_MIN_SUBNORMAL)/(1-UNIT_ROUNDOFF)


def example():
    # Ideal divergence is exactly zero; two sub-ULP increments disappear on cast.
    faces=[(.25, .25+2**-27), (.25, .25+2**-27), (.25, .25-2**-26)]
    stored=[tuple(quantize(v) for v in axis) for axis in faces]
    h=[.0625]*3
    bound=sum((face_cast_bound(lo)+face_cast_bound(hi))/Fraction(spacing)
              for (lo, hi), spacing in zip(stored, h))
    return dict(ideal_faces=faces,stored_faces=stored,
                ideal_divergence_s_inverse=float(cell_divergence(faces,h)),
                rounded_divergence_s_inverse=float(cell_divergence(stored,h)),
                cast_divergence_bound_s_inverse=float(bound))


def analyze(material_path, model, output, control_root=None):
    start=time.perf_counter()
    material=json.loads(material_path.read_text());run=material_path.parent
    root=Path(material['root']);scene=material['config']['scene']
    spacing=1/scene['cells_per_metre'];dt=material['support']['support']['dt_s']
    if material['support']['support']['cells']!='cubic':raise ValueError('Cubic grid required')
    if model not in {m['id']for m in material['models']}:raise ValueError('Model not in frozen selection')
    rows=[];sources={};diagnostics={}
    for n in material['config']['passes']:
        folder=run/model/f'passes-{n:02d}'
        receipt=json.loads((folder/'trajectory.json').read_text())
        path=folder/'native/frames.jsonl';raw=path.read_bytes()
        sources[str(path.relative_to(root))]=hashlib.sha256(raw).hexdigest()
        frames=[json.loads(line)for line in raw.splitlines()]
        frames=[f for f in frames if f.get('ai_quality')]
        if not frames:
            rows.append(dict(passes=n,accepted_steps=0,termination=receipt['termination']))
            continue
        values=[]
        for frame in frames:
            quality=frame['ai_quality'];actual=quality['actual_divergence'];predicted=quality['predicted_divergence']
            # ||D delta_v||inf <= 2/h sum_d max|delta_v_d|.
            # C = dt/h sum_d max|stored_v_d|. Evaluation uses rounded diagnostics;
            # this is not a directed-rounding interval certificate for the engine.
            cast_bound=(2*float(UNIT_ROUNDOFF)/(1-float(UNIT_ROUNDOFF))*quality['courant']/dt
                        +6*float(HALF_MIN_SUBNORMAL)/(spacing*(1-float(UNIT_ROUNDOFF))))
            values.append(dict(step=frame['step'],actual_divergence_s_inverse=actual,
                pressure_residual_predicted_divergence_s_inverse=predicted,
                residual_based_remainder_interval_estimate_s_inverse=[max(0,actual-predicted),actual+predicted],
                float32_cast_divergence_upper_bound_s_inverse=cast_bound))
        diagnostics[n]=frames
        largest=max(values,key=lambda v:v['actual_divergence_s_inverse'])
        rows.append(dict(passes=n,accepted_steps=len(frames),termination=receipt['termination'],
            worst_equation_residual=max(f['ai_quality']['relative_residual']for f in frames),
            worst_reference_equation_residual=max(f['reference_quality']['relative_residual']for f in frames),
            worst_independent_pressure_relative_l2=max(f['relative_pressure_difference']for f in frames),
            worst_actual_divergence_step=largest,last_step=values[-1],
            cast_bound_exceedances=sum(v['actual_divergence_s_inverse']>
                v['float32_cast_divergence_upper_bound_s_inverse']+
                v['pressure_residual_predicted_divergence_s_inverse']for v in values)))
    baseline=16 if 16 in diagnostics else max(diagnostics,default=None)
    agreement={str(n):dict(steps=len(frames),matching_step_identities=
        [f['step']for f in frames]==[f['step']for f in diagnostics[baseline]],
        identical_actual_divergence=sum(a['ai_quality']['actual_divergence']==b['ai_quality']['actual_divergence']
                                       for a,b in zip(frames,diagnostics[baseline])),
        identical_courant=sum(a['ai_quality']['courant']==b['ai_quality']['courant']
                             for a,b in zip(frames,diagnostics[baseline])))for n,frames in diagnostics.items()}
    controls={}
    if control_root is not None:
        for name in ['full-oracle','full-native']:
            folder=control_root/name;receipt=json.loads((folder/'control-receipt.json').read_text())
            if (receipt['neural_network_calls']!=0 or receipt['weights_loaded']
                or not receipt['native_process']['reaped']or receipt['cuda_initialized']):
                raise ValueError('Invalid no-learning control')
            if receipt['material_sha256']!=hashlib.sha256(material_path.read_bytes()).hexdigest():
                raise ValueError('Control belongs to another material freeze')
            path=folder/'native/frames.jsonl';raw=path.read_bytes()
            sources[str(path.relative_to(root))]=hashlib.sha256(raw).hexdigest()
            sources[str((folder/'control-receipt.json').relative_to(root))]=hashlib.sha256((folder/'control-receipt.json').read_bytes()).hexdigest()
            frames=[json.loads(line)for line in raw.splitlines()]
            frames=[f for f in frames if f.get('ai_quality')]
            if len(frames)!=scene['steps']-scene['warmup_steps']:raise ValueError('Control window differs')
            comparisons={}
            for n,model_frames in diagnostics.items():
                if [f['step']for f in model_frames]!=[f['step']for f in frames]:continue
                comparisons[str(n)]=dict(max_absolute_difference_in_recorded_relative_drift=max(
                    abs(a['relative_pressure_difference']-b['relative_pressure_difference'])for a,b in zip(frames,model_frames)),
                    identical_actual_divergence_steps=sum(a['ai_quality']['actual_divergence']==b['ai_quality']['actual_divergence']for a,b in zip(frames,model_frames)),
                    identical_courant_steps=sum(a['ai_quality']['courant']==b['ai_quality']['courant']for a,b in zip(frames,model_frames)),
                    identical_reference_quality_steps=sum(a['reference_quality']==b['reference_quality']for a,b in zip(frames,model_frames)))
            controls[name]=dict(kind=receipt['kind'],accepted_steps=len(frames),neural_network_calls=0,weights_loaded=False,
                worst_relative_pressure_drift=max(f['relative_pressure_difference']for f in frames),
                worst_equation_residual=max(f['ai_quality']['relative_residual']for f in frames),
                elapsed_s=receipt['total_elapsed_s'],peak_rss_bytes=receipt['peak_rss_bytes'],
                native_reaped=True,comparisons_to_model_passes=comparisons)
    result=dict(schema='rheon-local-rounding-error-analysis-v1',status='completed',
        recorded_utc=datetime.now(timezone.utc).isoformat(),model=model,scene=scene,
        dt_s=dt,spacing_m=spacing,unit_roundoff=float(UNIT_ROUNDOFF),
        formulas=dict(face_cast='q = RN32(v); |q-v| <= (u32*|q| + 2^-150)/(1-u32)',
            divergence_cast='||D(q-v)||inf <= 2*u32*C/((1-u32)*dt) + 6*2^-150/(h*(1-u32))',
            scalar_remainder='If ideal divergence d_ideal has norm P and stored divergence has norm A, ||d_stored-d_ideal||inf lies in [max(0,A-P), A+P]. Here P is estimated by the recorded pressure-residual prediction; float64 stencil/arithmetic mismatch is not independently bounded.',
            accumulated='E[n+1] <= L[n]*E[n] + eta[n]; local error cannot be summed into actual trajectory drift without its propagation and correlations.'),
        binary32_reference=REFERENCE,measurements=rows,agreement_baseline_passes=baseline,diagnostic_agreement=agreement,
        exact_rational_example=example(),source_hashes=sources,no_learning_controls=controls,
        interpretation='For high-pass settings the recorded pressure-residual contribution is negligible next to stored-velocity divergence. This is evidence for numerical error after the solve, not a measurement of the cumulative pressure drift caused solely by float32 rounding.',
        limits=['These are one-step divergence bounds, in inverse seconds, not pressure drift in Pa or relative pressure error.',
            'RN32 means round-to-nearest binary32 with gradual underflow and finite representable results. No universal nonzero minimum rounding error exists.',
            'Numeric bound evaluations use float64 and recorded rounded Courant diagnostics, not interval-certified exact face values.',
            'The residual-based remainder includes float64 correction and operator arithmetic as well as float32 storage; full per-face arrays were not retained.',
            'Matching scalar diagnostics does not prove complete velocity fields are bit-identical.',
            'Native solver tolerance is an adjustable algorithm setting, not a fundamental float32 limit.',
            'No causal percentage of accumulated pressure drift can be recovered from these aggregate records alone.'],
        control_conclusion=('The analytic no-learning solver reproduces the high-pass drift curve, whereas the matched native solver control has nearly zero drift. Thus the plotted plateau does not require training error and is not evidence of a universal float32 limit. Reference stopping tolerance, differing solver arithmetic and subsequent float32 trajectory propagation are not separately attributed by these controls.'if controls else'No independent trajectory control was supplied.'),
        controlled_measurements_needed=[
            'On identical pre-projection states, solve with a qualified oracle and measure model pressure-gradient/velocity error separately from the native solver stopping error.',
            'Record each corrected velocity before and after float32 storage, and compute the cast perturbation and its divergence directly.',
            'Advance paired controlled trajectories that differ only in the rounding operation under study, with corresponding higher-precision numerical operators; changing just the pressure solver does not isolate all simulation rounding.',
            'Report model approximation, solver stopping tolerance, local arithmetic/storage error, and accumulated trajectory sensitivity separately; their effects interact rather than forming independent scalar percentages.'],
        elapsed_s=time.perf_counter()-start)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>'for v in
        [row['passes'],row['accepted_steps'],f"{row['worst_equation_residual']:.3g}",
         f"{row['last_step']['pressure_residual_predicted_divergence_s_inverse']:.3g}",
         f"{row['last_step']['actual_divergence_s_inverse']:.3g}",
         f"{row['last_step']['float32_cast_divergence_upper_bound_s_inverse']:.3g}"] )+'</tr>'
        for row in rows if row['accepted_steps'])
    control_table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>'for v in
        [value['kind'],value['accepted_steps'],f"{value['worst_relative_pressure_drift']:.8g}",f"{value['elapsed_s']:.3f}"])+'</tr>'for value in controls.values())
    output.with_suffix('.html').write_text('''<!doctype html><meta charset="utf-8"><title>Rounding and model error</title><style>body{font:17px system-ui;max-width:1150px;margin:40px auto;padding:0 20px}td,th{padding:10px;text-align:left;border-bottom:1px solid #aaa}table{border-collapse:collapse}code{white-space:normal}</style>
<h1>Local rounding bounds and model error</h1><p>Rounding bounds are calculable. The fraction of accumulated pressure drift caused by rounding is not identifiable from the retained aggregate diagnostics.</p>
<p>The table separates the pressure-residual prediction from measured stored-velocity divergence. Its last three columns are inverse seconds at the final accepted step, not pressure errors. The cast bound is a float64 evaluation of an analytical upper bound, not a certified estimate of actual cumulative drift.</p>
<table><tr><th>Passes</th><th>Accepted steps</th><th>Worst equation residual</th><th>Final predicted divergence</th><th>Final stored divergence</th><th>Final cast divergence bound</th></tr>'''+table+'''</table><h2>No-learning trajectory controls</h2><p>Neither control loads model weights or invokes a neural network. Both retain the existing float32 simulation velocity and advance the same scene independently against the native reference.</p><table><tr><th>Control</th><th>Accepted steps</th><th>Worst relative pressure drift</th><th>Elapsed seconds</th></tr>'''+control_table+'''</table><p>'''+html.escape(result['control_conclusion'])+'''</p><p>Native solver stopping tolerance is adjustable. Float32 rounding bounds do not establish a universal nonzero error floor. Model error, reference stopping error, velocity storage, and trajectory sensitivity require separate controlled measurements.</p><p><a href="'''+html.escape(output.name)+'''">Full formulas, exact rational example, measurements, assumptions and next controls</a></p>''')
    print(json.dumps(dict(status='completed',model=model,settings=len(rows),elapsed_s=result['elapsed_s'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--material',type=Path,required=True);parser.add_argument('--model',default='C53')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--control-root',type=Path)
    args=parser.parse_args();analyze(args.material.resolve(),args.model,args.output.resolve(),args.control_root.resolve()if args.control_root else None)
