"""Freeze one guarded FP32 donor-interpolation trial from the qualified parent."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(protocol):
    reg = json.loads(protocol.read_text())
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen interpolation dependency changed: '+name)
    source, output, receipt = (ROOT/reg[k] for k in ('template_source', 'output_source', 'output_receipt'))
    if output.exists() or receipt.exists():
        raise ValueError('Fresh generated trial required')
    text = source.read_text()
    signature = '__device__ double nested_donors(const float* d,double fx,double fy,double fz,bool clip,int& fallback) {'
    if text.count(signature) != 1:
        raise ValueError('Expected exact qualified donor implementation')
    text = text.replace(signature, signature.replace('nested_donors(', 'parent_nested_donors('))
    marker = '__device__ double sample(const float* a,Shape s,double x,double y,double z,bool clip) {'
    trial = '''// Explicit FP32 interior arithmetic; coordinates and learned values stay F64.
__device__ __forceinline__ float trial_lerp(float a,float b,double f) {
 if(f==0.)return a;
 if(f==1.)return b;
 return __fmaf_rn(float(f),__fsub_rn(b,a),a);
}
__device__ double nested_donors(const float* d,double fx,double fy,double fz,bool clip,int& fallback) {
 bool eligible=fx>=0.&&fx<=1.&&fy>=0.&&fy<=1.&&fz>=0.&&fz<=1.;
 #pragma unroll
 for(int i=0;i<8;++i) {
  eligible=eligible&&((__float_as_uint(d[i])&0x7f800000U)!=0x7f800000U);
  eligible=eligible&&d[i]>=-0x1p125f&&d[i]<=0x1p125f;
 }
 if(!eligible)return parent_nested_donors(d,fx,fy,fz,clip,fallback);
 fallback=0;
 float x0=trial_lerp(d[0],d[1],fx),x1=trial_lerp(d[2],d[3],fx);
 float x2=trial_lerp(d[4],d[5],fx),x3=trial_lerp(d[6],d[7],fx);
 double value=double(trial_lerp(trial_lerp(x0,x1,fy),trial_lerp(x2,x3,fy),fz));
 if(clip){double lower=double(d[0]),upper=lower;
  #pragma unroll
  for(int i=1;i<8;++i){double v=double(d[i]);lower=v<lower?v:lower;upper=v>upper?v:upper;}
  value=value<lower?lower:value;value=value>upper?upper:value;
 }
 return value;
}
'''
    if text.count(marker) != 1:
        raise ValueError('Expected exact generic sample entry')
    text = text.replace(marker, trial+marker)
    output.write_text('// N8-only FP32 donor interpolation trial; fresh native scoring required.\n'+text)
    manifest = ROOT/reg['sampler_parameter_manifest']
    receipt.write_text(json.dumps(dict(status='generated_single_FP32_interpolator_trial',
        template_source=str(source.relative_to(ROOT)), template_source_sha256=digest(source),
        generated_source=str(output.relative_to(ROOT)), generated_source_sha256=digest(output),
        manifest_sha256=digest(manifest), protocol_sha256=digest(protocol),
        learned_parameters_changed=False, original_N1_path_unchanged=True,
        donor_arithmetic='FP32 explicit RN subtraction/FMA; F64 fraction cast at each lerp; F64 result and clipping',
        numeric_guard='Nonfinite/invalid fractions or any donor outside +/-2**125 use unchanged parent interpolation',
        model_forward_executed=False, cpu_model_forward=False, optimizer_updates=0), indent=2)+'\n')
    print('Generated one guarded FP32 donor trial; no accuracy or latency claim', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    generate(parser.parse_args().protocol)
