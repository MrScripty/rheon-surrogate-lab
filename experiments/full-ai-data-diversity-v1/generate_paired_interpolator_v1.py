"""One compensated FP32-pair interpolation source; learned values are untouched."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(protocol):
    reg = json.loads(protocol.read_text())
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Frozen pair-interpolation dependency changed: '+path)
    source, output, receipt = (ROOT/reg[k] for k in ('template_source', 'output_source', 'output_receipt'))
    if output.exists() or receipt.exists():
        raise ValueError('Fresh generated trial required')
    text = source.read_text()
    signature = '__device__ double nested_donors(const float* d,double fx,double fy,double fz,bool clip,int& fallback) {'
    if text.count(signature) != 1:
        raise ValueError('Exact qualified F64 donor implementation required')
    text = text.replace(signature, signature.replace('nested_donors(', 'parent_nested_donors('))
    marker = '__device__ double sample(const float* a,Shape s,double x,double y,double z,bool clip) {'
    trial = '''struct InterpolationPair {float hi,lo;};
__device__ __forceinline__ InterpolationPair pair_two_sum(float a,float b) {
 float s=__fadd_rn(a,b),v=__fsub_rn(s,a);
 float e=__fadd_rn(__fsub_rn(a,__fsub_rn(s,v)),__fsub_rn(b,v));
 return {s,e};
}
__device__ __forceinline__ InterpolationPair pair_add(InterpolationPair a,InterpolationPair b) {
 InterpolationPair s=pair_two_sum(a.hi,b.hi),t=pair_two_sum(a.lo,b.lo);
 InterpolationPair v=pair_two_sum(s.hi,__fadd_rn(s.lo,t.hi));
 return pair_two_sum(v.hi,__fadd_rn(v.lo,t.lo));
}
__device__ __forceinline__ InterpolationPair pair_multiply(InterpolationPair a,InterpolationPair b) {
 float p=__fmul_rn(a.hi,b.hi),e=__fmaf_rn(a.hi,b.hi,-p);
 e=__fadd_rn(e,__fmul_rn(a.hi,b.lo));
 e=__fadd_rn(e,__fmul_rn(a.lo,b.hi));
 e=__fadd_rn(e,__fmul_rn(a.lo,b.lo));
 return pair_two_sum(p,e);
}
__device__ __forceinline__ InterpolationPair pair_fraction(double f) {
 float hi=__double2float_rn(f);
 return {hi,__double2float_rn(__dsub_rn(f,double(hi)))};
}
__device__ __forceinline__ InterpolationPair pair_lerp(InterpolationPair a,InterpolationPair b,
 InterpolationPair f,double original_f) {
 if(original_f==0.)return a;
 if(original_f==1.)return b;
 InterpolationPair neg={-a.hi,-a.lo};
 return pair_add(a,pair_multiply(pair_add(b,neg),f));
}
__device__ double nested_donors(const float* d,double fx,double fy,double fz,bool clip,int& fallback) {
 bool eligible=fx>=0.&&fx<=1.&&fy>=0.&&fy<=1.&&fz>=0.&&fz<=1.;
 #pragma unroll
 for(int i=0;i<8;++i) {
  unsigned bits=__float_as_uint(d[i])&0x7fffffffU;
  eligible=eligible&&(bits==0U||(d[i]>=0x1p-60f&&d[i]<=0x1p60f)||(d[i]<=-0x1p-60f&&d[i]>=-0x1p60f));
 }
 if(!eligible){int ignored;double v=parent_nested_donors(d,fx,fy,fz,clip,ignored);fallback=1;return v;}
 fallback=0;
 InterpolationPair x=pair_fraction(fx),y=pair_fraction(fy),z=pair_fraction(fz);
 InterpolationPair a0={d[0],0.f},a1={d[1],0.f},a2={d[2],0.f},a3={d[3],0.f};
 InterpolationPair a4={d[4],0.f},a5={d[5],0.f},a6={d[6],0.f},a7={d[7],0.f};
 InterpolationPair x0=pair_lerp(a0,a1,x,fx),x1=pair_lerp(a2,a3,x,fx);
 InterpolationPair x2=pair_lerp(a4,a5,x,fx),x3=pair_lerp(a6,a7,x,fx);
 InterpolationPair v=pair_lerp(pair_lerp(x0,x1,y,fy),pair_lerp(x2,x3,y,fy),z,fz);
 double value=v.lo==0.f?double(v.hi):__dadd_rn(double(v.hi),double(v.lo));
 if(clip){double lower=double(d[0]),upper=lower;
  #pragma unroll
  for(int i=1;i<8;++i){double t=double(d[i]);lower=t<lower?t:lower;upper=t>upper?t:upper;}
  value=value<lower?lower:value;value=value>upper?upper:value;
 }
 return value;
}
'''
    if text.count(marker) != 1:
        raise ValueError('Exact qualified generic sample entry required')
    output.write_text('// N8-only compensated FP32-pair donor trial; native scoring required.\n'+text.replace(marker, trial+marker))
    manifest = ROOT/reg['sampler_parameter_manifest']
    receipt.write_text(json.dumps(dict(status='generated_single_compensated_FP32_pair_trial',
        template_source=str(source.relative_to(ROOT)), template_source_sha256=digest(source),
        generated_source=str(output.relative_to(ROOT)), generated_source_sha256=digest(output),
        manifest_sha256=digest(manifest), protocol_sha256=digest(protocol),
        learned_parameters_changed=False, original_N1_path_unchanged=True,
        donor_arithmetic='Explicit RN two-sum, compensated pair addition and FMA product residual; F64 coordinates/fraction splitting and final pair reconstruction',
        numeric_guard='All donors zero or magnitudes in[2**-60,2**60], finite fractions in[0,1]; otherwise unchanged parent F64 interpolation',
        model_forward_executed=False, cpu_model_forward=False, optimizer_updates=0), indent=2)+'\n')
    print('Generated one compensated FP32-pair interpolation candidate', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    generate(parser.parse_args().protocol)
