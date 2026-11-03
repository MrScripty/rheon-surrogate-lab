"""Serialize the same learned sampler with the validated alpha launch argument."""
import argparse,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def generate(protocol):
 reg=json.loads(protocol.read_text())
 for p,h in reg['source_sha256'].items():
  if digest(ROOT/p)!=h:raise ValueError('Frozen alpha-argument source changed: '+p)
 source,output,receipt=(ROOT/reg[k] for k in ('template_source','output_source','output_receipt'))
 if output.exists() or receipt.exists():raise ValueError('Fresh generated source required')
 text=source.read_text()
 if text.count('const double alpha=dt/h;')!=2:raise ValueError('Exactly generic and specialized parent quotients required')
 text=re.sub(r'\bdt\b','step_over_spacing',text)
 text=text.replace('const double alpha=step_over_spacing/h;','const double alpha=step_over_spacing;')
 text+='\n// Scalar controls: no learned arithmetic. Caller uses existing validated alpha.\n'
 text+='extern "C" __global__ void alpha_argument_array(long long count,const double* dt,const double* h,const double* supplied,unsigned long long* quotient,unsigned long long* passed) {\n long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;\n quotient[i]=__double_as_longlong(__ddiv_rn(dt[i],h[i]));passed[i]=__double_as_longlong(supplied[i]);\n}\nextern "C" __global__ void alpha_argument_scalar(double dt,double h,double supplied,unsigned long long* bits) {\n if(blockIdx.x||threadIdx.x)return;bits[0]=__double_as_longlong(__ddiv_rn(dt,h));bits[1]=__double_as_longlong(supplied);\n}\n'
 output.write_text('// Exact alpha reuse: transport ABI supplies validated binary64 dt/h, not dt.\n'+text)
 receipt.write_text(json.dumps(dict(status='generated_exact_alpha_argument_sampler',template_source=str(source.relative_to(ROOT)),template_source_sha256=digest(source),generated_source=str(output.relative_to(ROOT)),generated_source_sha256=digest(output),manifest_sha256=digest(ROOT/reg['sampler_parameter_manifest']),protocol_sha256=digest(protocol),learned_parameters_changed=False,original_N1_path_unchanged=True,arithmetic='All existing F64 point/gate/mix/coordinate/interpolation operations unchanged; validated caller alpha replaces redundant GPU division only.',sampler_alpha_argument='validated_existing_RN64_dt_over_h',cpu_model_forward=False,optimizer_updates=0),indent=2)+'\n')
 print('Generated source with exactly two redundant sampler divisions removed',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);generate(p.parse_args().protocol)
