"""Serialize a source-bound managed-restored sampler, preserving ordered math.

No model forward or fitting. All coefficient literals derive from the exact
deployed parameter manifest, never from an analytical physical operator.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[2]


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    reg = json.loads(args.protocol.read_text())
    for p, h in reg['source_sha256'].items():
        if digest(ROOT/p) != h: raise ValueError('Frozen codegen dependency changed: '+p)
    output = ROOT/reg['output_source']; receipt = ROOT/reg['output_receipt']
    if output.exists() or receipt.exists(): raise ValueError('Fresh generated source and provenance required')
    manifest = json.loads((ROOT/reg['manifest']).read_text())
    if manifest['status'] != 'exported_exact_deployed_sampler_parameter_inventory': raise ValueError('Exact deployed source manifest required')
    t = manifest['tensors']; point = t['point']; coefficients = point['coefficients']
    records = [coefficients, t['mid_gate'], t['mix']]
    for record in records:
        values = [float.fromhex(v) for v in record['values_hex']]
        if record['dtype'] != '<f8' or len(values) != math.prod(record['shape']) or not all(map(math.isfinite, values)):
            raise ValueError('Exact finite deployed F64values required')
        bits = [struct.unpack('<Q', struct.pack('<d', v))[0] for v in values]
        if bits != record['bits_u64'] or hashlib.sha256(b''.join(struct.pack('<d', v) for v in values)).hexdigest() != record['sha256']:
            raise ValueError('Hex serialization changed coefficient bytes')
    if len(point['rows']) != 13 or point['rows'][0] != 0 or point['rows'][-1] != len(point['channels']) or len(point['offsets']) != len(point['channels']):
        raise ValueError('Complete12head learned CSRmanifest required')
    if any(type(c) is not int or not 0 <= c < 3 for c in point['channels']) or any(len(o)!=3 or any(type(v)is not int or not -1<=v<=1 for v in o) for o in point['offsets']):
        raise ValueError('Exact declared learned local metadata required')
    def literal_function(name, values):
        return '__device__ __forceinline__ double '+name+'(int i) {\n switch(i) {\n'+''.join(' case '+str(i)+':return '+float.fromhex(v).hex()+';\n' for i,v in enumerate(values))+' default:return __longlong_as_double(0x7ff8000000000000ULL);\n }\n}\n'
    generated = '\n// Serialized managed learned parameters, manifest SHA256 '+digest(ROOT/reg['manifest'])+'\n'
    generated += literal_function('bound_point_weight', coefficients['values_hex'])
    generated += literal_function('bound_gate', t['mid_gate']['values_hex'])
    generated += literal_function('bound_mix', t['mix']['values_hex'])
    generated += 'template<int HEAD>\n__device__ __forceinline__ double bound_point_head(const float*const* a,const Shape* shapes,int x,int y,int z) {\n double total=0.;\n'
    for head in range(12):
        generated += (' if' if head == 0 else ' else if')+' constexpr(HEAD=='+str(head)+') {\n'
        for i in range(point['rows'][head],point['rows'][head+1]):
            c = point['channels'][i]; x,y,z = point['offsets'][i]
            generated += f'  total=total+bound_point_weight({i})*read_float(a[{c}],shapes[{c}],x+({x}),y+({y}),z+({z}));\n'
        generated += ' }\n'
    generated += ' return total;\n}\n'
    original = (ROOT/reg['template_source']).read_text()
    a = original.index('__device__ double point_linear('); b = original.index('extern "C" __global__ void learned_transport', a)
    dynamic = original[a:b].replace('__device__ double point_linear(', '__device__ double dynamic_point_reference(', 1)
    dispatch = '__device__ double point_linear(const float*const* a,const Shape* shapes,int head,int x,int y,int z,\n const int* rows,const int* channels,const int* offsets,const double* weights) {\n switch(head) {\n'
    dispatch += ''.join(f' case {h}:return bound_point_head<{h}>(a,shapes,x,y,z);\n' for h in range(12))
    dispatch += ' default:return __longlong_as_double(0x7ff8000000000000ULL);\n }\n}\n'
    source = original[:a]+generated+dynamic+dispatch+original[b:]
    a = source.index('template<int NX,int NY,int NZ>\n__device__ __forceinline__ double geometry_point(')
    b = source.index('template<int NX,int NY,int NZ,int KIND>', a)
    geometry = 'template<int NX,int NY,int NZ,int HEAD>\n__device__ __forceinline__ double bound_geometry_point_head(const GeometrySourceTable& sources,int x,int y,int z) {\n double total=0.;\n'
    for head in range(12):
        geometry += (' if' if head == 0 else ' else if')+f' constexpr(HEAD=={head}) {{\n'
        for i in range(point['rows'][head],point['rows'][head+1]):
            c = point['channels'][i]; x,y,z = point['offsets'][i]
            geometry += f'  total=total+bound_point_weight({i})*read_float(sources.input[{c}],Shape{{NX+{int(c==0)},NY+{int(c==1)},NZ+{int(c==2)}}},x+({x}),y+({y}),z+({z}));\n'
        geometry += ' }\n'
    geometry += ' return total;\n}\n'
    geometry += 'template<int NX,int NY,int NZ>\n__device__ __forceinline__ double geometry_point(const GeometrySourceTable& sources,int head,int x,int y,int z,\n const int* rows,const int* channels,const int* offsets,const double* weights) {\n switch(head) {\n'
    geometry += ''.join(f' case {h}:return bound_geometry_point_head<NX,NY,NZ,{h}>(sources,x,y,z);\n' for h in range(12))
    geometry += ' default:return __longlong_as_double(0x7ff8000000000000ULL);\n }\n}\n'
    source = source[:a]+geometry+source[b:]
    # The two original sampler bodies keep every ordered dot operation.
    if source.count('gate[kind*3+a]') != 2 or source.count('mix[(kind*3+a)*3+c]') != 2:
        raise ValueError('Exact two sampler templates required')
    source = source.replace('gate[kind*3+a]', 'bound_gate(kind*3+a)').replace('mix[(kind*3+a)*3+c]', 'bound_mix((kind*3+a)*3+c)')
    metadata = point['rows']+point['channels']+[v for o in point['offsets'] for v in o]
    source += '\nextern "C" __global__ void bound_parameter_bits(unsigned long long* bits,int* metadata) {\n int i=int(blockIdx.x*blockDim.x+threadIdx.x);\n'
    source += f' if(i<{len(coefficients["values_hex"])} )bits[i]=__double_as_longlong(bound_point_weight(i));\n'
    point_count=len(coefficients['values_hex']); gate_count=len(t['mid_gate']['values_hex']); mix_count=len(t['mix']['values_hex'])
    source += f' if(i<{gate_count})bits[{point_count}+i]=__double_as_longlong(bound_gate(i));\n if(i<{mix_count})bits[{point_count+gate_count}+i]=__double_as_longlong(bound_mix(i));\n switch(i) {{\n'
    source += ''.join(f' case {i}:metadata[i]={v};break;\n' for i,v in enumerate(metadata))+' }\n}\n'
    source += r'''
extern "C" __global__ void bound_point_primitive(long long count,const float* vx,const float* vy,const float* vz,
 int nx,int ny,int nz,const int* queries,const int* rows,const int* channels,const int* offsets,const double* weights,
 double* output,double* reference) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const float* fields[3]={vx,vy,vz};Shape shapes[3]={dims(0,nx,ny,nz),dims(1,nx,ny,nz),dims(2,nx,ny,nz)};
 const int* q=queries+4*i;
 output[i]=point_linear(fields,shapes,q[0],q[1],q[2],q[3],rows,channels,offsets,weights);
 reference[i]=dynamic_point_reference(fields,shapes,q[0],q[1],q[2],q[3],rows,channels,offsets,weights);
}
extern "C" __global__ void bound_mix_gate_primitive(long long count,const double* values,const int* heads,
 const double* gates,const double* mixes,double* output,double* reference) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 int head=heads[i];const double* v=values+5*i;double a=0.,b=0.;
 for(int c=0;c<3;++c){a=a+bound_mix(head*3+c)*v[c];b=b+mixes[head*3+c]*v[c];}
 output[2*i]=a;reference[2*i]=b;
 output[2*i+1]=(v[3]*bound_gate(head))*v[4];reference[2*i+1]=(v[3]*gates[head])*v[4];
}
'''
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Codegen dependency changed during generation: '+p)
    source='// Generated frozen learned-checkpoint specialization. Inherited dynamic-parameter comments apply to reference kernels.\n'+source
    output.write_text(source)
    receipt.write_text(json.dumps(dict(status='generated_exact_managed_deployed_sampler_source', manifest=reg['manifest'],
        manifest_sha256=digest(ROOT/reg['manifest']), template_source=reg['template_source'], template_sha256=digest(ROOT/reg['template_source']),
        generated_source=reg['output_source'], generated_source_sha256=digest(output), parameter_sha256={n:r['sha256'] for n,r in zip(('point','mid_gate','mix'),records)},
        ordered_arithmetic='No manual deletion of products or sums; all33pointterms,12gatesand36mixentriesserializedwithIEEEbits. SameF64operations/order, clamping/interpolation/pressure.',
        cpu_model_forward=False, model_forward_executed=False, optimizer_updates=0, native_fields_or_RHS_used=False,
        expanded_reserved52_references_opened=0, protocol_sha256=digest(args.protocol)), indent=2)+'\n')
    print('Generatedsourceboundlearnedsampler', output, flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);run(p.parse_args())
