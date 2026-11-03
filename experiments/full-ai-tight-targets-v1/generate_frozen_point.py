"""Specialize execution from restored learned point coefficients; no fitting."""
import argparse
import json
from pathlib import Path
from profile_frozen_components import ROOT, read, write, digest
from recurrent_sampler_assets import source_tensors, csr
import torch


def run(args):
    if args.output.exists():
        raise ValueError('Fresh learned-coefficient specialization required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen point source changed: ' + path)
    torch.set_num_threads(1)
    tensors, identity = source_tensors()
    coefficients = csr(tensors['point'])
    rows, channels, offsets, weights = [value.tolist() for value in coefficients]
    code = ['// Generated only from the exact restored/quantized learned point tensor.',
            '// No analytical interpolation stencil or newly fitted coefficient is assigned.',
            '#pragma once',
            'template<int channel,int dx,int dy,int dz> static inline __m256d frozen_at4(',
            '    const float*const* fields,const Shape* shapes,int x,int y,int z){',
            '    const Shape s=shapes[channel];',
            '    const int yy=std::clamp(y+dy,0,s.y-1),zz=std::clamp(z+dz,0,s.z-1);',
            '    const int xx=x+dx;const float* row=fields[channel]+int64_t(s.x)*(yy+int64_t(s.y)*zz);',
            '    if(xx>=0&&xx+3<s.x)return _mm256_cvtps_pd(_mm_loadu_ps(row+xx));',
            '    const auto ix=_mm_min_epi32(_mm_max_epi32(_mm_setr_epi32(xx,xx+1,xx+2,xx+3),_mm_setzero_si128()),_mm_set1_epi32(s.x-1));',
            '    return gather4(row,ix,false);',
            '}',
            'template<int head> static inline __m256d frozen_point4(',
            '    const float*const* fields,const Shape* shapes,int x,int y,int z){',
            '    auto total=_mm256_setzero_pd();']
    for head in range(len(rows) - 1):
        code.append(f'    {"if" if head == 0 else "else if"} constexpr(head=={head})' + '{')
        for i in range(rows[head], rows[head + 1]):
            params = ','.join(map(str, [channels[i], *offsets[i]]))
            code.append(f'        total=_mm256_add_pd(total,_mm256_mul_pd(_mm256_set1_pd({weights[i].hex()}),frozen_at4<{params}>(fields,shapes,x,y,z)));')
        code.append('    }')
    code += ['    return total;', '}',
             'template<int head> static inline double frozen_point(',
             '    const float*const* fields,const Shape* shapes,int x,int y,int z){',
             '    double total=0;']
    for head in range(len(rows) - 1):
        code.append(f'    {"if" if head == 0 else "else if"} constexpr(head=={head})' + '{')
        for i in range(rows[head], rows[head + 1]):
            c = channels[i]
            dx, dy, dz = offsets[i]
            code.append(f'        total=total+{weights[i].hex()}*at(fields[{c}],shapes[{c}],x+({dx}),y+({dy}),z+({dz}));')
        code.append('    }')
    code += ['    return total;', '}',
             'static bool frozen_point_matches(const int* rows,const int* channels,const int* offsets,const double* weights){']
    for name, values in [('rows', rows), ('channels', channels),
                         ('offsets', [v for row in offsets for v in row]), ('weights', weights)]:
        formatted = [v.hex() if isinstance(v, float) else str(v) for v in values]
        code.append(f'    static const {"double" if name == "weights" else "int"} expected_{name}[]=' + '{' + ','.join(formatted) + '};')
        code.append(f'    for(int i=0;i<{len(values)};++i)if({name}[i]!=expected_{name}[i])return false;')
    code += ['    return true;', '}', '']
    args.output.mkdir(parents=True)
    header = args.output / 'frozen_point.hpp'
    header.write_text('\n'.join(code))
    write(args.output / 'receipt.json', dict(status='generated_from_exact_learned_tensor',
          optimizer_updates=0, analytic_coefficients_assigned=False,
          point_nonzero_coefficients=len(weights), learned_asset_identity=identity,
          generated_header_sha256=digest(header), generator_sha256=digest(__file__),
          protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
