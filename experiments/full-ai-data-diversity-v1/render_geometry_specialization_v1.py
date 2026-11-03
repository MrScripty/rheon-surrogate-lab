"""Reproduce precise specialization sources from external grid metadata only.

The reviewed generic templates contain no learned constants or physical fields.
New shapes use these same templates; no reference labels are read.
"""
import argparse
import hashlib
import json
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent


def render(counts_xyz, scalar_registry=None):
    counts = sorted({tuple(row) for row in counts_xyz})
    if not counts or any(len(row) != 3 or any(type(n) is not int or not 5 <= n <= 1024 for n in row)
                         for row in counts):
        raise ValueError('Nonempty finite integer grid metadata required')
    base = (DIRECTORY / 'learned_cuda_kernels_v10.cu').read_text()
    geometry = (DIRECTORY / 'learned_cuda_kernels_v12.cu').read_text()
    marker = 'extern "C" __global__ void geometry_transport_'
    if not geometry.startswith(base):
        raise ValueError('Reviewed generic base/template ancestry changed')
    result = base + geometry[len(base):geometry.index(marker)]
    for nx, ny, nz in counts:
        result += f'''extern "C" __global__ void geometry_transport_{nx}_{ny}_{nz}(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h) {{
 switch(blockIdx.y) {{
'''
        for kind in range(4):
            result += f' case {kind}: geometry_transport_body<{nx},{ny},{nz},{kind}>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;\n'
        result += ' }\n}\n'
    if scalar_registry is None:
        return result
    pairs = []
    for row in scalar_registry:
        shape, cpm, n = tuple(row['counts_xyz']), row['cells_per_metre'], row['internal_substeps']
        if (shape not in counts or type(cpm) is not int or not 1 <= cpm <= 1024
                or type(n) is not int or n not in (1, 8)):
            raise ValueError('Registered external scalar metadata required')
        if row['spacing_hex'] != (1 / cpm).hex() or row['timestep_hex'] != (1 / (60 * n)).hex():
            raise ValueError('Exact external scalar literal mismatch')
        pairs.append((shape, cpm, n))
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError('Distinct nonempty scalar configurations required')
    scalar_source = (DIRECTORY / 'learned_cuda_kernels_v13.cu').read_text()
    body_start = scalar_source.index('template<int NX,int NY,int NZ,int KIND,int CPM,int SUBSTEPS>')
    body_end = scalar_source.index('extern "C" __global__ void scalar_geometry_transport_', body_start)
    result += '\n// Partial evaluation of supplied physical metadata only; all learned values remain arguments.\ntemplate<int CPM,int SUBSTEPS> struct PhysicalScalarValues;\n'
    for cpm, n in sorted({(cpm, n) for _, cpm, n in pairs}):
        result += f'template<> struct PhysicalScalarValues<{cpm},{n}> {{static constexpr double h={(1/cpm).hex()};static constexpr double dt={(1/(60*n)).hex()};}};\n'
    result += scalar_source[body_start:body_end]
    for (nx, ny, nz), cpm, n in sorted(pairs):
        name = f'scalar_geometry_transport_{nx}_{ny}_{nz}_cpm{cpm}_n{n}'
        result += f'extern "C" __global__ void {name}(const __grid_constant__ GeometrySourceTable sources,\n const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix) {{\n switch(blockIdx.y) {{\n'
        for kind in range(4):
            result += f' case {kind}:scalar_geometry_transport_body<{nx},{ny},{nz},{kind},{cpm},{n}>(sources,rows,channels,offsets,weights,gate,mix);break;\n'
        result += ' }\n}\n'
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry-protocol', type=Path, required=True)
    p.add_argument('--scalar-registry', type=Path)
    destination = p.add_mutually_exclusive_group(required=True)
    destination.add_argument('--output', type=Path)
    destination.add_argument('--verify', type=Path)
    args = p.parse_args()
    counts = json.loads(args.geometry_protocol.read_text())['geometry_counts_xyz']
    registry = json.loads(args.scalar_registry.read_text())['geometry_scalar_registry'] if args.scalar_registry else None
    encoded = render(counts, registry).encode()
    if args.verify:
        if args.verify.read_bytes() != encoded:
            raise ValueError('Generated source differs from the frozen CUDA source')
        print('Exact external-metadata source reproduction:', args.verify, hashlib.sha256(encoded).hexdigest())
    else:
        if args.output.exists():
            raise ValueError('Fresh source path required')
        args.output.write_bytes(encoded)
