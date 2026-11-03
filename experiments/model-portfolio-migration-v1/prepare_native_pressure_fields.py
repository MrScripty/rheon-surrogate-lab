"""Wrap exact retained pressure payloads for Tuldok's native field inspector.

Uses only the standard library. No model, tensor runtime or reference generator
is imported. Only NPY headers change from batched storage to a single 3D field.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs/plans/model-portfolio-migration-v1'
OUTPUT = ROOT / 'runs/model-portfolio-migration-v1/native-pressure-fields-01'
INPUTS = ROOT / 'runs/water-early-pressure-v1/control/project/inputs'
CASE = ('baseline--water-validation-0060', 63)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def npy(raw):
    if raw[:8] != b'\x93NUMPY\x01\x00':
        raise ValueError('Expected original NPY1.0 payload')
    end = 10 + struct.unpack('<H', raw[8:10])[0]
    header = ast.literal_eval(raw[10:end].decode('ascii'))
    if (set(header) != {'descr', 'fortran_order', 'shape'} or header['descr'] != '<f8'
            or header['fortran_order'] is not False
            or len(raw[end:]) != math.prod(header['shape']) * 8):
        raise ValueError('Original field storage differs')
    return header['shape'], raw[end:]


def run():
    if OUTPUT.exists():
        raise ValueError('Fresh retained field-bundle output required')
    score = json.loads((DOCS / 'reports/pressure-common-score-01.json').read_text())
    payloads, receipts, chosen = {}, {}, None
    for identifier in ('C107', 'C128'):
        selected = score['models'][identifier]['result']
        path = ROOT / selected['path']
        assert sha(path.read_bytes()) == selected['sha256']
        result = json.loads(path.read_text())
        index, row = next((i, r) for i, r in enumerate(result['rows'])
                          if (r['trajectory_id'], r['native_frame']) == CASE)
        identity = {k: row[k] for k in ('trajectory_id', 'native_frame', 'parent', 'counts_xyz',
                                       'cell_size_m', 'reference_pressure_sha256', 'input_sha256')}
        if chosen is not None:
            assert chosen == identity
        chosen = identity
        archive = path.parent / result['output_archive']['path']
        assert sha(archive.read_bytes()) == result['output_archive']['sha256']
        with zipfile.ZipFile(archive) as source:
            shape, raw = npy(source.read(f'case{index:03}_pressure.npy'))
        assert shape == (1, 1, *row['counts_xyz'][::-1]) and sha(raw) == row['output_sha256']
        payloads['pressure_' + identifier] = raw
        receipts[identifier] = dict(result=selected, archive=result['output_archive'],
                                    original_row=row, weights_sha256=result['weights_sha256'])
    manifest_path = INPUTS / 'validation_splits.json'
    manifest = json.loads(manifest_path.read_text())
    parent = manifest['parents'][chosen['parent']]
    key = f"p{chosen['parent']:04}_reference"
    slot = next(i for i, frame in enumerate(parent['frames']) if frame['index'] == CASE[1])
    packet_names = ('validation_dataset.npz', 'validation_dataset_extra.npz')
    for name, shard in zip(packet_names, manifest['shards'], strict=True):
        if key not in shard['keys']:
            continue
        packet = INPUTS / name
        assert sha(packet.read_bytes()) == shard['sha256']
        with zipfile.ZipFile(packet) as source:
            shape, raw = npy(source.read(key + '.npy'))
        spec = manifest['arrays'][key]
        assert list(shape) == spec['shape'] and sha(raw) == spec['sha256']
        size = math.prod(chosen['counts_xyz']) * 8
        payloads['pressure_reference'] = raw[slot * size:(slot + 1) * size]
        break
    assert sha(payloads['pressure_reference']) == chosen['reference_pressure_sha256']
    source = ROOT / 'runs/model-portfolio-migration-v1/sources/Tuldok'
    sys.path.insert(0, str(source / 'tests'))
    from qualify_scientific_fields import append_frame, append_source, base_manifest, bundle
    grid = dict(shape_xyz=chosen['counts_xyz'], origin_xyz=[0, 0, 0],
                spacing_xyz=[chosen['cell_size_m']] * 3,
                orientation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                units='m in local grid coordinates; world origin unavailable')
    fields = [dict(id=name, label=name.replace('_', ' '), dtype='<f8', centering='cell',
                   units='Pa', stage='Retained native pressure request; no gauge or value conversion')
              for name in ('pressure_reference', 'pressure_C107', 'pressure_C128')]
    data = base_manifest(grid, fields,
        dict(namespace='rheon-frozen-old-consumed-pressure', id=CASE[0] + '-frame63'),
        'Lossless packaging of previously generated GPU predictions and old-reference pressure',
        sha((DOCS / 'reports/pressure-common-score-01.json').read_bytes()))
    data['time_units'] = 'native frame index; physical timestamp not reconstructed'
    members = {}
    append_frame(data, members, '63', {name: (payloads[name], chosen['counts_xyz'][::-1])
                                     for name in payloads})
    provenance = dict(case=chosen, original_parent=parent, model_receipts=receipts,
        classification='Consumed-case diagnostic, not independent validation or scientific acceptance',
        coordinates='Local tensor xyz basis; zero local corner is a display origin, not a recovered world origin.',
        clock='63 is the retained native frame index. No physical timestamp is inferred from it.')
    append_source(data, members, 'sources/retained-case.json',
                  (json.dumps(provenance, indent=2) + '\n').encode(),
                  'Exact source and metric identities, grid density and explicit display-clock/origin limits')
    raw = bundle(data, members)
    OUTPUT.mkdir()
    (OUTPUT / 'paired-pressure-fields.zip').write_bytes(raw)
    bundle_path = (OUTPUT / 'paired-pressure-fields.zip').relative_to(ROOT).as_posix()
    declaration = dict(schema_version=1, id=sha(bundle_path.encode() + bytes.fromhex(sha(raw)))[:32],
        name='C107 C128 and reference: retained common pressure case',
        data=[dict(id='paired-pressure-fields', label='Exact reference C107 C128 pressure fields',
                   path=bundle_path, sha256=sha(raw))], recipes=[], evidence=[], requirements=[])
    (OUTPUT / 'source-project.json').write_text(json.dumps(declaration, indent=2) + '\n')
    report = dict(format='rheon_lossless_native_pressure_fields_v1', case=chosen,
        model_receipts=receipts, bundle_sha256=sha(raw), bundle_bytes=len(raw),
        payload_sha256={name: sha(value) for name, value in payloads.items()},
        tensor_payloads_unchanged=True, reference_manifest_sha256=sha(manifest_path.read_bytes()),
        models_executed=False, physical_clock_reconstructed=False,
        world_origin_reconstructed=False, new_Rheon_adopted=False)
    (OUTPUT / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(OUTPUT.relative_to(ROOT)), sha256=sha(raw), bytes=len(raw))))


if __name__ == '__main__':
    run()
