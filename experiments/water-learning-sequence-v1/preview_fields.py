"""Export a native preview timeline to Tuldok's existing scientific field format."""
import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

from policy import LAB, MIB, canonical, decode, fresh, read, require, sha, write


def export(session_path, tuldok, output):
    raw = read(session_path, 32 * MIB)
    session = decode(raw)
    require(session['schema'] == 'rheon-water-training-timeline-v1',
            'preview_fields', 'qualified native timeline')
    receipt = decode(read(session_path.parent / 'receipt.json', MIB))
    require(receipt['status'] == 'verified_native_replay' and receipt['session_sha256'] == sha(raw),
            'preview_custody', 'exact verified replay bytes')
    sys.path.insert(0, str(tuldok))
    from adapters.rheon.generate import npy
    import scientific_fields

    counts = session['counts_xyz']
    h = session['cell_size_m']
    fields = [('pressure', '<f8', 'Pa'), ('tracer', '<f4', 'dimensionless'),
              ('velocity_x', '<f8', 'm/s'), ('velocity_y', '<f8', 'm/s'),
              ('velocity_z', '<f8', 'm/s'), ('speed', '<f8', 'm/s')]
    manifest = {
        'schema_version': 1, 'format': 'scientific_field_sequence_v1',
        'grid': {'shape_xyz': counts, 'origin_xyz': [0, 0, 0], 'spacing_xyz': [h]*3,
                 'orientation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'units': 'm'},
        'time_units': 's',
        'fields': [{'id': name, 'label': name.replace('_', ' '), 'dtype': dtype,
                    'centering': 'cell', 'units': units,
                    'stage': 'Native water preview; steps beyond 64 are observation continuation only'}
                   for name, dtype, units in fields],
        'frames': [], 'family': {'namespace': 'sealed-water-forcing-v1',
                                'id': session['trajectory_id']},
        'sources': [], 'provenance': {'producer': 'Measured Rheon native training-scene replay',
                                    'source_id': sha(raw)},
    }
    members = {}
    for frame in session['frames']:
        velocity = np.asarray(frame['velocity'], dtype='<f8')
        values = [frame['pressure'], frame['tracer'], *velocity.T,
                  np.linalg.norm(velocity, axis=1)]
        row = {'time': str(frame['time_s']), 'arrays': []}
        for (name, dtype, _), value in zip(fields, values):
            array = np.asarray(value, dtype=dtype)
            require(array.shape == (int(np.prod(counts)),) and np.isfinite(array).all(),
                    'preview_values', 'finite whole-grid fields')
            path = f'arrays/{frame["index"]:04d}-{name}.npy'
            encoded = npy(array.tobytes(), dtype, counts[::-1])
            members[path] = encoded
            row['arrays'].append({'field': name, 'path': path, 'bytes': len(encoded),
                                  'sha256': sha(encoded)})
        manifest['frames'].append(row)
    source = {key: value for key, value in session.items() if key != 'frames'}
    source['session_sha256'] = sha(raw)
    source['recorded_native_steps'] = [frame['index'] + 1 for frame in session['frames']]
    source['interpretation'] = 'Frames through step 64 are the original simulation prefix; subsequent native steps are not new training data.'
    encoded = canonical(source)
    members['scene-provenance.json'] = encoded
    manifest['sources'] = [{'path': 'scene-provenance.json', 'bytes': len(encoded),
                            'sha256': sha(encoded),
                            'description': 'Exact force, water material, physical grid, native steps and replay lineage'}]
    members['manifest.json'] = canonical(manifest)
    require(sum(map(len, members.values())) < scientific_fields.BUNDLE_LIMIT - MIB,
            'preview_bundle', 'existing Tuldok bundle budget')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, value in sorted(members.items()):
            archive.writestr(name, value)
    bundle = stream.getvalue()
    scientific_fields.parse_bundle(bundle)
    output = fresh(output)
    (output / 'fields.zip').write_bytes(bundle)
    result = {'status': 'verified_existing_tuldok_format', 'path': (output / 'fields.zip').relative_to(LAB).as_posix(),
              'sha256': sha(bundle), 'bytes': len(bundle), 'frames': len(session['frames']),
              'native_steps': session['provenance']['native_steps_qualified'],
              'trajectory_id': session['trajectory_id'], 'session_sha256': sha(raw)}
    write(output / 'receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'tuldok', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.session.absolute(), args.tuldok.absolute(), args.output.absolute())))
