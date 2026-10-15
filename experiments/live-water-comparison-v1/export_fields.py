"""Export a retained native/AI comparison through Tuldok's existing field format."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import uuid
import zipfile

LAB = Path(__file__).resolve().parents[2]
FIELDS = (
    ('pressure', 'Rheon reference pressure', '<f8', 'Pa'),
    ('ai_pressure', 'AI diagnostic pressure', '<f8', 'Pa'),
    ('pressure_difference', 'AI minus Rheon pressure', '<f8', 'Pa'),
    ('tracer', 'Reference passive tracer', '<f4', 'dimensionless'),
    ('ai_tracer', 'AI passive tracer', '<f4', 'dimensionless'),
)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, maximum=8 * 1024**2):
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if not 0 < len(raw) <= maximum:
        raise ValueError('Comparison source exceeds its bound: ' + path.name)
    return raw


def export(run, tuldok, output):
    import numpy as np
    sys.path.insert(0, str(tuldok))
    from adapters.rheon.generate import npy
    import scientific_fields

    if output.exists():
        raise ValueError('Choose a fresh export directory')
    output.relative_to(LAB)
    sources = {name: read(run / name) for name in ('scene.json', 'model.json', 'completion.json')}
    scene = json.loads(sources['scene.json'])['scene']
    model = json.loads(sources['model.json'])
    files = sorted(run.glob('frame-*.json'))
    if not 1 <= len(files) <= 256:
        raise ValueError('Tuldok supports 1–256 complete frames; no implicit thinning')
    counts = scene['counts']
    h = 1 / scene['cells_per_metre']
    manifest = {
        'schema_version': 1, 'format': 'scientific_field_sequence_v1',
        'grid': {'shape_xyz': counts, 'origin_xyz': [0, 0, 0], 'spacing_xyz': [h] * 3,
                 'orientation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'units': 'm'},
        'time_units': 's',
        'fields': [{'id': key, 'label': label, 'dtype': dtype, 'centering': 'cell', 'units': units,
                    'stage': 'Diagnostic comparison: shared native prefix, then independent histories; see verdicts'}
                   for key, label, dtype, units in FIELDS],
        'frames': [], 'family': {'namespace': 'live-water-pressure-comparison-v1', 'id': run.name},
        'sources': [], 'provenance': {'producer': 'Rheon native / frozen CPU AI comparison',
                                    'source_id': digest(sources['scene.json'] + sources['model.json'])},
    }
    members = {}
    verdicts = []
    total = 0
    for index, path in enumerate(files):
        raw = read(path)
        total += len(raw)
        if total > 128 * 1024**2:
            raise ValueError('Retained JSON exceeds export read budget')
        value = json.loads(raw)
        frame = value['frame']
        if (value['schema'] != 'rheon-live-water-comparison-frame-v2'
                or value['counts_xyz'] != counts or value['cell_size_m'] != h
                or value['checkpoint_sha256'] != model['checkpoint_sha256']
                or frame['step'] != index):
            raise ValueError('Mixed geometry, model or incomplete timeline')
        reference = np.asarray(frame['reference_pressure'], dtype='<f8')
        ai = np.asarray(frame['ai_pressure'], dtype='<f8')
        values = (reference, ai, ai - reference, frame['reference_tracer'], frame['ai_tracer'])
        arrays = []
        for (key, _label, dtype, _units), values_ in zip(FIELDS, values):
            array = np.asarray(values_, dtype=dtype)
            if array.shape != (math.prod(counts),) or not np.isfinite(array).all():
                raise ValueError('Invalid full-grid field')
            member = f'arrays/{index:04d}-{key}.npy'
            encoded = npy(array.tobytes(), dtype, counts[::-1])
            members[member] = encoded
            arrays.append({'field': key, 'path': member, 'bytes': len(encoded), 'sha256': digest(encoded)})
        manifest['frames'].append({'time': str(frame['time_s']), 'arrays': arrays})
        verdicts.append({'step': index, 'phase': frame['phase'], 'source_frame_sha256': digest(raw),
                         'reference_quality': frame['reference_quality'], 'ai_quality': frame['ai_quality']})
    sources['verdicts.json'] = canonical({'schema': 'rheon-paired-pressure-verdicts-v1', 'frames': verdicts})
    for name, raw in sources.items():
        members[name] = raw
        manifest['sources'].append({'path': name, 'bytes': len(raw), 'sha256': digest(raw),
                                    'description': 'Original native comparison provenance and per-step verdicts'})
    members['manifest.json'] = canonical(manifest)
    if sum(map(len, members.values())) > scientific_fields.BUNDLE_LIMIT - 1024**2:
        raise ValueError('Field bundle exceeds Tuldok budget; choose a smaller scene/run')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, raw in sorted(members.items()):
            archive.writestr(name, raw)
    raw = stream.getvalue()
    scientific_fields.parse_bundle(raw)
    output.mkdir(parents=True)
    (output / 'comparison.zip').write_bytes(raw)
    project = {
        'schema_version': 1, 'id': uuid.uuid4().hex,
        'name': 'Water pressure comparison · native warm-up · AI diagnostic',
        'data': [{'id': 'comparison', 'label': 'Reference, raw AI and signed error · all retained frames',
                  'path': (output / 'comparison.zip').relative_to(LAB).as_posix(), 'sha256': digest(raw)}],
        'evidence': [], 'requirements': [], 'recipes': [],
    }
    (output / 'project.json').write_bytes(canonical(project))
    return {'project': str(output / 'project.json'), 'sha256': digest(raw), 'bytes': len(raw),
            'frames': len(files), 'fields': len(FIELDS), 'checkpoint_sha256': model['checkpoint_sha256']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'tuldok', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run.absolute(), args.tuldok.absolute(), args.output.absolute())))
