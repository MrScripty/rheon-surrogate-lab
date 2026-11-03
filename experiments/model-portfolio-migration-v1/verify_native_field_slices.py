"""Compare native project field reads with independently decoded original bytes.

One bounded read transaction per declared field/axis. No retries, mutation,
model execution, numerical conversion or live-service credentials are used.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/model-portfolio-migration-v1/sources/Tuldok'
AUTHORING = ROOT / 'runs/model-portfolio-migration-v1/native-scientific-authoring-01'
BUNDLE = ROOT / 'runs/model-portfolio-migration-v1/native-pressure-fields-01/paired-pressure-fields.zip'


def run(controller, output):
    if output.exists():
        raise ValueError('Fresh read-evidence output required')
    output.mkdir(parents=True)
    state = json.loads((controller / 'controller.json').read_text())
    assert state['status'] == 'ready'
    sys.path.insert(0, str(SOURCE))
    from agent_api import ProjectFilesOwnerClient
    client = ProjectFilesOwnerClient(state['origin'],
        (controller / 'private/owner-session.txt').read_text().strip(), state['catalog_id'], timeout=60)
    client.bootstrap()
    def read(name, operation, arguments):
        (output / (name + '.request.json')).write_text(json.dumps(dict(operation=operation, arguments=arguments), indent=2) + '\n')
        value = client.call(operation, arguments)
        (output / (name + '.response.json')).write_text(json.dumps(value, indent=2) + '\n')
        assert value['status'] == 200, value
        return value['result']
    status = read('00-status', 'project_files.status', {})
    selected = json.loads((AUTHORING / '05-import-fields.response.json').read_text())['result']['value']
    bound = dict(binding=status['binding'], checkpoint=status['project']['checkpoint'])
    assert status['project']['project_id'] == '716dfb680bbc4586a527d42be6b927ad'
    record = read('01-record', 'project_files.data_get', dict(bound, record_id=selected['id']))['value']
    assert record == selected and record['content_hash'] == hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    slices, elements = [], 0
    with zipfile.ZipFile(BUNDLE) as source:
        manifest = json.loads(source.read('manifest.json'))
        nx, ny, nz = manifest['grid']['shape_xyz']
        dimensions = dict(x=nx, y=ny, z=nz)
        for item in manifest['frames'][0]['arrays']:
            raw = source.read(item['path'])
            assert hashlib.sha256(raw).hexdigest() == item['sha256']
            assert raw[:8] == b'\x93NUMPY\x01\x00'
            end = 10 + struct.unpack('<H', raw[8:10])[0]
            header = ast.literal_eval(raw[10:end].decode('ascii'))
            assert header == dict(descr='<f8', fortran_order=False, shape=(nz, ny, nx))
            payload = raw[end:]
            field = next(f for f in manifest['fields'] if f['id'] == item['field'])
            for axis in 'xyz':
                name = f'{len(slices) + 2:02}-{field["id"]}-{axis}'
                result = read(name, 'project_files.data_scientific',
                    dict(bound, record_id=record['id'], content_hash=record['content_hash'],
                         field=field['id'], frame=0, axis=axis, index=0))['value']
                row_axis, column_axis = dict(x=('z', 'y'), y=('z', 'x'), z=('y', 'x'))[axis]
                assert (result['row_axis'], result['column_axis']) == (row_axis, column_axis)
                assert len(result['values']) == dimensions[row_axis]
                for row, values in enumerate(result['values']):
                    assert len(values) == dimensions[column_axis]
                    for column, value in enumerate(values):
                        xyz = dict(x=0, y=0, z=0)
                        xyz[row_axis], xyz[column_axis] = row, column
                        offset = ((xyz['z'] * ny + xyz['y']) * nx + xyz['x']) * 8
                        assert struct.pack('<d', value) == payload[offset:offset + 8]
                        elements += 1
                assert result['time'] == '63' and result['time_units'] == manifest['time_units']
                assert result['units'] == field['units'] and result['stage'] == field['stage']
                assert result['physical_origin'] == [h * .5 for h in manifest['grid']['spacing_xyz']]
                slices.append(dict(field=field['id'], axis=axis, values=sum(map(len, result['values'])), exact=True))
    final = read('99-final-status', 'project_files.status', {})
    assert final == status
    report = dict(format='rheon_native_exact_field_reads_v1', status='passed',
        controller_catalog_id=state['catalog_id'], project_id=status['project']['project_id'],
        checkpoint=status['project']['checkpoint'], configured_roots=status['roots'],
        record_id=record['id'], content_hash=record['content_hash'], slices=slices,
        exact_float64_values=elements, checkpoint_unchanged=True,
        models_executed=False, scientific_acceptance=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='passed', slices=len(slices), exact_float64_values=elements,
                         roots=status['roots'], output=str(output.relative_to(ROOT)))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.controller.resolve(), args.output.resolve())
