"""Decode original native F64 pressure/RHS fields without executing a model."""
import hashlib
from pathlib import Path

import numpy as np

from import_common_generator_measurements import ROOT, read, require
from component_material import observed_file


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_fields(contract, protocol):
    receipts, fields = {}, []
    for row in contract['rows']:
        path = ROOT/row['source']
        receipt_path = path.parent/'receipt.json'
        name = str(receipt_path.relative_to(ROOT))
        if name not in receipts:
            observed_file(ROOT,name,dict(sha256=protocol['source_sha256'][name]))
            receipts[name] = read(receipt_path)
        receipt = receipts[name]
        scene = receipt['scene']
        frame = receipt['states'][row['state']]
        require(scene['counts'] == row['counts_zyx'][::-1]
                and scene['cells_per_metre'] == row['cells_per_metre']
                and frame['sha256'] == row['source_sha256'] and frame['step'] == row['state']
                and path == receipt_path.parent/frame['path'], 'Original scene/frame metadata differs')
        observed_file(ROOT,row['source'],dict(sha256=row['source_sha256'],bytes=frame['bytes']))
        values = {}
        with np.load(path,allow_pickle=False) as archive:
            for field in ('pressure','rhs'):
                value = archive[field]
                spec = frame['arrays'][field]
                require(value.dtype.str == spec['dtype'] == '<f8' and list(value.shape) == spec['shape']
                        and value.size == int(np.prod(row['counts_zyx'])) and np.isfinite(value).all()
                        and digest(value) == spec['sha256'], 'Original native F64 field identity differs')
                values[field] = value.reshape(row['counts_zyx']).copy()
        require(values['pressure'][0,0,0] == 0., 'Original native cell-zero reference gauge required')
        fields.append(dict(identity=row,rhs=values['rhs'],reference=values['pressure'],
            h=1./row['cells_per_metre'],rho=1000.,
            rhs_sha256=digest(values['rhs']),reference_sha256=digest(values['pressure'])))
    require(len(fields) == 144, 'Complete matched native field population required')
    return fields
