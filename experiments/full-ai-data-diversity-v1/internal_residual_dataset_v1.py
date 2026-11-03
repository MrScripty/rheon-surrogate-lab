"""Portable, training-only full-grid archive and exact 148-scene weighting."""
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

from internal_residual_loss_v1 import FrozenCoupledTraining


FIELDS=('pressure','vx','vy','vz','tracer')
METRICS=('equation','pressure','velocity','tracer')


class TrainingView:
    def __init__(self,manifest,archive):
        self.manifest=json.loads(Path(manifest).read_text())
        m=self.manifest
        if (m['status']!='verified_all148_internal_residual_training_view'
                or m['development_or_reserved_used'] or m['expanded_reserved52_references_opened']
                or m['future_native_labels_used_for_features'] or not m['all148training_scenes_in_objective']):
            raise ValueError('Exact complete training-only view required')
        self.active={r['identifier'] for r in m['scene_rows'] if r['teacher_substeps']==8}
        self.fixed={r['identifier'] for r in m['scene_rows'] if r['teacher_substeps']==1}
        if len(self.active)!=29 or len(self.fixed)!=119 or len(m['scene_rows'])!=148:
            raise ValueError('All29active and119fixed scenes required')
        expected={(name,start,g) for name in self.active for start in m['starts'] for g in range(3)}
        actual=[(p['identifier'],p['start'],p['group_index']) for p in m['parts']]
        controls=[(p['identifier'],p['start']) for p in m['control_rows']]
        if (len(actual)!=783 or set(actual)!=expected or len(set(actual))!=len(actual)
                or len(controls)!=1071 or set(controls)!={(n,s) for n in self.fixed for s in m['starts']}):
            raise ValueError('Missing or duplicate complete-scene windows')
        if any(p['width']!=(2,2,3)[p['group_index']] for p in m['parts']):
            raise ValueError('Original2/2/3internal group widths required')
        self.archive=zipfile.ZipFile(archive)
        self.specs={p['member']:p for p in [m['sparse_part'],*m['geometry_parts'],*m['parts']]}
        if len(self.archive.infolist())!=len(self.specs) or set(self.archive.namelist())!=set(self.specs):
            self.close();raise ValueError('Archive inventory differs')
        self.sparse=self.arrays(m['sparse_part']['member'])
        self.geometries={};self.responses={}
        self.fixed_weight=1./148
        self.active_weight=29./148
        self.sample_gradient_weight=self.active_weight/783

    def close(self):
        if getattr(self,'archive',None) is not None:
            self.archive.close();self.archive=None

    def arrays(self,member):
        spec=self.specs[member]
        with self.archive.open(member) as handle:raw=handle.read(spec['bytes']+1)
        if len(raw)!=spec['bytes'] or hashlib.sha256(raw).hexdigest()!=spec['sha256']:
            raise ValueError('Training archive member changed: '+member)
        with np.load(io.BytesIO(raw),allow_pickle=False) as captured:
            if set(captured.files)!=set(spec['arrays']):raise ValueError('Typed array inventory differs')
            arrays={n:captured[n] for n in captured.files}
        for name,value in arrays.items():
            expected=spec['arrays'][name]
            if value.dtype.str!=expected['dtype'] or list(value.shape)!=expected['shape'] or not np.isfinite(value).all():
                raise ValueError('Typed finite array differs: '+member+':'+name)
        return arrays

    def sample(self,index):
        row=self.manifest['parts'][index]
        arrays=self.arrays(row['member'])
        if (float(arrays['h'])!=1./row['cells_per_metre'] or float(arrays['rho'])!=1000.
                or arrays['features'].shape!=(1,42,*reversed(row['counts_xyz']))):
            raise ValueError('Physical input identity differs')
        member=row['geometry_member']
        if member not in self.responses:
            geometry=self.arrays(member)
            self.responses[member]=FrozenCoupledTraining(self.sparse,geometry)
        required={'features','target_rhs',*[prefix+n for prefix in ('base_','target_') for n in FIELDS]}
        sample={n:torch.from_numpy(arrays[n]).to('cuda') for n in required}
        sample.update({n:float(arrays[n]) for n in ('h','dt','rho')})
        return row,sample,self.responses[member]

    def fixed_objective(self,normalization):
        per_scene={n:[] for n in self.fixed}
        for row in self.manifest['control_rows']:
            values={}
            count=math.prod(row['counts_xyz'])
            for name in METRICS:
                statistics=row['statistics'][name]
                floor=(max(normalization.get('optimization_equation_rms_floor',0.),
                           normalization.get('equation_rhs_floor',1e-30)/math.sqrt(count))
                       if name=='equation' else normalization['relative_denominator_rms_floor'][name])
                denominator=max(statistics['reference_rms'],floor)
                values[name]=(statistics['absolute_error_rms']/denominator)**2
            per_scene[row['identifier']].append(values)
        if any(len(rows)!=9 for rows in per_scene.values()):raise ValueError('All fixed9-window controls required')
        summed={name:sum(sum(row[name] for row in rows)/9 for rows in per_scene.values()) for name in METRICS}
        return dict(per_metric_scene_sum=summed,objective=sum(summed.values())/148,
                    scenes=119,windows=1071,parameter_derivative=0.)
