"""Sequential CUDA teacher regeneration from a closed captured source tree."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import zipfile

import numpy as np
import torch


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for b in iter(lambda:handle.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def extract(archive,destination):
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            p=Path(item.filename)
            if p.is_absolute() or '..' in p.parts or stat.S_ISLNK(item.external_attr>>16):
                raise ValueError('Safe captured member path required')
            target=destination/p
            if target.exists():raise ValueError('Duplicate captured member')
            source.extract(item,destination)


class CompactTrainingView:
    def __init__(self,manifest,archive,teacher_code,scratch_parent,source_root,allowed_inputs):
        if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('CUDA teacher only')
        self.config=json.loads(Path(manifest).read_text())
        cfg=self.config
        if cfg['development_or_reserved_used'] or cfg['active_current_states']!=261:
            raise ValueError('Closed original training source required')
        if digest(teacher_code)!=cfg['teacher_code']['sha256'] or digest(archive)!=cfg['archive']['sha256']:
            raise ValueError('Captured teacher source/input identity differs')
        self.temporary=tempfile.TemporaryDirectory(prefix='compact-teacher-',dir=scratch_parent)
        self.root=Path(self.temporary.name).resolve()
        extract(teacher_code,self.root);extract(archive,self.root)
        raw=base64.b64decode(cfg['inline_loss_assets_zip_base64'],validate=True)
        if hashlib.sha256(raw).hexdigest()!=cfg['inline_loss_assets_zip_sha256']:
            raise ValueError('Captured learned loss assets differ')
        extract(io.BytesIO(raw),self.root/'.compact-loss-assets')
        for name,spec in {**cfg['source_code'],**cfg['captured_inputs']}.items():
            path=self.root/name
            if path.stat().st_size!=spec['bytes'] or digest(path)!=spec['sha256']:
                raise ValueError('Decoded captured file changed: '+name)
        for name,spec in cfg['geometry_and_sparse_assets'].items():
            path=self.root/'.compact-loss-assets'/name
            if path.stat().st_size!=spec['bytes'] or digest(path)!=spec['sha256']:
                raise ValueError('Decoded learned response changed')
        original=Path(cfg['ambient_repository_root']).resolve()
        source_root=Path(source_root).resolve()
        # All model modules must resolve against captured teacher code. Flat
        # captured trainer helpers already imported are byte-bound recipe sources.
        sys.path[:]=[p for p in sys.path if not Path(p or os.getcwd()).resolve().is_relative_to(original) or Path(p or os.getcwd()).resolve().is_relative_to(source_root)]
        sys.path[:0]=[str(self.root/'experiments'/n) for n in ('full-ai-data-diversity-v1','full-ai-tight-targets-v1','full-ai-next-priorities-v1')]
        allowed={Path(p).resolve() for p in allowed_inputs}
        self.denied=[]
        def audit(event,args):
            if event!='open' or not isinstance(args[0],(str,bytes)):return
            path=Path(os.fsdecode(args[0])).resolve()
            if not path.is_relative_to(original) or path.is_relative_to(self.root) or path.is_relative_to(source_root) or path in allowed:return
            mode=args[1]
            if isinstance(mode,str) and any(c in mode for c in 'wax+') and path.is_relative_to(Path(scratch_parent).resolve()):return
            self.denied.append(str(path));raise PermissionError('Ambient repository read forbidden in compact trainer')
        sys.addaudithook(audit)
        from internal_group_residual_v1 import InternalGroupGenerator
        from internal_residual_dataset_v1 import TrainingView
        import support
        if support.ROOT!=self.root:raise ValueError('Teacher resolved against ambient source')
        self.manifest=json.loads((self.root/cfg['original_manifest']).read_text())
        self.fixed={r['identifier'] for r in self.manifest['scene_rows'] if r['teacher_substeps']==1}
        self.active_weight=29./148;self.sample_gradient_weight=self.active_weight/783
        self._fixed_function=TrainingView.fixed_objective
        t=cfg['teacher_configuration']
        self.teacher=InternalGroupGenerator(t['transport_pair'],t['shared_pair'],t['build'],t['control_build'])
        self.responses={};self.next_index=0;self.iterator=self._samples()

    def asset(self,member):
        with np.load(self.root/'.compact-loss-assets'/member,allow_pickle=False) as archive:
            return {n:archive[n].copy() for n in archive.files}

    def _samples(self):
        from internal_group_residual_v1 import encode
        from diagnose_group_transition_v1 import Transitions
        from recurrent_sampler_native import dimensions
        from internal_residual_loss_v1 import FrozenCoupledTraining
        from cuda_native_transfer_v1 import FIELDS
        from numerics import fine_action
        specs={(p['identifier'],p['start'],p['group_index']):p for p in self.manifest['parts']}
        sparse=self.asset(self.manifest['sparse_part']['member'])
        runner=Transitions(self.teacher,self.config['teacher_configuration'])
        for case in self.config['cases']:
            scene=json.loads((self.root/case['receipt']).read_text())['scene']
            counts=tuple(scene['counts']);shape=tuple(reversed(counts));sizes=dimensions(counts)
            h=1./scene['cells_per_metre'];rho=1000.
            for start in self.config['starts']:
                state=next(s for s in case['states'] if s['step']==start)
                with np.load(self.root/state['path'],allow_pickle=False) as a:
                    fields={n:torch.from_numpy(a[n].copy()).to('cuda') for n in FIELDS}
                with torch.no_grad():external=self.teacher.controls.generate_from_packet(counts,self.teacher.packet(scene,start,1,8).to('cuda'))
                fine=fields;offset=0
                for group,width in enumerate((2,2,3)):
                    with torch.no_grad():
                        initial=fine
                        for j in range(width):fine,_=runner.advance(fine,scene,external,offset+j,1)
                        phase=self.teacher.phase(initial,scene,external,offset,width)
                        base=self.teacher.emit(phase,float_carriers=False)
                        sample=dict(features=encode(phase),h=h,dt=phase['dt'],rho=rho)
                        for n,size in zip(FIELDS,(shape,*sizes)):
                            sample['base_'+n]=base[n].reshape(size).contiguous()
                            sample['target_'+n]=fine[n].reshape(size).contiguous()
                        p=fine['pressure'].reshape(shape)
                        sample['target_rhs']=fine_action(p.reshape(1,1,*shape),p.new_tensor([h]),p.new_tensor([rho]))[0,0]
                    row=specs[case['identifier'],start,group];member=row['geometry_member']
                    if member not in self.responses:self.responses[member]=FrozenCoupledTraining(sparse,self.asset(member))
                    yield row,sample,self.responses[member]
                    offset+=width

    def sample(self,index):
        if index!=self.next_index:raise ValueError('Regenerate sequentially once, then use the owned CUDA cache')
        row,sample,response=next(self.iterator)
        if row!=self.manifest['parts'][index]:raise ValueError('Original sample order differs')
        self.next_index+=1
        return row,sample,response

    def fixed_objective(self,normalization):return self._fixed_function(self,normalization)

    def finish_generation(self):
        if self.next_index!=783:raise ValueError('Complete training bank required')
        try:next(self.iterator)
        except StopIteration:pass
        else:raise ValueError('Unexpected extra training sample')
        self.teacher.close();self.teacher=None

    def close(self):
        if self.teacher is not None:self.teacher.close();self.teacher=None
        self.temporary.cleanup()
