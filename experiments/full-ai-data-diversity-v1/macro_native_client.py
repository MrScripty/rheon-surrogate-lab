"""Owned fine-step references with complete physical states published at60Hz."""
from support import BASE
from bounded_native_client import BoundedNativeClient
import math
import numpy as np


class MacroNativeClient(BoundedNativeClient):
    def __init__(self,scene,directory,**kwargs):
        self.reference_substeps=scene.get('reference_substeps',1)
        if self.reference_substeps not in (1,2,4,8):
            raise ValueError('Qualified reference substep count required')
        self.reference_dt_s=1/(60*self.reference_substeps)
        self._macro_step=0
        self.last_native_frames=[]
        super().__init__(scene,directory,build=BASE/'reference-substeps-build-03',**kwargs)

    def inspect(self):
        packet,state=super().inspect()
        native_step=packet['step']
        if native_step%self.reference_substeps:
            raise ValueError('Only complete macro states may be published')
        macro_step=native_step//self.reference_substeps
        if packet['time_s']!=macro_step/60:
            raise ValueError('Native clock and macro physical time disagree')
        self._macro_step=macro_step
        packet.update(step=macro_step,native_step=native_step,
            reference_substeps=self.reference_substeps,reference_dt_s=self.reference_dt_s)
        return packet,state

    def restore(self,step,state):
        result=super().restore(step*self.reference_substeps,state)
        self._macro_step=step
        return dict(result,step=step)

    def advance(self,method='spectral',predict=None):
        if method not in ('spectral','native-pcg') or predict is not None:
            raise ValueError('Explicit offline numerical reference only')
        frames=[]
        for index in range(self.reference_substeps):
            packet=super().advance(method=method)
            frames.append(packet)
            fine=packet['frame']
            if not fine['advanced']:
                break
            expected=self._macro_step*self.reference_substeps+index+1
            if fine['step']!=expected:
                raise ValueError('Native fine-step sequence changed')
        advanced=len(frames)==self.reference_substeps and all(p['frame']['advanced']for p in frames)
        if advanced:self._macro_step+=1
        aggregate=dict(frames[-1]['frame'])
        aggregate.update(step=self._macro_step,advanced=advanced)
        for key in ('frame_seconds','preparation_seconds','pressure_seconds','projection_seconds'):
            aggregate[key]=sum(p['frame'][key]for p in frames)
        qualities=[p['frame']['quality']for p in frames]
        quality=dict(qualities[-1])
        for key in ('courant','actual_divergence','predicted_divergence','residual_l2'):
            quality[key]=max(q[key]for q in qualities)
        relative=[q['relative_residual']for q in qualities if q['relative_residual']is not None]
        quality['relative_residual']=max(relative)if relative else None
        quality['qualified']=all(q['qualified']for q in qualities)
        quality['hard_stop']=any(q['hard_stop']for q in qualities)
        aggregate['quality']=quality
        self.last_native_frames=frames
        self.last_frame=dict(frame=aggregate,fine_frames=frames,
            reference_substeps=self.reference_substeps,reference_dt_s=self.reference_dt_s,
            macro_dt_s=1/60,all_internal_reference_work_included=True)
        if self.last_request is not None:
            self.last_request['reference_dt_s']=self.reference_dt_s
        return self.last_frame

    def preview_force(self,fine_step,velocity):
        if not 0<=fine_step<self.scene['steps']*self.reference_substeps:
            raise ValueError('Bounded actual fine step required')
        self.write_json(dict(kind='forcing',step=fine_step))
        for value in velocity:
            self.child.stdin.write(memoryview(np.ascontiguousarray(value,dtype='<f4')).cast('B'))
        self.child.stdin.flush()
        packet=self.read_json()
        if packet!={'kind':'forcing','step':fine_step,'state_unchanged':True}:
            raise ValueError('Read-only native fine-force preview required')
        counts=self.scene['counts']
        return [self.read_array((counts[d]+1)*math.prod(counts[a]for a in range(3)if a!=d),'<f4')for d in range(3)]
