"""Use exact managed learned modal weights as a frozen causal pressure readout."""
from pathlib import Path
import torch
from torch import nn
from profile_frozen_components import ROOT, read, digest
from learned_modal_pressure import LearnedModalPressure
from train_modal_pressure import load, inventory
from fftw_pressure_transform_v2 import MeasuredDCTMultiplier


class RestoredModalCore(nn.Module):
    def __init__(self, pair, library_path, capture=False):
        super().__init__(); pair=Path(pair)
        receipt=read(pair/'receipt.json'); restored=read(pair/'restore/receipt.json')
        if receipt['status']!='completed' or not all(receipt['artifact_parity'].values()) or not restored['full_cpu_probe_bitwise_verified'] or restored['optimizer_updates']!=0:
            raise ValueError('Exact completed managed double pressure fit/restore required')
        if digest(pair/'restore/weights.npz')!=restored['outputs']['weights.npz']['sha256']:
            raise ValueError('Exact double pressure checkpoint changed')
        assembly=read(ROOT/receipt['source_project']/'assembly.json')
        if digest(ROOT/receipt['source_project']/'assembly.json')!=receipt['source_assembly_sha256']:
            raise ValueError('Captured pressure Project changed')
        manifest=read(ROOT/assembly['static_inputs']['manifest']['path'])
        self.model=LearnedModalPressure(manifest['geometries'])
        load(self.model,pair/'restore/weights.npz',restored['final_inventory'])
        self.model.eval().requires_grad_(False)
        self.library_path=library_path;self.owners={};self.gain_cache={};self.capture=capture;self.observations=[]
        self.identity=dict(pair=str(pair.relative_to(ROOT)),weights_sha256=digest(pair/'restore/weights.npz'),
                           training_run_id=receipt['training_run_id'],restore_run_id=receipt['restore_run_id'],
                           parameter_inventory=inventory(self.model),optimizer_updates=0,analytic_inverse_constructed=False)

    def reset(self):self.observations=[]

    def forward(self,q):
        if q.dtype!=torch.float64 or q.device.type!='cpu' or q.ndim!=5 or q.shape[1]!=1:
            raise ValueError('Actual causal pressure input must preserve CPU double precision')
        key='g'+'_'.join(map(str,q.shape[2:]));shape=tuple(q.shape)
        if shape not in self.owners:self.owners[shape]=MeasuredDCTMultiplier(shape,self.library_path)
        if key not in self.gain_cache:
            self.gain_cache[key]=self.model.gains(key).reshape(1,1,*q.shape[2:]).contiguous()
        result=self.owners[shape](q.contiguous(),self.gain_cache[key])
        result=result-result[:,:,:1,:1,:1]
        if self.capture:self.observations.append(dict(q=q.detach().clone(),pressure=result.detach().clone()))
        return result

    def close(self):
        for owner in self.owners.values():owner.close()
        self.owners={}
