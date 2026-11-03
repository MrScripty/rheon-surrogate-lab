"""Current velocities and prescribed forcing only; no reference or analytical inverse."""
import importlib.util
from pathlib import Path
import torch
from torch.nn import functional as F
from study_support import ROOT,read,sha
from temporal_features_v2 import force_features


class RequestPrediction:
    def __init__(self,entry_path):
        self.entry=read(entry_path)
        if self.entry['format']!='causal-pressure-request-component-v1'or not self.entry['component_only']:
            raise ValueError('Verified causal component entry required')
        for name,digest in self.entry['identities'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Restored component identity changed')
        project=ROOT/self.entry['project'];restore=ROOT/self.entry['restore'];assembly=read(project/'assembly.json')
        for row in assembly['sources']:
            if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Captured inference source changed')
        def module(name,path):
            spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(result);return result
        # Load neural model and artifact helpers only. The teacher objective is
        # intentionally never imported into the inference path.
        model=module('causal_request_snapshot_model',project/'pressure_request_model.py')
        helpers=module('causal_request_snapshot_artifacts',project/'managed_fit_artifacts.py')
        self.norm=read(restore/'normalization.json');self.config=read(project/'config.json')
        if self.config['horizons']!=1 or self.config['physical_context']['dt_s']!=1/60:
            raise ValueError('Registered single-step component contract required')
        with torch.random.fork_rng(devices=[]):
            self.network=model.CausalRequestNet(1,self.norm['scales']['velocity'],self.norm['scales']['q'])
        helpers.load_weights(self.network,restore/'weights.npz',read(restore/'history.json')['final_inventory'])
        self.network.eval();self.inventory=helpers.inventory(self.network);self.helpers=helpers

    def predict_normalized(self,fields,scene,step):
        if torch.is_grad_enabled():raise ValueError('Frozen component inference requires inference mode')
        if set(fields)!={'vx','vy','vz'}:raise ValueError('Only current staggered velocities may enter component prediction')
        nx,ny,nz=scene['counts'];shapes=((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx))
        values=[]
        for name,shape in zip(('vx','vy','vz'),shapes):
            value=torch.as_tensor(fields[name],dtype=torch.float32).reshape(1,1,*shape)/self.norm['scales']['velocity']
            if not bool(torch.isfinite(value).all()):raise ValueError('Finite current velocity required')
            values.append(F.pad(value,(0,nx+1-shape[2],0,ny+1-shape[1],0,nz+1-shape[0]),mode='replicate'))
        known=force_features(scene,step,1,self.norm['scales']['velocity'],'cpu')
        context=self.config['physical_context']
        scalars=torch.tensor([[1/scene['cells_per_metre'],context['dt_s'],context['rho_kg_m3'],context['mu_pa_s']]],dtype=torch.float32)
        predicted=self.network(torch.cat(values,dim=1),known,scalars)
        if predicted.shape!=(1,1,1,nz,ny,nx)or not bool(torch.isfinite(predicted).all()):
            raise ValueError('Finite colocated future pressure-request output required')
        return predicted

    def __call__(self,fields,scene,step):
        return self.predict_normalized(fields,scene,step).double()*self.norm['scales']['q']
