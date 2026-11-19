"""A local, linear learned smoother; the numerical hierarchy owns global work."""
import json
from pathlib import Path
import sys
import torch
from torch import nn
import baseline_model as baseline
import fine_tuning
ARCHITECTURE={}
configure=baseline.configure
configure_cpu=baseline.configure_cpu
full_operator=baseline.full_operator
owner=baseline.owner
frozen=baseline.frozen
CHANNELS=['dimensionless_laplacian_rhs']

def select(config):
    baseline.select(config)
    spec=json.loads(Path(__file__).with_name('fine-tuning.json').read_text())
    fine_tuning.select(spec)
    ARCHITECTURE.clear();ARCHITECTURE.update(schema='rheon-multilevel-local-smoother-v1',input_channels=1,
        channels=['dimensionless_laplacian_rhs'],widths=([1,4,4,1] if spec['architecture']=='tiny' else [1,1]),kernels=spec['architecture'],bias=False,activation='linear',
        parameters=spec['parameters'],boundary='Replicated one-cell halo at each convolution',fixed_skip='q / 8',
        preprocessing='q=(rho/h)*residual, divided by fixed original train-only q RMS',
        reconstruction='Raw correction times the same q RMS; cell-zero gauge; add supplied base',
        coarse_training_cycles=spec['coarse_training_cycles'],deployment='Local correction called after an independently validated numerical multilevel cycle',
        restrictions='Uniform-density cubic sealed Neumann operator, up to 128 cells per axis for training; larger inference requires separate qualification')

class AxisConvolution(nn.Conv3d):
    def __init__(self,axis):
        kernel=[1,1,1];kernel[axis-2]=3
        super().__init__(1,1,tuple(kernel),bias=False);self.axis=axis
    def forward(self,values):
        a=self.axis
        values=torch.cat((values.narrow(a,0,1),values,values.narrow(a,values.shape[a]-1,1)),dim=a)
        return torch.nn.functional.conv3d(values,self.weight)

class PressureCorrectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        conv=baseline.owner.ReplicatedConvolution
        spec=fine_tuning.SPEC
        if spec['architecture']=='tiny':
            self.layers=nn.Sequential(*[conv(a,b,3,padding=1,padding_mode='replicate',bias=False)for a,b in [(1,4),(4,4),(4,1)]])
        elif spec['architecture']=='stencil':self.layers=nn.Sequential(conv(1,1,3,padding=1,padding_mode='replicate',bias=False))
        elif spec['architecture']=='separable':
            self.layers=nn.Sequential(*[AxisConvolution(axis) for axis in [2,3,4]])
        else:raise ValueError('Declared architecture required')
        with torch.no_grad():self.layers[-1].weight.mul_(.01)
    def forward(self,values):
        if values.ndim!=5 or values.shape[1]!=1 or values.dtype!=torch.float32 or not bool(torch.isfinite(values).all()):
            raise ValueError('Finite single-channel float32 field required')
        return values*.125+self.layers(values)

def fit_normalization(parents,previous):
    if any(p['entry']['split']!='train'for p in parents):raise ValueError('Train-only normalization required')
    return json.loads(Path(__file__).with_name('fixed-normalization.json').read_text())

def predict_pressure(network,sample,normalization):
    base=sample['base'];residual=sample['native_input_residual']
    if not bool(residual.ne(0).any()):return base.clone()
    import historical_physical
    if residual.ndim!=5 or residual.shape[:2]!=(1,1) or min(residual.shape[2:])<1 or max(residual.shape[2:])>256 or residual.dtype!=torch.float64 or not bool(torch.isfinite(residual).all()):raise ValueError('Bounded finite field required')
    values=((residual*(sample['rho']/sample['cell_size_m']).reshape(1,1,1,1,1))/historical_physical.scales(normalization)['laplacian_rhs']).to(torch.float32)
    raw=network(values)
    scale=historical_physical.scales(normalization)['laplacian_rhs']
    delta=raw.to(torch.float64)*scale
    return base+delta-delta[:,:,:1,:1,:1]

def pressure_loss(prediction,sample,normalization):return fine_tuning.supervised(sys.modules[__name__],prediction,sample)
def training_loss(network,sample,normalization,criterion=None):return fine_tuning.loss(sys.modules[__name__],network,sample,normalization,criterion)

