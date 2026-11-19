"""Batched differentiable port of the frozen learned a-tail8 pressure network.

Input q has pressure units: q=(rho/h)*native_RHS on an isotropic sealed grid.
There is no analytical inverse or numerical pressure solve here. The cosine
gains and spatial weights come only from the frozen parent neural network.
This port is a component, not a causal full-state temporal predictor.
"""
from collections import OrderedDict
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class NonoverlappingAverage(torch.autograd.Function):
    """Parent average forward and deterministic disjoint backward."""
    @staticmethod
    def forward(ctx, value, kernel):
        ctx.spatial=value.shape[2:];ctx.kernel=kernel
        return F.avg_pool3d(value,kernel,stride=kernel,ceil_mode=True,count_include_pad=False)

    @staticmethod
    def backward(ctx, gradient):
        for axis,(size,kernel) in enumerate(zip(ctx.spatial,ctx.kernel),2):
            count=torch.full((gradient.shape[axis],),kernel,device=gradient.device,dtype=gradient.dtype)
            count[-1]=size-(gradient.shape[axis]-1)*kernel
            shape=[1]*5;shape[axis]=-1
            gradient=(gradient/count.reshape(shape)).repeat_interleave(kernel,dim=axis).narrow(axis,0,size)
        return gradient,None


class NearestResize(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,size):
        ctx.spatial=value.shape[2:];ctx.output=size
        return F.interpolate(value,size=size,mode='nearest')

    @staticmethod
    def backward(ctx,gradient):
        for axis,(small,large) in enumerate(zip(ctx.spatial,ctx.output),2):
            bounds=[(i*large+small-1)//small for i in range(small+1)]
            gradient=torch.stack([gradient.narrow(axis,bounds[i],bounds[i+1]-bounds[i]).sum(dim=axis)
                                  for i in range(small)],dim=axis)
        return gradient,None


class ReplicatedConvolution(nn.Conv3d):
    def _conv_forward(self,value,weight,bias):
        if value.device.type=='cpu':return super()._conv_forward(value,weight,bias)
        for axis,padding in enumerate(self.padding,2):
            if padding:
                value=torch.cat((value.narrow(axis,0,1).expand(*value.shape[:axis],padding,*value.shape[axis+1:]),
                    value,value.narrow(axis,value.shape[axis]-1,1).expand(*value.shape[:axis],padding,*value.shape[axis+1:])),dim=axis)
        return F.conv3d(value,weight,bias,self.stride,0,self.dilation,self.groups)


class LayerContainer(nn.Module):
    def __init__(self,layers):
        super().__init__();self.layers=layers

    def forward(self,value):return self.layers(value)


def layer(spec):
    kind=spec['type']
    if kind=='Conv3d':
        arguments={key:value for key,value in spec.items()if key!='type'}
        convolution=ReplicatedConvolution if spec['padding_mode']=='replicate'else nn.Conv3d
        return convolution(**arguments)
    if kind=='Identity':return nn.Identity()
    if kind=='AdaptiveAvgPool3d':
        if spec['output_size']not in(1,[1,1,1]):raise ValueError('Only qualified global pooling supported')
        return nn.AdaptiveAvgPool3d(1)
    if kind=='Sequential':return nn.Sequential(OrderedDict((name,layer(child))for name,child in spec['children'].items()))
    if kind=='ModuleList':
        if list(spec['children'])!=[str(i)for i in range(len(spec['children']))]:raise ValueError('Ordered list required')
        return nn.ModuleList([layer(child)for child in spec['children'].values()])
    if kind=='LayerContainer':return LayerContainer(layer(spec['layers']))
    raise ValueError('Unsupported frozen spatial layer '+kind)


class FrozenSpatial(nn.Module):
    def __init__(self,architecture):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            for name,spec in architecture.items():setattr(self,name,layer(spec))

    def forward(self,value):
        skips=[]
        for encoder in self.encoders:
            value=encoder(value);skips.append(value)
            kernel=tuple(min(2,n)for n in value.shape[2:])
            value=NonoverlappingAverage.apply(value,kernel) if value.device.type=='cuda' else F.avg_pool3d(
                value,kernel,stride=kernel,ceil_mode=True,count_include_pad=False)
        value=self.bottleneck(value)
        # AdaptiveAvgPool3d(1) is the full spatial mean. This spelling also
        # permits deterministic CUDA gradients through the frozen component.
        value=value+value.mean(dim=(2,3,4),keepdim=True)
        for decoder,skip in zip(self.decoders,reversed(skips)):
            value=NearestResize.apply(value,skip.shape[2:]) if value.device.type=='cuda' else F.interpolate(
                value,size=skip.shape[2:],mode='nearest')
            value=decoder(torch.cat((value,skip),dim=1))
        return self.head(value)


def index_laplacian(value):
    """Fixed forward stencil already used inside the parent learned network."""
    result=torch.zeros_like(value)
    for physical_axis,tensor_axis in enumerate((4,3,2)):
        difference=value.narrow(tensor_axis,1,value.shape[tensor_axis]-1)-value.narrow(
            tensor_axis,0,value.shape[tensor_axis]-1)
        lower=[0]*6;upper=[0]*6
        lower[2*physical_axis]=1;upper[2*physical_axis+1]=1
        result=result+(F.pad(difference,lower)-F.pad(difference,upper))
    return result


def bases(counts,modes):
    result=[]
    for n in counts:
        mode=torch.arange(min(modes,n),dtype=torch.float64)
        cells=torch.arange(n,dtype=torch.float64)
        axis=torch.cos(math.pi/n*mode[:,None]*(cells[None]+.5))*math.sqrt(2/n)
        axis[0]/=math.sqrt(2);result.append(axis)
    return result


def restrict(value,axes):
    z,y,x=axes
    value=torch.einsum('mx,bczyx->bczym',x,value)
    value=torch.einsum('ly,bczym->bczlm',y,value)
    return torch.einsum('kz,bczlm->bcklm',z,value)


def prolong(value,axes):
    z,y,x=axes
    value=torch.einsum('mx,bcklm->bcklx',x,value)
    value=torch.einsum('ly,bcklx->bckyx',y,value)
    return torch.einsum('kz,bckyx->bczyx',z,value)


class FrozenPressureCore(nn.Module):
    def __init__(self,asset_directory):
        super().__init__();directory=Path(asset_directory)
        receipt=json.loads((directory/'receipt.json').read_text())
        if receipt['status']!='exported_for_qualification':raise ValueError('Frozen export receipt required')
        for name,spec in receipt['files'].items():
            raw=(directory/name).read_bytes()
            if len(raw)!=spec['bytes']or hashlib.sha256(raw).hexdigest()!=spec['sha256']:
                raise ValueError('Frozen pressure asset changed: '+name)
        config=json.loads((directory/'configuration.json').read_text())
        if config['schema']!='rheon-frozen-learned-pressure-core-assets-v1' or not config['gains_are_cached_learned_head_outputs']:
            raise ValueError('Qualified learned-asset schema required')
        self.fine=FrozenSpatial(config['architecture'])
        with np.load(directory/'fine-weights.npz',allow_pickle=False)as arrays:
            self.fine.load_state_dict({name:torch.from_numpy(arrays[name].copy())for name in arrays.files},strict=True)
        if sum(p.numel()for p in self.fine.parameters())!=config['spatial_parameters']:
            raise ValueError('Frozen spatial inventory changed')
        self.requires_grad_(False)
        self.steps=config['steps'];self.rhs_scale=config['laplacian_rhs_scale'];self.output_scale=config['fine_output_scale']
        self.geometry={}
        with np.load(directory/'geometry-gains.npz',allow_pickle=False)as arrays:
            for index,row in enumerate(config['geometries']):
                counts=tuple(row['counts_zyx']);prefix=f'g{index}'
                for name,modes in (('low',4),('tail',8)):
                    key=row[name+'_gain_key']
                    if key is None:continue
                    self.register_buffer(prefix+'_'+name,torch.from_numpy(arrays[key].copy()),persistent=False)
                    for axis,value in enumerate(bases(counts,modes)):
                        self.register_buffer(f'{prefix}_{name}_axis{axis}',value,persistent=False)
                self.geometry[counts]=(prefix,row['tail_gain_key']is not None)
        # Match the independently confirmed parent execution route, including
        # its precise learned CNN map for <=2048 cells. Construct only from
        # frozen CNN unit responses; no physical operator or labels enter it.
        self.dense_preparation=[]
        with torch.no_grad():
            spatial=copy.deepcopy(self.fine).double().eval()
            for counts,(prefix,_) in self.geometry.items():
                cells=math.prod(counts)
                if max(counts)>16 or cells>2048:continue
                tick=time.perf_counter();matrix=torch.empty((cells,cells),dtype=torch.float64)
                for start in range(0,cells,4):
                    batch=min(4,cells-start)
                    value=torch.zeros((batch,1,*counts),dtype=torch.float64)
                    value.flatten(1)[torch.arange(batch),start+torch.arange(batch)]=1.
                    matrix[:,start:start+batch]=spatial(value).flatten(1).T
                if not bool(torch.isfinite(matrix).all()):raise ValueError('Nonfinite frozen CNN unit responses')
                self.register_buffer(prefix+'_fine_matrix',matrix,persistent=False)
                self.dense_preparation.append(dict(counts_zyx=counts,cells=cells,seconds=time.perf_counter()-tick,
                    matrix_bytes=matrix.numel()*8,construction_forward_calls=(cells+3)//4,
                    method='Float64 unit responses of frozen float32 learned coefficients; no physical inverse'))

    def learned_proposal(self,residual,prefix,name):
        axes=tuple(getattr(self,f'{prefix}_{name}_axis{i}')for i in range(3))
        gain=getattr(self,prefix+'_'+name)[:axes[0].shape[0],:axes[1].shape[0],:axes[2].shape[0]]
        proposal=prolong(restrict(residual,axes)*gain,axes)
        proposal=proposal-proposal[:,:,:1,:1,:1]
        remaining=residual-index_laplacian(proposal)
        accept=(remaining.flatten(1).square().sum(1)<=residual.flatten(1).square().sum(1)).reshape(-1,1,1,1,1)
        return torch.where(accept,proposal,torch.zeros_like(proposal)),torch.where(accept,remaining,residual)

    def forward(self,q):
        if q.ndim!=5 or q.shape[1]!=1 or not 1<=q.shape[0]<=4 or q.dtype not in(torch.float32,torch.float64):
            raise ValueError('One to four pressure-unit requests B,1,Z,Y,X required')
        counts=tuple(q.shape[2:])
        if counts not in self.geometry:raise ValueError('Pressure geometry not exported/qualified')
        prefix,tail=self.geometry[counts]
        if q.device!=getattr(self,prefix+'_low').device or not bool(torch.isfinite(q).all()):
            raise ValueError('Finite request on the component device required')
        q=q.double();pressure=torch.zeros_like(q)
        for stage in range(self.steps):
            residual=q if stage==0 else q-index_laplacian(pressure)
            proposal,remaining=self.learned_proposal(residual,prefix,'low')
            values=(remaining/self.rhs_scale).float()
            matrix=getattr(self,prefix+'_fine_matrix',None)
            raw=self.fine(values)if matrix is None else F.linear(values.flatten(1).double(),matrix).float().reshape_as(values)
            correction=raw.double()*self.output_scale
            correction=correction-correction[:,:,:1,:1,:1]
            nonzero=(remaining.flatten(1)!=0).any(dim=1).reshape(-1,1,1,1,1)
            pressure=pressure+proposal+torch.where(nonzero,correction,torch.zeros_like(correction))
        if tail:
            proposal,_=self.learned_proposal(q-index_laplacian(pressure),prefix,'tail')
            pressure=pressure+proposal
        return pressure
