"""Bounded learned-operator precision controls; no numerical pressure inverse."""
import torch
from torch import nn
from refined_pressure_sampler import load_model as restored_model
from recurrent_sampler_assets import csr
from modal_basis import bases


class TensorDCTPressure(nn.Module):
    def __init__(self,source):
        super().__init__();self.model=source.model;self.identity=source.identity
        self.axes={};self.gains={}

    def forward(self,q):
        if q.dtype!=torch.float64 or q.device.type!='cpu'or q.ndim!=5 or q.shape[:2]!=(1,1)or not bool(torch.isfinite(q).all()):
            raise ValueError('Finite CPU double complete learned pressure input required')
        shape=tuple(q.shape[2:]);key='g'+'_'.join(map(str,shape))
        if shape not in self.axes:self.axes[shape]=bases(shape)
        if key not in self.gains:self.gains[key]=self.model.gains(key)
        z,y,x=self.axes[shape]
        coefficients=torch.einsum('kz,bzyx->bkyx',z,q[:,0])
        coefficients=torch.einsum('ly,bkyx->bklx',y,coefficients)
        coefficients=torch.einsum('mx,bklx->bklm',x,coefficients)
        coefficients=coefficients*self.gains[key]
        result=torch.einsum('mx,bklm->bklx',x,coefficients)
        result=torch.einsum('ly,bklx->bkyx',y,result)
        result=torch.einsum('kz,bkyx->bzyx',z,result)[:,None]
        return result-result[:,:,:1,:1,:1]

    def close(self):self.axes={};self.gains={}


def load_model(pair,library_path,pressure_pair,policy):
    if policy not in ('tensor_dct','quantized_locals','tensor_dct_quantized_locals'):
        raise ValueError('Three predeclared precision alternatives only')
    model=restored_model(pair,library_path,pressure_pair)
    try:
        if 'tensor_dct'in policy:
            original=model.pressure_core;original.close();model.pressure_core=TensorDCTPressure(original)
            model.asset_identity['pressure_basis_execution']='Fixed explicit orthonormal cosine/einsum transforms, multiplied by exact managed learned gains; no eigenvalues, division or analytical inverse'
        if 'quantized_locals'in policy:
            step=2.**-24;changes={}
            with torch.no_grad():
                for name in ('point','viscosity','flux','mid_gate','mix'):
                    parameter=getattr(model,name);rounded=(parameter/step).round()*step
                    changes[name]=dict(maximum_absolute_change=float((rounded-parameter).abs().max()),nonzero_before=int(parameter.count_nonzero()),nonzero_after=int(rounded.count_nonzero()))
                    parameter.copy_(rounded)
                model.coefficients={name:csr(getattr(model,name))for name in ('point','viscosity','flux')}
                model.coefficients['momentum']=csr(-model.flux.flip(dims=(2,3,4)).transpose(0,1).contiguous())
            model.asset_identity['local_precision_policy']=dict(uniform_absolute_step=step,scope='All point/viscosity/flux/query/mixing parameters; global learned pressure retained double',
                changes=changes,optimizer_updates=0,analytic_coefficients_assigned=False)
        model.frozen_versions={name:p._version for name,p in model.named_parameters()}
        return model
    except BaseException:model.close();raise
