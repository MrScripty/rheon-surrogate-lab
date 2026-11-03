"""Adaptive mean pooling as separable fixed linear maps with deterministic backward."""
from functools import lru_cache
import torch
from torch.nn import functional as F

@lru_cache(maxsize=128)
def weights(length,outputs,device,dtype,kind='adaptive'):
    with torch.inference_mode(False):
        result=torch.zeros((outputs,length),device=device,dtype=dtype)
        for index in range(outputs):
            start=index*length//outputs if kind=='adaptive'else 2*index
            end=((index+1)*length+outputs-1)//outputs if kind=='adaptive'else min(length,2*index+2)
            result[index,start:end]=1/(end-start)
    return result

def adaptive_pool3d(value,output_size):
    z,y,x=value.shape[-3:]
    a,b,c=output_size;device=str(value.device)
    wz=weights(z,a,device,value.dtype);wy=weights(y,b,device,value.dtype);wx=weights(x,c,device,value.dtype)
    result=torch.einsum('az,nczyx->ncayx',wz,value)
    result=torch.einsum('by,ncayx->ncabx',wy,result)
    return torch.einsum('dx,ncabx->ncabd',wx,result)

def fixed_pool2(value):
    z,y,x=value.shape[-3:];device=str(value.device)
    wz=weights(z,(z+1)//2,device,value.dtype,'fixed2')
    wy=weights(y,(y+1)//2,device,value.dtype,'fixed2')
    wx=weights(x,(x+1)//2,device,value.dtype,'fixed2')
    result=torch.einsum('az,nczyx->ncayx',wz,value)
    result=torch.einsum('by,ncayx->ncabx',wy,result)
    return torch.einsum('dx,ncabx->ncabd',wx,result)

def model_pool3d(value,output_size):
    return adaptive_pool3d(value,output_size)if value.device.type=='cuda'else F.adaptive_avg_pool3d(value,output_size)

def model_pool2(value):
    return fixed_pool2(value)if value.device.type=='cuda'else F.avg_pool3d(value,2,ceil_mode=True)
