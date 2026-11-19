"""CPU inference execution of the fixed readout using learned displacements."""
import ctypes
from functools import lru_cache
import torch
from study_support import BASE,HERE,read,sha


@lru_cache(maxsize=1)
def library():
    directory=BASE/'fused-field-readout-build-01';receipt=read(directory/'receipt.json')
    binary=directory/'libneural_field_readout.so'
    if sha(binary)!=receipt['binary_sha256']or sha(HERE/'fused_field_readout.cpp')!=receipt['source_sha256']:
        raise ValueError('Qualified fixed-readout binary/source changed')
    value=ctypes.CDLL(str(binary.resolve()))
    value.rheon_neural_field_readout.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
        ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_int]
    value.rheon_neural_field_readout.restype=ctypes.c_int
    return value


def convex_field_readout(fields,displacements):
    if fields.device.type!='cpu'or displacements.device.type!='cpu'or fields.dtype!=torch.float32:
        raise ValueError('Fused readout requires CPU float32 carrier fields')
    if fields.ndim!=5 or fields.shape[:2]!=(1,4)or displacements.shape!=(1,4,3,*fields.shape[2:]):
        raise ValueError('Four carrier fields and their neural displacements required')
    if displacements.dtype not in (torch.float32,torch.float64):raise ValueError('Float32/64 neural displacements required')
    if torch.is_grad_enabled()and(fields.requires_grad or displacements.requires_grad):
        raise ValueError('Fused CPU readout is inference-only; use the registered training readout for gradients')
    source=fields.contiguous();shifts=displacements.contiguous();output=torch.empty_like(source)
    nz,ny,nx=source.shape[2:]
    status=library().rheon_neural_field_readout(source.data_ptr(),shifts.data_ptr(),output.data_ptr(),nz,ny,nx,0 if shifts.dtype==torch.float32 else 1)
    if status in (2,3):raise FloatingPointError('Finite fields and learned displacements required')
    if status==4:raise ValueError('Registered subcell readout support exceeded')
    if status!=0:raise ValueError('Fused readout shape/pointer contract failed')
    return output
