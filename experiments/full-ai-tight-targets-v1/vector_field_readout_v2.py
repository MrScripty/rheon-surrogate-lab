"""Strict native-ISA execution of the fixed neural displacement field readout."""
import ctypes
from functools import lru_cache
from pathlib import Path
from profile_frozen_components import ROOT,read,digest
import torch


@lru_cache(maxsize=1)
def library():
    directory=ROOT/'runs/full-ai-tight-targets-v1/vector-field-readout-build-02'
    receipt=read(directory/'receipt.json');binary=directory/'libvector_field_readout_v2.so'
    if digest(binary)!=receipt['binary_sha256']or digest(Path(__file__).with_suffix('.cpp'))!=receipt['source_sha256']:
        raise ValueError('Registered strict readout changed')
    lib=ctypes.CDLL(str(binary))
    lib.rheon_neural_field_readout.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int64]*3+[ctypes.c_int]
    lib.rheon_neural_field_readout.restype=ctypes.c_int
    return lib


def convex_field_readout(fields,displacements):
    if fields.device.type!='cpu'or displacements.device.type!='cpu'or fields.dtype!=torch.float32:
        raise ValueError('Strict readout requires CPU float32 carrier fields')
    if fields.ndim!=5 or fields.shape[:2]!=(1,4)or displacements.shape!=(1,4,3,*fields.shape[2:]):
        raise ValueError('Four carriers and their neural displacements required')
    if displacements.dtype not in(torch.float32,torch.float64):raise ValueError('Float32/64 displacement required')
    if torch.is_grad_enabled()and(fields.requires_grad or displacements.requires_grad):
        raise ValueError('Strict CPU readout is inference only')
    values=fields.contiguous();shifts=displacements.contiguous();output=torch.empty_like(values)
    nz,ny,nx=values.shape[2:]
    status=library().rheon_neural_field_readout(values.data_ptr(),shifts.data_ptr(),output.data_ptr(),nz,ny,nx,0 if shifts.dtype==torch.float32 else 1)
    if status in(2,3):raise FloatingPointError('Finite carrier fields and neural shifts required')
    if status==4:raise ValueError('Registered displacement support exceeded')
    if status:raise ValueError('Strict readout geometry contract failed')
    return output
