"""Known external force increments from explicit scene schedules."""
import ctypes
from functools import lru_cache
from support import BASE, HERE, read, digest
from recurrent_sampler_native import dimensions
import numpy as np
import torch


@lru_cache(maxsize=1)
def library():
    directory=BASE/'known-force-substeps-build-01'
    receipt=read(directory/'receipt.json');binary=directory/'libscheduled_force.so'
    if digest(binary)!=receipt['binary_sha256'] or digest(HERE/'scheduled_force_substeps.cpp')!=receipt['source_sha256']:
        raise ValueError('Known prescribed-force binary changed')
    lib=ctypes.CDLL(str(binary))
    lib.rheon_scheduled_force_substeps.argtypes=[ctypes.c_int]*6+[ctypes.c_uint64,ctypes.c_void_p]+[ctypes.c_double]*6+[ctypes.c_void_p]*9
    lib.rheon_scheduled_force_substeps.restype=ctypes.c_int
    return lib


def force(scene,step,base=None):
    factor=scene.get('reference_substeps',1)
    if not 0<=step<len(scene['forcing_schedule'])*factor:raise ValueError('Known scheduled step required')
    shapes=dimensions(scene['counts'])[:3]
    base=[torch.zeros(s,dtype=torch.float32)for s in shapes]if base is None else base
    if len(base)!=3 or any(v.dtype!=torch.float32 or v.device.type!='cpu' or not v.is_contiguous() or tuple(v.shape)!=s for v,s in zip(base,shapes)):
        raise ValueError('Complete contiguous CPU single-precision force inputs required')
    frame=scene['forcing_schedule'][step//factor]
    center=np.asarray(frame['center'],dtype='<f8')
    force=[torch.empty(s,dtype=torch.float64)for s in shapes]
    merged=[torch.empty(s,dtype=torch.float32)for s in shapes]
    status=library().rheon_scheduled_force_substeps(*scene['counts'],scene['cells_per_metre'],step,factor,scene['seed'],center.ctypes.data,
        frame['pulse'],scene['acceleration'],scene['radius'],scene['swirl'],scene['fluctuation'],scene['frequency'],
        *[v.data_ptr()for v in (*base,*force,*merged)])
    if status:raise ValueError('Known forcing rejected input contract')
    return force,merged
