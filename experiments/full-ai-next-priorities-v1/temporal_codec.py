"""Lossless float32 trajectory transport; temporal XOR followed by byte planes."""
import numpy as np

def encode(value):
    value=np.ascontiguousarray(value,dtype='<f4')
    if value.ndim!=2 or not np.isfinite(value).all():raise ValueError('Finite T by field-length float32 trajectory required')
    bits=value.view('<u4');delta=bits.copy();delta[1:]=np.bitwise_xor(bits[1:],bits[:-1])
    return delta.view('u1').reshape(*value.shape,4).transpose(2,0,1).copy()

def decode(value):
    if value.dtype!=np.uint8 or value.ndim!=3 or value.shape[0]!=4:raise ValueError('Four byte planes required')
    steps,cells=value.shape[1:]
    delta=value.transpose(1,2,0).copy().reshape(steps,cells*4).view('<u4')
    return np.bitwise_xor.accumulate(delta,axis=0).view('<f4')
