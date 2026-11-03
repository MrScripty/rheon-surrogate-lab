"""Lossless byte-plane and prior-array XOR transport for captured native labels.

This encodes bits; it never rounds a floating point value. Arrays are decoded
before CUDA training. The independent native snapshots remain unchanged.
"""
import hashlib
import zlib

import numpy as np


def encode(array, previous=None):
    if array.dtype.str not in ('<f4', '<f8') or not array.flags.c_contiguous:
        raise ValueError('Exact contiguous native little-endian float array required')
    bits = array.view(np.uint8).reshape(-1, array.dtype.itemsize)
    if previous is not None:
        if previous.shape != array.shape or previous.dtype != array.dtype:
            raise ValueError('XOR reference layout differs')
        bits = np.bitwise_xor(bits, previous.view(np.uint8).reshape(bits.shape))
    payload = zlib.compress(bits.T.copy().tobytes(), level=6)
    return dict(dtype=array.dtype.str, shape=list(array.shape), bytes=array.nbytes,
        sha256=hashlib.sha256(array.tobytes()).hexdigest(), xor_previous=previous is not None), payload


def decode(spec, payload, previous=None):
    dtype = np.dtype(spec['dtype'])
    if dtype.str not in ('<f4', '<f8') or type(spec['bytes']) is not int or spec['bytes'] < 0:
        raise ValueError('Invalid native transport schema')
    shape = tuple(spec['shape'])
    if any(type(n) is not int or n < 0 for n in shape) or np.prod(shape, dtype=object) * dtype.itemsize != spec['bytes']:
        raise ValueError('Typed payload size differs')
    codec = zlib.decompressobj()
    raw = codec.decompress(payload, spec['bytes'] + 1)
    if len(raw) != spec['bytes'] or not codec.eof or codec.unconsumed_tail or codec.unused_data:
        raise ValueError('Noncanonical or oversized compressed array')
    bits = np.frombuffer(raw, dtype=np.uint8).reshape(dtype.itemsize, -1).T.copy()
    if spec['xor_previous']:
        if previous is None or previous.dtype != dtype or previous.shape != shape:
            raise ValueError('Exact previous-array layout required')
        bits ^= previous.view(np.uint8).reshape(bits.shape)
    elif previous is not None:
        raise ValueError('Unexpected previous array')
    result = bits.reshape(-1).view(dtype).reshape(shape)
    if hashlib.sha256(result.tobytes()).hexdigest() != spec['sha256']:
        raise ValueError('Lossless array identity differs')
    return result
