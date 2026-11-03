"""Typed learned source tensors and a general uniform inference quantization."""
import hashlib
from pathlib import Path
from profile_frozen_components import ROOT,read,digest
import numpy as np
import torch


def source_tensors():
    pair=ROOT/'runs/full-ai-next-priorities-v1/b1-fixed-viscosity-pair-01'
    receipt=read(pair/'receipt.json');restored=read(pair/'restore/receipt.json');weights=pair/'restore/weights.npz'
    if receipt['status']!='completed' or not restored['full_cpu_probe_bitwise_verified'] or restored['optimizer_updates']!=0:
        raise ValueError('Exact inherited managed CPU restore required')
    if digest(weights)!=restored['outputs']['weights.npz']['sha256']:raise ValueError('Inherited learned tensors changed')
    keys={'point':'displacement.point_weight','viscosity':'viscous_head.weight','flux':'flux.weight'}
    tensors={};provenance={}
    with np.load(weights,allow_pickle=False)as archive:
        for name,key in keys.items():
            raw=archive[key].copy();expected=restored['final_inventory'][key]
            if raw.dtype.str!='<f4' or list(raw.shape)!=expected['shape']or hashlib.sha256(raw.tobytes()).hexdigest()!=expected['sha256']:
                raise ValueError('Inherited typed learned coefficient identity failed')
            value=torch.from_numpy(raw).double();quantized=(value/(2.**-40)).round()*(2.**-40)
            tensors[name]=quantized
            provenance[name]=dict(source_parameter=key,source_sha256=expected['sha256'],source_dtype='<f4',
                quantization_absolute_step=2.**-40,quantized_dtype='<f8',
                quantized_sha256=hashlib.sha256(quantized.numpy().tobytes()).hexdigest(),
                maximum_coefficient_difference=float((value-quantized).abs().max()),nonzero_coefficients=int(quantized.count_nonzero()))
    tensors['momentum']=-tensors['flux'].flip(dims=(2,3,4)).transpose(0,1).contiguous()
    return tensors,dict(parent_pair=str(pair.relative_to(ROOT)),parent_weights_sha256=digest(weights),tensors=provenance,
        momentum_provenance='Negative transpose/spatial flip of the inherited learned flux tensor',optimizer_updates=0,
        analytic_stencil_assigned=False)


def csr(weight):
    if weight.dtype!=torch.float64 or weight.ndim!=5 or weight.shape[2:]!=(3,3,3):raise ValueError('Declared double learned convolution required')
    rows=[0];channels=[];offsets=[];values=[]
    for head in weight:
        for c in range(head.shape[0]):
            for z in range(3):
                for y in range(3):
                    for x in range(3):
                        value=float(head[c,z,y,x])
                        if value!=0:channels.append(c);offsets.append([x-1,y-1,z-1]);values.append(value)
        rows.append(len(values))
    return tuple(torch.tensor(value,dtype=dtype).contiguous()for value,dtype in
        ((rows,torch.int32),(channels,torch.int32),(offsets,torch.int32),(values,torch.float64)))
