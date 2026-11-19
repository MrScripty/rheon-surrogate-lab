"""Cancellation-resistant evaluation of an arbitrary learned affine Fourier map.

CPU work prepares generic one-dimensional trig features from integer shape only.
All four trained coefficients, their bias contraction and response evaluation
remainCUDA. The cos(theta)-1 identity supplies no physical operator coefficient.
"""
import math
import torch

def stable_response_grid(coefficients,shape):
    if coefficients.device.type!='cuda' or coefficients.dtype!=torch.float64 or tuple(coefficients.shape)!=(4,):
        raise ValueError('Four restored learned CUDA double coefficients required')
    terms=[]
    for size,index in zip(shape,(3,2,1)):
        if type(size)is not int or not 1<=size<=1024:raise ValueError('Bounded concrete shape required')
        phase=math.pi*torch.arange(size,dtype=torch.float64,device='cpu')/(2*size)
        feature=-2*torch.sin(phase).square()
        terms.append(coefficients[index]*feature.to(coefficients.device))
    dc=coefficients[0]+coefficients[1]+coefficients[2]+coefficients[3]
    return ((dc+terms[0][:,None,None])+terms[1][None,:,None])+terms[2][None,None,:]
