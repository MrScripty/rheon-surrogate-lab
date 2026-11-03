"""Coarse policies stopping on both residual and native predicted-divergence gates."""
import importlib
import torch
import numerical

def fine_step(state,sample,current):
    # Call the retained physical fine network without its training-time wrapper.
    physical=importlib.import_module('historical_physical' if (state.project/'historical_physical.py').exists() else 'physical_residual')
    residual=sample['native_input_residual']-state.model.full_operator(current-sample['base'],sample['cell_size_m'],sample['rho'])
    return physical.predict_pressure(state.network,dict(sample,base=current,native_input_residual=residual),
        state.normalization,state.model.owner,{**state.model.CORRECTION,'steps':1})

def run(state,sample,nodes=4,placement='each',max_passes=8,adaptive=False,tolerance=1e-6):
    if placement not in ('once','each'):raise ValueError('Declared placement required')
    pressure=sample['base'];base=pressure;initial=sample['native_input_residual']
    h=sample['cell_size_m'];rho=sample['rho']
    denominator=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
    limit=max(1e-12,tolerance*denominator)
    def qualified(residual):
        predicted=float(sample['dt'][0]*residual.abs().max()/sample['cell_size_m'][0]**3)
        return float(torch.linalg.vector_norm(residual))<=limit and predicted<=1e-7
    trace=[];calls=0;coarse_calls=0
    for stage in range(max_passes):
        residual=initial-state.model.full_operator(pressure-base,h,rho)
        if adaptive and qualified(residual):break
        if placement=='each' or stage==0:
            pressure=pressure+numerical.correction(residual,h,rho,nodes);coarse_calls+=1
            residual=initial-state.model.full_operator(pressure-base,h,rho)
            if adaptive and qualified(residual):break
        pressure=fine_step(state,sample,pressure);calls+=1
        residual=initial-state.model.full_operator(pressure-base,h,rho)
        relative=float(torch.linalg.vector_norm(residual))/denominator
        trace.append(dict(stage=stage+1,relative_residual=relative))
        if adaptive and qualified(residual):break
    residual=initial-state.model.full_operator(pressure-base,h,rho)
    return pressure,dict(network_calls=calls,coarse_calls=coarse_calls,trace=trace,
        relative_residual=float(torch.linalg.vector_norm(residual))/denominator)
