"""Uniform GPU deployment precision for the managed learned shared head.

All four inherited coefficients use one quantization rule. No physical operator
coefficient is assigned. Raw managed artifacts and field dtypes stay unchanged.
"""
import torch
from expanded_cuda_generator_v1 import ExpandedCudaGenerator
from train_shared_spectral_v1 import inventory


class ExpandedCudaGeneratorV2(ExpandedCudaGenerator):
    def __init__(self,transport_pair,shared_pair):
        super().__init__(transport_pair,shared_pair)
        raw=inventory(self.shared_pressure)
        step=2.**-36
        with torch.no_grad():
            parameter=self.shared_pressure.coefficients
            quantized=(parameter/step).round()*step
            maximum_change=float((quantized-parameter).abs().max())
            parameter.copy_(quantized)
        self.asset_identity.update(logical_model_id='expanded-full-state-shared-pressure-02',
            shared_pressure_deployment_precision=dict(kind='uniform_absolute_quantization',step=step,
                scope='All four managed-restored learned double coefficients, onGPU before geometry preparation',
                maximum_absolute_change=maximum_change,raw_inventory=raw,deployed_inventory=inventory(self.shared_pressure),
                raw_managed_artifacts_unchanged=True,optimizer_updates=0,physical_coefficients_assigned=False))
