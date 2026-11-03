"""Stable generic learned response plus precise native timestep division."""
import math
import torch
from fused_cuda_generator_v1 import FusedCudaGenerator
from stable_learned_response_v1 import stable_response_grid
from recurrent_sampler_native import dimensions
from cuda_divisor_generator_v1 import FIELDS

class FusedCudaGeneratorV2(FusedCudaGenerator):
    def __init__(self,pair,shared_pair,build):
        super().__init__(pair,shared_pair,build,'transport_and_linear')
        self.stable_geometries=set()
        self.asset_identity.update(numerical_evaluation='stable arbitrary affine Fourier response and CUDA double timestep denominator',
            response_identity='a0+sum(ai*cos(theta_i))=(a0+sum(ai))+sum(ai*(-2*sin(theta_i/2)^2))',
            fixed_feature_preparation='CPU1Dgeneric trig from integer shape only; all learned coefficients/evaluation onCUDA',
            learned_parameters_changed=False,native_reference_oracles_in_candidate=False)

    def prepare_geometry(self,counts):
        counts=tuple(counts)
        super().prepare_geometry(counts)
        if counts in self.stable_geometries:return
        shape=tuple(reversed(counts))
        value=stable_response_grid(self.shared_pressure.coefficients,shape)
        if not bool(value.flatten()[1:].isfinite().all()) or not bool((value.flatten()[1:]>0).all()):
            raise ValueError('Finite positive learned response required for every nonconstant mode')
        value=value.clone();value[0,0,0]=1.
        self.divisors['g'+'_'.join(map(str,shape))]=value
        self.stable_geometries.add(counts)

    @torch.inference_mode()
    def predict(self, fields, scene, step, horizons, external, substeps=1):
        counts = tuple(scene['counts']); shape = tuple(reversed(counts))
        if (horizons not in (1, 2, 3, 4) or substeps not in (1, 8) or not 0 < scene['cells_per_metre'] <= 1024
                or scene.get('sgs', False) or counts not in self.geometry_cache):
            raise ValueError('Prepared registered reference-regime geometry required')
        if set(fields) != set(FIELDS) or len(external) != horizons*substeps:
            raise ValueError('Complete current state and prescribed input block required')
        for name, expected_shape in [('pressure', shape), *zip(FIELDS[1:], dimensions(counts))]:
            value = fields[name]
            if (value.device != self.device or value.dtype != (torch.float64 if name == 'pressure' else torch.float32)
                    or value.numel() != math.prod(expected_shape) or not value.is_contiguous()):
                raise ValueError('Exact native dtype and contiguous CUDA state required')
        for block in external:
            if len(block) != 3 or any(v.device != self.device or v.dtype != torch.float64
                                    or tuple(v.shape) != size for v, size in zip(block, dimensions(counts)[:3])):
                raise ValueError('Complete CUDA double prescribed velocities required')
        current = [fields[name].reshape(size) for name, size in zip(FIELDS[1:], dimensions(counts))]
        previous = fields['pressure'].reshape(1, 1, *shape)
        dt, h, rho = 1/(60*substeps), 1/scene['cells_per_metre'], 1000.
        results = []
        for lead in range(horizons*substeps):
            advected = self.transport(current, counts, dt/h)
            entering = [(value.double()+addition).float() for value, addition in zip(advected[:3], external[lead])]
            provisional = [self.wall((entering[c].double()+self.linear(entering, size, 'viscosity', c)
                                      *(.001/rho*dt/h**2)).float(), c)
                           for c, size in enumerate(dimensions(counts)[:3])]
            rhs = torch.zeros(shape, dtype=torch.float64, device=self.device)
            for channel in range(3):
                rhs = rhs+(h*h)*self.linear(provisional, shape, 'flux', 0, channel)
            # A CUDA double denominator preserves actual IEEE division;
            # scalar reciprocal folding changes the native pressure RHS.
            rhs = rhs/torch.full((),dt,dtype=torch.float64,device=self.device)
            history = []
            scale = (h*h)*(1/(rho*h))
            for component, size in enumerate(dimensions(counts)[:3]):
                history.append(self.wall(self.linear([previous[0, 0]], size, 'momentum', component)*scale, component))
            action = self.linear(history, shape, 'flux', 0, sign=-1)
            residual = rhs-action
            correction = self.pressure((residual*(rho/h)).reshape(1, 1, *shape), counts)
            pressure = previous+correction
            pressure = pressure-pressure[:, :, :1, :1, :1]
            velocity = [self.wall((provisional[c].double()+self.linear([pressure[0, 0]], size, 'momentum', c)
                                   *(dt*(1/(rho*h)))).float(), c)
                        for c, size in enumerate(dimensions(counts)[:3])]
            if (lead+1) % substeps == 0:
                results.append(dict(pressure=pressure.contiguous().flatten(),
                    vx=velocity[0].flatten(), vy=velocity[1].flatten(), vz=velocity[2].flatten(), tracer=advected[3].flatten()))
            current, previous = [*velocity, advected[3]], pressure
        return results
