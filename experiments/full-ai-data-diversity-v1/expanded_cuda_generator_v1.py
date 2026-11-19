"""Complete CUDA generator with restored shape-shared learned pressure.

Internal substeps are explicit candidate work, counted in output-step latency.
No future pressure/RHS/native physical labels enter candidate inference.
"""
import math
import torch
from support import ROOT,read,digest
from cuda_divisor_generator_v1 import FIELDS
from cuda_divisor_generator_v2 import CudaDivisorGeneratorV2
from recurrent_sampler_native import dimensions
from shared_spectral_model_v1 import SharedSpectralResponse
from train_shared_spectral_v1 import load as load_weights
from modal_basis import bases
from scheduled_force_tables import force


def prescribed_controls(scene,step,horizons,substeps,device='cuda'):
    """Known external program only; includes host construction and upload."""
    if substeps not in (1,8) or horizons not in (1,2,3,4) or not 0<=step<step+horizons<=scene['steps']:
        raise ValueError('Registered bounded complete control interval required')
    prescribed=dict(scene,reference_substeps=substeps,forcing_control_substeps=1)
    if 'forcing_schedule' not in prescribed or len(prescribed['forcing_schedule'])!=scene['steps']:
        raise ValueError('Expanded sampled-control schedule required')
    return [[value.to(device) for value in force(prescribed,step*substeps+fine)[0]]
            for fine in range(horizons*substeps)]


class ExpandedCudaGenerator(CudaDivisorGeneratorV2):
    def __init__(self,transport_pair='runs/full-ai-tight-targets-v1/sampler-head-pair-01',
                 shared_pair='runs/full-ai-data-diversity-v1/shared-spectral-pair-01'):
        # This parent constructor restores tensors only; no CPU forward occurs.
        super().__init__(transport_pair,'runs/full-ai-tight-targets-v1/mixed-modal-pair-11')
        del self.pressure_model
        pair=ROOT/shared_pair;receipt=read(pair/'receipt.json');restored=read(pair/'restore/receipt.json')
        if (receipt['status']!='completed' or receipt['optimizer_updates']!=160 or not all(receipt['artifact_parity'].values())
                or not restored['typed_cpu_weight_restore_verified'] or restored['cpu_model_forward_executed']):
            raise ValueError('Managed learned response and exact typed CPU weight restore required')
        for key in ('training_run_id','restore_run_id'):
            terminal=read(pair/('terminal-'+receipt[key]+'.json'))
            if terminal['state']!='completed' or terminal['custody_phase']!='drained':raise ValueError('Completed managed custody required')
        path=pair/'restore/weights.npz'
        if digest(path)!=restored['outputs']['weights.npz']['sha256']:raise ValueError('Restored learned response changed')
        self.shared_pressure=SharedSpectralResponse('cuda');load_weights(self.shared_pressure,path,restored['final_inventory'])
        self.shared_pressure.eval().requires_grad_(False)
        self.asset_identity['pressure']=dict(pair=shared_pair,receipt_sha256=digest(pair/'receipt.json'),weights_sha256=digest(path),
            parameter_inventory=restored['final_inventory'],training_scenes=148,component_development_only=True)
        self.asset_identity.update(logical_model_id='expanded-full-state-shared-pressure-01',inference_device='cuda',
            internal_substep_contract='One or eight explicit learned advances per published1/60-second output. Count all work.',
            geometry_tables_used_for_pressure=False,numerical_pressure_solver=False,optimizer_updates=0)
        self.eval().requires_grad_(False)

    def prepare_geometry(self,counts):
        counts=tuple(counts)
        if counts in self.geometry_cache:return
        if len(counts)!=3 or any(type(n)is not int or not 1<=n<=1024 for n in counts):raise ValueError('Concrete bounded grid required')
        shape=tuple(reversed(counts));coordinates=[]
        for carrier in dimensions(counts):
            z,y,x=torch.meshgrid(*(torch.arange(n,dtype=torch.float64,device=self.device) for n in carrier),indexing='ij')
            coordinates.append(torch.stack((x,y,z)))
        axes=tuple(axis.to(self.device) for axis in bases(shape))
        response=self.shared_pressure.response_grid(shape)
        if not bool(response.flatten()[1:].isfinite().all()) or not bool((response.flatten()[1:]>0).all()):
            raise ValueError('Finite positive learned response required for every nonconstant mode')
        response=response.clone();response[0,0,0]=1.
        self.geometry_cache[counts]=(coordinates,axes)
        self.divisors['g'+'_'.join(map(str,shape))]=response

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
            rhs = rhs/dt
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
