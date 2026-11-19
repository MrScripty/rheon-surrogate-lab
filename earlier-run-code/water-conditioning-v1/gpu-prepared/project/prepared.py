"""Prepare immutable conditioning and loss terms once; keep learned state checked."""
import math
import torch
from torch.nn import functional as F
from policy import require
import model


class PreparedSample:
    """Own static training tensors after canonical sample/physics validation.

    This path supports fully active captured samples. Publication and deployment
    keep the ordinary checked model API, including its exact-zero bypass.
    """
    def __init__(self,sample,normalization):
        require(not any(value.requires_grad for value in sample.values()),'prepared_inputs','immutable training fields')
        # This consumes the canonical field, scalar, shape, finite and operator
        # contracts once, including reference/RHS checks independent of inputs.
        with torch.no_grad():
            model.pressure_loss(sample['base'],sample,normalization)
            require(bool((sample['native_input_residual'].flatten(1)!=0).any(dim=1).all()),
                    'prepared_inputs','all rows must have nonzero residual; use ordinary path for exact solutions')
            self.values=model.build_inputs(sample['native_input_residual'],sample['base'],sample['cell_size_m'],sample['rho'],sample['dt'],normalization)
            self.selected=torch.nonzero((sample['native_input_residual'].flatten(1)!=0).any(dim=1),as_tuple=False).flatten()
            self.sample=sample
            self.gain=model.amplitude(sample['native_input_residual'],normalization).reshape(-1,1,1,1,1)
            scale=model.owner.scales(normalization);self.pressure_scale=scale['pressure']
            cells=sample['base'][0].numel()
            self.denominator_p=sample['reference'].flatten(1).square().sum(dim=1).clamp_min((1e-6*math.sqrt(cells)*scale['pressure'])**2)
            self.denominator_b=sample['rhs'].flatten(1).square().sum(dim=1).clamp_min((1e-6*math.sqrt(cells)*scale['rhs'])**2)
            spacing=sample['cell_size_m'][:,None].expand(-1,3)
            self.axes=[]
            for physical_axis,tensor_axis in enumerate((4,3,2)):
                other=[axis for axis in range(3) if axis!=physical_axis]
                area=spacing[:,other[0]]*spacing[:,other[1]]
                weight=(area*(1.0/(sample['rho']*spacing[:,physical_axis]))).reshape(-1,1,1,1,1)
                lower=[0]*6;upper=[0]*6;lower[2*physical_axis]=1;upper[2*physical_axis+1]=1
                self.axes.append((tensor_axis,weight,lower,upper))

    def operator(self,pressure):
        result=torch.zeros_like(pressure)
        for axis,weight,lower,upper in self.axes:
            difference=pressure.narrow(axis,1,pressure.shape[axis]-1)-pressure.narrow(axis,0,pressure.shape[axis]-1)
            result=result+weight*(F.pad(difference,lower)-F.pad(difference,upper))
        return result

    def prediction(self,network):
        raw=network(self.values.index_select(0,self.selected))
        require(raw.dtype==torch.float32 and raw.shape==self.sample['base'].shape
                and bool(torch.isfinite(raw).all()),'training_nonfinite','finite same-grid float32 raw correction')
        correction=model.frozen.gauge_correction(raw,self.pressure_scale)*self.gain.index_select(0,self.selected)
        base=self.sample['base']
        pressure=base.clone().index_copy(0,self.selected,base.index_select(0,self.selected)+correction)
        require(bool(torch.isfinite(pressure).all()),'training_nonfinite','finite reconstructed pressure')
        return pressure

    def loss(self,network):
        pressure=self.prediction(network)
        residual=self.operator(pressure)-self.sample['rhs']
        return .5*(pressure-self.sample['reference']).flatten(1).square().sum(dim=1)/self.denominator_p + .5*residual.flatten(1).square().sum(dim=1)/self.denominator_b
