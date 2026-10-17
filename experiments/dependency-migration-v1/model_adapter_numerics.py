"""Current model assembly over the exact captured numerical implementation.

The one-time packager supplies numerical sources and a current numerical.json.
Training recipes, Run records and historical deployment formats are not read.
"""
import copy

import torch
import portable_model as owner

KIND = None
CORRECTION = None
ARCHITECTURE = None
full_operator = owner.full_operator
configure = owner.configure


def select(spec):
    global KIND, CORRECTION, ARCHITECTURE
    KIND = spec['kind']
    CORRECTION = copy.deepcopy(spec['correction'])
    ARCHITECTURE = copy.deepcopy(owner.ARCHITECTURE)
    if KIND == 'physical_residual':
        ARCHITECTURE.update(schema='rheon-physical-residual-pressure-v1', input_channels=1,
            channels=['dimensionless_laplacian_rhs'], conditioning='physical-residual',
            coordinates='implicit grid indices; no coordinate feature maps',
            block_normalization={'name': 'Identity'}, correction=copy.deepcopy(CORRECTION))
    elif KIND != 'seven_channel':
        raise ValueError('Unsupported current numerical mapping')


class PressureCorrectionNet(owner.PressureCorrectionNet):
    def __init__(self):
        super().__init__()
        if KIND == 'physical_residual':
            # Preserve the original assembly order and float32 parameter tensors.
            for block in [*self.encoders, self.bottleneck, *self.decoders]:
                for index, layer in enumerate(block.layers):
                    if isinstance(layer, torch.nn.GroupNorm):
                        block.layers[index] = torch.nn.Identity()
            import physical_residual
            physical_residual.transform(self, owner, CORRECTION)

    def forward(self, values):
        if KIND == 'seven_channel':
            return super().forward(values)
        import physical_residual
        return physical_residual.forward(self, values, owner)


def build_inputs(residual, base, cell_size, rho, dt, normalization):
    if KIND == 'seven_channel':
        return owner.build_inputs(residual, base, cell_size, rho, dt, normalization)
    import physical_residual
    return physical_residual.build_inputs(residual, base, cell_size, rho, dt, normalization, owner)


def predict_pressure(network, sample, normalization):
    if KIND == 'seven_channel':
        return owner.predict_pressure(network, sample, normalization)
    import physical_residual
    return physical_residual.predict_pressure(network, sample, normalization, owner, CORRECTION)
