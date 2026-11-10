"""Small synthetic states for tests. Nothing here resembles a real scene."""
import torch

from surrogate.fields import FluidState, Scene, curl, seal_potential


def random_potential(batch, grid, generator, dtype=torch.float64):
    z, y, x = grid
    shapes = ((z + 1, y + 1, x), (z + 1, y, x + 1), (z, y + 1, x + 1))
    return tuple(torch.randn(batch, *shape, generator=generator, dtype=dtype) for shape in shapes)


def sealed_state(batch, grid, seed=0, dtype=torch.float64):
    """A random state whose velocity is divergence free with no flow through the walls."""
    generator = torch.Generator().manual_seed(seed)
    u, v, w = curl(*seal_potential(*random_potential(batch, grid, generator, dtype)))
    cells = lambda: torch.randn(batch, *grid, generator=generator, dtype=dtype)
    return FluidState(u=u, v=v, w=w, tracer=cells(), pressure=cells())


def scene(batch, spacing=1 / 16, dtype=torch.float64):
    full = lambda value: torch.full((batch,), value, dtype=dtype)
    return Scene(spacing=full(spacing), timestep=full(1 / 60), density=full(1000.0), viscosity=full(1e-3))


def forcing(batch, outputs, grid, seed=1, dtype=torch.float64):
    z, y, x = grid
    generator = torch.Generator().manual_seed(seed)
    shapes = ((z, y, x + 1), (z, y + 1, x), (z + 1, y, x))
    return tuple(0.01 * torch.randn(batch, outputs, *shape, generator=generator, dtype=dtype) for shape in shapes)
