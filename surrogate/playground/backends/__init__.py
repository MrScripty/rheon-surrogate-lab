"""Playground backends by name. Each module is imported only when its backend is asked for."""
import importlib

from .base import Backend, SceneSpec, Unsupported  # noqa: F401

BACKENDS = {
    'composition': ('composition', 'CompositionBackend'),
    'generator': ('generator', 'GeneratorBackend'),
    'reference': ('reference', 'ReferenceBackend'),
    'fake': ('fake', 'FakeBackend'),
}


def backend_class(name):
    if name not in BACKENDS:
        raise Unsupported(f'There is no backend {name!r}; the backends are {", ".join(BACKENDS)}.')
    module, cls = BACKENDS[name]
    return getattr(importlib.import_module(f'.{module}', __name__), cls)


def listing():
    """Every backend with whether it can run here and, if not, why."""
    rows = []
    for name in BACKENDS:
        cls = backend_class(name)
        ok, reason = cls.available()
        rows.append(dict(id=name, label=cls.label, kind=cls.kind, device=cls.device, available=ok, reason=reason,
                         grids=[list(g) for g in cls.grids()], cells_per_metre=list(cls.cells_per_metre_options())))
    return rows
