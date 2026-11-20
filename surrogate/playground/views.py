"""Two-dimensional views of a staggered state: one plane of cells, at cell centres.

A plane is named by the axis it cuts and its cell index. Its array has rows along the plane's vertical axis and
columns along its horizontal axis, row 0 at the low end, so it draws with the vertical axis pointing up:

    cut z: horizontal x, vertical y        cut y: horizontal x, vertical z        cut x: horizontal z, vertical y
"""
import torch

FIELDS = ('speed', 'vx', 'vy', 'vz', 'pressure', 'tracer', 'vorticity', 'divergence')
SIGNED = frozenset(('vx', 'vy', 'vz', 'pressure', 'divergence'))
UNITS = {'speed': 'm/s', 'vx': 'm/s', 'vy': 'm/s', 'vz': 'm/s', 'pressure': 'Pa', 'tracer': '',
         'vorticity': '1/s', 'divergence': '1/s'}
AXES = ('x', 'y', 'z')
PLANES = {'z': ('x', 'y'), 'y': ('x', 'z'), 'x': ('z', 'y')}    # (horizontal, vertical)
_DIM = {'x': 2, 'y': 1, 'z': 0}                                    # tensor dimension of each axis


def _cells(counts, axis):
    return counts['xyz'.index(axis)]


def _plane(volume, axis, index):
    """The 2D plane of a [Z, Y, X]-ordered (sub)volume, rows vertical, columns horizontal."""
    plane = volume.select(_DIM[axis], index)
    return plane.transpose(0, 1) if axis == 'x' else plane


def _centred(u, v, w, axis, lo, hi):
    """Cell-centred velocity (x, y, z) of the cells lo..hi-1 along `axis`, [Z, Y, X]-ordered slabs."""
    dim = _DIM[axis]
    out = []
    for d, face in enumerate((u, v, w)):
        face_dim = 2 - d
        slab = face.narrow(dim, lo, hi - lo + (1 if face_dim == dim else 0))
        n = slab.shape[face_dim]
        out.append(0.5 * (slab.narrow(face_dim, 0, n - 1) + slab.narrow(face_dim, 1, n - 1)))
    return out


def check(counts, axis, index, field):
    """Raise ValueError unless (axis, index, field) names a plane and field of a grid with `counts` (x, y, z)."""
    if axis not in AXES:
        raise ValueError(f'The axis is one of {", ".join(AXES)}.')
    if field not in FIELDS:
        raise ValueError(f'The field is one of {", ".join(FIELDS)}.')
    if not 0 <= index < _cells(counts, axis):
        raise ValueError(f'The {axis} index is 0 to {_cells(counts, axis) - 1}.')


def plane(state, counts, spacing, axis, index, field):
    """One plane of `field` as float32 [rows, columns] on the CPU.

    `state` maps u, v, w (staggered faces), pressure and tracer to [Z, Y, X]-ordered tensors on any device.
    """
    check(counts, axis, index, field)
    u, v, w = state['u'], state['v'], state['w']
    if field in ('tracer', 'pressure'):
        values = _plane(state[field], axis, index)
    elif field == 'divergence':
        dim = _DIM[axis]
        div = 0.0
        for d, face in enumerate((u, v, w)):
            face_dim = 2 - d
            slab = face.narrow(dim, index, 2 if face_dim == dim else 1).double()
            n = slab.shape[face_dim]
            div = div + slab.narrow(face_dim, 1, n - 1) - slab.narrow(face_dim, 0, n - 1)
        values = _plane(div / spacing, axis, 0)
    elif field == 'vorticity':
        n = _cells(counts, axis)
        lo, hi = max(index - 1, 0), min(index + 2, n)
        cx, cy, cz = (c.double() for c in _centred(u, v, w, axis, lo, hi))

        def grad(f, a):
            if f.shape[_DIM[a]] < 2:
                return torch.zeros_like(f)
            return torch.gradient(f, spacing=spacing, dim=_DIM[a])[0]
        wx = grad(cz, 'y') - grad(cy, 'z')
        wy = grad(cx, 'z') - grad(cz, 'x')
        wz = grad(cy, 'x') - grad(cx, 'y')
        values = _plane(torch.sqrt(wx * wx + wy * wy + wz * wz), axis, index - lo)
    else:
        cx, cy, cz = _centred(u, v, w, axis, index, index + 1)
        if field == 'speed':
            values = torch.sqrt(cx.double() ** 2 + cy.double() ** 2 + cz.double() ** 2)
        else:
            values = {'vx': cx, 'vy': cy, 'vz': cz}[field]
        values = _plane(values, axis, 0)
    return values.float().cpu().contiguous()


def in_plane_velocity(state, counts, axis, index):
    """The velocity components along the plane's horizontal and vertical axes, float32 [rows, columns] each."""
    check(counts, axis, index, 'speed')
    centred = dict(zip('xyz', _centred(state['u'], state['v'], state['w'], axis, index, index + 1)))
    horizontal, vertical = PLANES[axis]
    return tuple(_plane(centred[a], axis, 0).float().cpu().contiguous() for a in (horizontal, vertical))
