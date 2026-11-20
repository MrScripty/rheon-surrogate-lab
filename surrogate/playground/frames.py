"""A frame: the planes the page shows, compact enough to send 30 times a second.

Each plane is quantised to uint8 over its range (symmetric about zero for signed fields, 0 to 1 for the tracer) and
sent as base64, row 0 at the bottom. Arrows are the in-plane velocity averaged over blocks of cells, as int8 scaled
by the largest arrow, with that speed in m/s.
"""
import base64
import math

import torch
from torch.nn import functional as F

from . import views

FIXED_RANGES = {'tracer': (0.0, 1.0)}
ARROWS_PER_SIDE = 24


def _b64(tensor):
    return base64.b64encode(tensor.contiguous().numpy().tobytes()).decode('ascii')


def encode(values, field):
    """A float32 [rows, columns] plane as uint8 over its range."""
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if field in FIXED_RANGES:
        lo, hi = FIXED_RANGES[field]
    elif field in views.SIGNED:
        m = float(values.abs().max()) if values.numel() else 0.0
        lo, hi = -m, m
    else:
        lo, hi = (0.0, float(values.max())) if values.numel() else (0.0, 0.0)
    span = hi - lo
    data = torch.zeros_like(values) if span <= 0 else ((values - lo) / span * 255).round()
    rows, columns = values.shape
    return dict(min=lo, max=hi, height=rows, width=columns, data=_b64(data.clamp(0, 255).to(torch.uint8)))


def arrows(horizontal, vertical, per_side=ARROWS_PER_SIDE):
    """Block-averaged in-plane velocity as int8 pairs."""
    rows, columns = horizontal.shape
    block = max(1, math.ceil(max(rows, columns) / per_side))
    h = F.avg_pool2d(horizontal[None, None].double(), block, ceil_mode=True)[0, 0]
    v = F.avg_pool2d(vertical[None, None].double(), block, ceil_mode=True)[0, 0]
    speed = float(torch.sqrt(h * h + v * v).max()) if h.numel() else 0.0
    scale = 127 / speed if speed > 0 else 0.0
    quantise = lambda t: (t * scale).round().clamp(-127, 127).to(torch.int8)
    return dict(block=block, height=h.shape[0], width=h.shape[1], max_speed=speed,
                h=_b64(quantise(h)), v=_b64(quantise(v)))


def parse_view(value, counts):
    """A view request {axis, index, field} checked against the grid; a missing index means the middle."""
    axis = value.get('axis', 'z')
    field = value.get('field', 'speed')
    n = counts['xyz'.index(axis)] if axis in views.AXES else 1
    try:
        index = int(value['index']) if value.get('index') not in (None, '') else n // 2
    except (TypeError, ValueError):
        raise ValueError('The slice index is a whole number.') from None
    views.check(counts, axis, index, field)
    return dict(axis=axis, index=index, field=field)


def render(backend, requested, with_arrows=False):
    """The planes of `requested` views (already parsed) from the backend's current state; arrows on the first."""
    out = []
    for position, view in enumerate(requested):
        plane = backend.slice(view['axis'], view['index'], view['field'])
        horizontal, vertical = views.PLANES[view['axis']]
        item = dict(view, horizontal=horizontal, vertical=vertical, unit=views.UNITS[view['field']],
                    signed=view['field'] in views.SIGNED, **encode(plane, view['field']))
        if with_arrows and position == 0:
            item['arrows'] = arrows(*backend.in_plane(view['axis'], view['index']))
        out.append(item)
    return out
