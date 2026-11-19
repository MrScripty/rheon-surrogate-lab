"""CPU double DCT multiplier execution; coefficients must be supplied by caller.

This implements a change of basis, not a pressure inverse. It has no physical
operator, eigenvalues, fitted coefficients, reference labels or solver fallback.
Training uses the separately qualified differentiable Torch representation.
"""
import ctypes
import math
from pathlib import Path
import torch


class DCTMultiplier:
    """Own single-threaded FFTW plans for one fixed B,C,Z,Y,X layout."""
    def __init__(self, shape, library_path):
        if len(shape) != 5 or not all(type(n) is int and n > 0 for n in shape):
            raise ValueError('A concrete positive B,C,Z,Y,X layout is required')
        self.shape = tuple(shape)
        self.volume = math.prod(shape[2:])
        self.lib = ctypes.CDLL(str(Path(library_path).resolve()))
        pointer = ctypes.c_void_p
        integers = ctypes.POINTER(ctypes.c_int)
        planner = self.lib.fftw_plan_many_r2r
        planner.argtypes = [ctypes.c_int, integers, ctypes.c_int, pointer, integers,
                            ctypes.c_int, ctypes.c_int, pointer, integers, ctypes.c_int,
                            ctypes.c_int, integers, ctypes.c_uint]
        planner.restype = pointer
        self.lib.fftw_execute_r2r.argtypes = [pointer, pointer, pointer]
        self.lib.fftw_execute_r2r.restype = None
        self.lib.fftw_destroy_plan.argtypes = [pointer]
        self.lib.fftw_destroy_plan.restype = None
        self.lib.fftw_alignment_of.argtypes = [pointer]
        self.lib.fftw_alignment_of.restype = ctypes.c_int
        # FFTW_ESTIMATE does not modify planning arrays. Only the ordinary
        # serial library is loaded; no threaded FFTW API is initialized.
        dimensions = (ctypes.c_int * 3)(*shape[2:])
        source = torch.empty(shape, dtype=torch.float64)
        target = torch.empty_like(source)
        self.alignments = (self.lib.fftw_alignment_of(source.data_ptr()),
                           self.lib.fftw_alignment_of(target.data_ptr()))
        self.plans = []
        for kind in (5, 4):  # FFTW_REDFT10 (II), FFTW_REDFT01 (III).
            kinds = (ctypes.c_int * 3)(kind, kind, kind)
            plan = planner(3, dimensions, shape[0] * shape[1], source.data_ptr(), None,
                           1, self.volume, target.data_ptr(), None, 1, self.volume, kinds, 64)
            if not plan:
                self.close()
                raise RuntimeError('FFTW refused the declared real transform plan')
            self.plans.append(plan)

    def _check(self, value):
        if tuple(value.shape) != self.shape or value.device.type != 'cpu' or value.dtype != torch.float64 or not value.is_contiguous():
            raise ValueError('Exact contiguous CPU double plan layout required')
        if value.requires_grad:
            raise ValueError('This backend is inference only')

    def _execute(self, index, source, target):
        self._check(source); self._check(target)
        if len(self.plans) != 2:
            raise RuntimeError('Closed transform owner')
        alignments = (self.lib.fftw_alignment_of(source.data_ptr()),
                      self.lib.fftw_alignment_of(target.data_ptr()))
        if alignments != self.alignments:
            raise ValueError('New arrays differ from registered FFTW plan alignment')
        self.lib.fftw_execute_r2r(self.plans[index], source.data_ptr(), target.data_ptr())

    def __call__(self, value, gains):
        self._check(value)
        if gains.device.type != 'cpu' or gains.dtype != torch.float64 or gains.requires_grad:
            raise ValueError('Frozen CPU double multiplier coefficients required')
        if tuple(gains.shape) not in (self.shape, (1, 1, *self.shape[2:])):
            raise ValueError('Exact colocated or shared spatial gain layout required')
        if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(gains).all()):
            raise FloatingPointError('Finite transform inputs required')
        coefficients = torch.empty_like(value)
        output = torch.empty_like(value)
        self._execute(0, value, coefficients)
        coefficients.mul_(gains)
        self._execute(1, coefficients, output)
        # Per-axis inverse scale is 2*N. This is transform normalization only.
        output.mul_(1. / (8 * self.volume))
        return output

    def close(self):
        for plan in getattr(self, 'plans', ()):
            self.lib.fftw_destroy_plan(plan)
        self.plans = []

    def __del__(self):
        self.close()
