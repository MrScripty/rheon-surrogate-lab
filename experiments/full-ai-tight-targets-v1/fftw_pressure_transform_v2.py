"""Bounded measured-plan policy for the same double multiplier computation."""
import ctypes
import time
import torch
from fftw_pressure_transform import DCTMultiplier


class MeasuredDCTMultiplier(DCTMultiplier):
    def __init__(self, shape, library_path, seconds_per_plan=.05):
        super().__init__(shape, library_path)
        self.close()
        self.lib.fftw_set_timelimit.argtypes = [ctypes.c_double]
        self.lib.fftw_set_timelimit.restype = None
        dimensions = (ctypes.c_int * 3)(*shape[2:])
        source = torch.empty(shape, dtype=torch.float64)
        target = torch.empty_like(source)
        self.alignments = (self.lib.fftw_alignment_of(source.data_ptr()), self.lib.fftw_alignment_of(target.data_ptr()))
        tick = time.perf_counter()
        self.lib.fftw_set_timelimit(seconds_per_plan)
        try:
            for kind in (5, 4):
                kinds = (ctypes.c_int * 3)(kind, kind, kind)
                plan = self.lib.fftw_plan_many_r2r(3, dimensions, shape[0] * shape[1], source.data_ptr(), None,
                    1, self.volume, target.data_ptr(), None, 1, self.volume, kinds, 0)
                if not plan:
                    raise RuntimeError('Bounded FFTW_MEASURE plan was refused')
                self.plans.append(plan)
        except BaseException:
            self.close()
            raise
        finally:
            self.lib.fftw_set_timelimit(-1.)
        self.measured_planning_seconds = time.perf_counter() - tick
