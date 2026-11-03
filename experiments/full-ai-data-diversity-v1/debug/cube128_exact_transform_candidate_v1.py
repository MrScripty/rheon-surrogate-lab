"""DEBUG candidate: generic fixed GPU transforms with learned response only.

No native pressure, numerical response/eigenvalues or teacher request enters
this candidate. Every physical stage and learned response stays on CUDA.
"""
import ctypes as C
import types
import torch
from complete_cuda_graph_v3 import CompleteCudaGraphGeneratorV3
from cuda_kernel_runtime_v1 import CudaKernels, check


class FixedAxisTransform:
    def __init__(self, build):
        self.kernels = CudaKernels(build)
        function = C.c_void_p()
        check(self.kernels.driver.cuModuleGetFunction(C.byref(function), self.kernels.module,
                                                     b'fixed_axis_transform'), 'fixed transform entry')
        self.kernels.functions['fixed_axis_transform'] = function

    def contract(self, value, basis, axis, inverse):
        if value.device.type != 'cuda' or value.dtype != torch.float64 or value.ndim != 3:
            raise ValueError('One complete CUDA double fixed-transform field required')
        value = value.contiguous()
        shape = tuple(value.shape)
        if basis.device != value.device or basis.dtype != value.dtype or not basis.is_contiguous():
            raise ValueError('Prepared fixed CUDA basis required')
        if tuple(basis.shape) != (shape[2 - axis], shape[2 - axis]):
            raise ValueError('Fixed basis and transform dimension disagree')
        output = torch.empty_like(value)
        nx, ny, nz = reversed(shape)
        ptr = lambda v: C.c_void_p(v.data_ptr())
        parameters = [ptr(value), ptr(basis), ptr(output), C.c_int(nx), C.c_int(ny), C.c_int(nz),
                      C.c_int(axis), C.c_int(int(inverse)), C.c_int(-1), C.c_int(1)]
        self.kernels.launch('fixed_axis_transform', value.numel(), parameters)
        return output

    def pressure(self, core, q, counts):
        shape = tuple(reversed(counts))
        if tuple(q.shape) != (1, 1, *shape):
            raise ValueError('Batch-one complete physical pressure representation required')
        axes = core.geometry_cache[tuple(counts)][1]
        value = q.reshape(shape)
        for axis in (2, 1, 0):
            value = self.contract(value, axes[2 - axis], axis, False)
        # This is the same restored learned response as the existing candidate.
        # No numerical response or physical solver coefficient is constructed.
        value = value / core.divisors['g' + '_'.join(map(str, shape))]
        value[0, 0, 0] = 0.
        for axis in (0, 1, 2):
            value = self.contract(value, axes[2 - axis], axis, True)
        value = value.reshape_as(q)
        return value - value[:, :, :1, :1, :1]

    def close(self):
        self.kernels.close()


class Cube128ExactTransformCandidate(CompleteCudaGraphGeneratorV3):
    def __init__(self, pair, shared_pair, build, control_build):
        super().__init__(pair, shared_pair, build, control_build)
        self.fixed_transform = FixedAxisTransform(
            'runs/full-ai-data-diversity-v1/debug/cube128-equation-failure-01/dct-debug-build-02')
        def pressure(core, q, counts):
            return self.fixed_transform.pressure(core, q, counts)
        self.model.pressure = types.MethodType(pressure, self.model)
        self.asset_identity.update(
            fixed_transform_execution='Generic GPU FMA contractions with independent layout-tail accumulation',
            fixed_transform_build=self.fixed_transform.kernels.identity,
            learned_response_and_all_coefficients_unchanged=True,
            native_numerical_oracle_in_candidate=False)

    def close(self):
        try:
            super().close()
        finally:
            self.fixed_transform.close()
