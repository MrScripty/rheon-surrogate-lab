"""Length-N separable F64 DCT with fused axis transitions and learned response.

No physical coefficients or numerical pressure inverse are constructed. Generic
forward/inverse methods and the fused learned-response entry share the same
geometry-only kernels and independently qualified orthonormal transform.
"""
import ctypes as C
import ctypes.util
import math
import torch
from support import ROOT, read, digest
from fft_dct_v2 import FFTDCT
from cuda_kernel_runtime_v1 import check, CudaKernels


class CompactLayoutKernels:
    launch = CudaKernels.launch
    close = CudaKernels.close

    def __init__(self, directory):
        directory = ROOT / directory
        receipt = read(directory / 'receipt.json')
        if (receipt['status'] != 'compiled_source_bound_cuda_kernels'
                or receipt['fma_contraction'] or receipt['fast_math']
                or digest(directory / 'kernels.ptx') != receipt['ptx_sha256']
                or digest(ROOT / receipt['source']) != receipt['source_sha256']):
            raise ValueError('Precise source-bound compact layout build required')
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if [props.major, props.minor] != receipt['compute_capability']:
            raise ValueError('Compact layout architecture differs')
        self.driver = C.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
        self.driver.cuInit.argtypes = [C.c_uint]
        self.driver.cuModuleLoadDataEx.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint, C.c_void_p, C.c_void_p]
        self.driver.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        self.driver.cuLaunchKernel.argtypes = [C.c_void_p, *([C.c_uint] * 7), C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p]
        self.driver.cuModuleUnload.argtypes = [C.c_void_p]
        check(self.driver.cuInit(0), 'cuInit')
        context_anchor = torch.empty((), device='cuda', dtype=torch.float64)
        self.module = C.c_void_p()
        ptx = C.create_string_buffer((directory / 'kernels.ptx').read_bytes())
        check(self.driver.cuModuleLoadDataEx(C.byref(self.module), ptx, 0, None, None), 'cuModuleLoadDataEx')
        self.functions = {}
        for name in ('compact_forward_pack', 'compact_forward_transition', 'compact_inverse_transition'):
            function = C.c_void_p()
            check(self.driver.cuModuleGetFunction(C.byref(function), self.module, name.encode()), 'cuModuleGetFunction')
            self.functions[name] = function
        self.identity = dict(build=str(directory.relative_to(ROOT)), receipt_sha256=digest(directory / 'receipt.json'), **receipt)


class CompactFFTDCT(FFTDCT):
    def __init__(self, build='runs/full-ai-data-diversity-v1/compact-fft-layout-build-01'):
        super().__init__()
        self.layout = CompactLayoutKernels(build)
        self.constants = {}
        self.identity.update(transform='length_N_separable_real_FFT_even_odd_reordering_fused_axis_transitions',
            layout=self.layout.identity, learned_response_fused=True, pressure_anchor_fused=True,
            constant_origin='geometry-only phase and orthonormal scales')

    def launch(self, name, elements, tensors, integers):
        self.layout.launch(name, elements, [*[C.c_void_p(v.data_ptr()) for v in tensors],
                                           *[C.c_int(n) for n in integers]])

    def prepare(self, shape):
        shape = tuple(shape)
        if shape in self.prepared:
            return
        if len(shape) != 3 or any(n < 1 for n in shape) or torch.cuda.is_current_stream_capturing():
            raise ValueError('Prepare positive 3D geometry outside capture')
        for n in shape:
            if n not in self.constants:
                k = torch.arange(n, device='cuda', dtype=torch.float64)
                theta = k * (math.pi / (2 * n))
                cosine, sine = torch.cos(theta), torch.sin(theta)
                alpha = torch.full((n,), math.sqrt(2 / n), device='cuda', dtype=torch.float64)
                alpha[0] = 1 / math.sqrt(n)
                self.constants[n] = cosine * alpha, sine * alpha, cosine, sine, alpha.reciprocal()
        dummy = torch.zeros(shape, device='cuda', dtype=torch.float64)
        for _ in range(2):
            self.inverse(self.forward(dummy))
        torch.cuda.synchronize()
        self.prepared.add(shape)
        cache = torch.backends.cuda.cufft_plan_cache[torch.cuda.current_device()]
        self.setup_observations.append(dict(shape_zyx=list(shape), actual_cached_plans=cache.size,
            cache_max_size=cache.max_size, warmups=2,
            axis_constant_bytes=sum(v.numel() * v.element_size() for n in set(shape) for v in self.constants[n]),
            FFT_lengths=list(shape), real_workspace_elements_per_axis=math.prod(shape)))

    def check_value(self, value):
        if (value.device.type != 'cuda' or value.dtype != torch.float64 or value.ndim != 3
                or any(n not in self.constants for n in value.shape)):
            raise ValueError('Prepared complete 3D CUDA double field required')
        return value.contiguous()

    def forward(self, value, divisor=None):
        value = self.check_value(value); shape = tuple(value.shape)
        axes = [a for a, n in enumerate(shape) if n > 1]
        if not axes:
            return value.clone() if divisor is None else torch.zeros_like(value)
        if divisor is not None and (divisor.device.type != 'cuda' or divisor.dtype != torch.float64
                or tuple(divisor.shape) != shape or not divisor.is_contiguous()):
            raise ValueError('Original contiguous CUDA F64 learned response tensor required')
        first = axes[0]; volume = value.numel()
        work = torch.empty((volume // shape[first], shape[first]), device='cuda', dtype=torch.float64)
        self.launch('compact_forward_pack', volume, [value, work], [*shape, first])
        for index, axis in enumerate(axes):
            n = shape[axis]
            spectrum = torch.fft.rfft(work, dim=-1, norm='backward')
            following = axes[index + 1] if index + 1 < len(axes) else -1
            output_shape = (volume // shape[following], shape[following]) if following >= 0 else shape
            output = torch.empty(output_shape, device='cuda', dtype=torch.float64)
            fr, fi, _, _, _ = self.constants[n]
            response = divisor if divisor is not None else value
            self.launch('compact_forward_transition', volume, [spectrum, fr, fi, response, output],
                [*shape, axis, following, *spectrum.stride(), int(following < 0 and divisor is not None)])
            work = output
        return work

    def inverse(self, value, gauge=False):
        value = self.check_value(value); shape = tuple(value.shape)
        axes = [a for a in (2, 1, 0) if shape[a] > 1]
        if not axes:
            return torch.zeros_like(value) if gauge else value.clone()
        volume = value.numel(); work = value; previous = -1
        for axis in axes:
            n = shape[axis]
            spectrum = torch.empty((volume // n, n // 2 + 1), device='cuda', dtype=torch.complex128)
            _, _, cosine, sine, inverse_alpha = self.constants[n]
            strides = work.stride() if previous >= 0 else (0, 0)
            self.launch('compact_inverse_transition', spectrum.numel(), [work, cosine, sine, inverse_alpha, spectrum],
                        [*shape, previous, axis, *strides, 0])
            work = torch.fft.irfft(spectrum, n=n, dim=-1, norm='backward')
            previous = axis
        output = torch.empty(shape, device='cuda', dtype=torch.float64)
        _, _, cosine, sine, inverse_alpha = self.constants[shape[previous]]
        self.launch('compact_inverse_transition', volume, [work, cosine, sine, inverse_alpha, output],
                    [*shape, previous, -1, *work.stride(), int(gauge)])
        return output

    def pressure(self, core, value, counts):
        shape = tuple(reversed(counts))
        if shape not in self.prepared or tuple(value.shape) != (1, 1, *shape):
            raise ValueError('Prepared complete batch-one pressure field required')
        divisor = core.divisors['g' + '_'.join(map(str, shape))]
        return self.inverse(self.forward(value.reshape(shape), divisor=divisor), gauge=True).reshape_as(value)

    def close(self):
        self.layout.close()
