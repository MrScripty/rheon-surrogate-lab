"""Generic 3D even-extension DCT via one rfftn/irfftn and four layout kernels.

Only geometry-derived phase/orthonormalization constants are stored. The learned
response comes exclusively from the existing restored core.divisors tensor.
"""
import ctypes as C
import ctypes.util
import math
import torch
from support import ROOT, read, digest
from fft_dct_v2 import FFTDCT
from cuda_kernel_runtime_v1 import check, CudaKernels


class LayoutKernels:
    launch = CudaKernels.launch
    close = CudaKernels.close

    def __init__(self, directory):
        directory = ROOT / directory
        receipt = read(directory / 'receipt.json')
        if (receipt['status'] != 'compiled_source_bound_cuda_kernels'
                or receipt['fma_contraction'] or receipt['fast_math']
                or digest(directory / 'kernels.ptx') != receipt['ptx_sha256']
                or digest(ROOT / receipt['source']) != receipt['source_sha256']):
            raise ValueError('Precise source-bound generic layout build required')
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if [props.major, props.minor] != receipt['compute_capability']:
            raise ValueError('Layout build architecture differs')
        self.driver = C.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
        self.driver.cuInit.argtypes = [C.c_uint]
        self.driver.cuModuleLoadDataEx.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint, C.c_void_p, C.c_void_p]
        self.driver.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        self.driver.cuLaunchKernel.argtypes = [C.c_void_p, *([C.c_uint] * 7), C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p]
        self.driver.cuModuleUnload.argtypes = [C.c_void_p]
        check(self.driver.cuInit(0), 'cuInit')
        self.module = C.c_void_p()
        ptx = C.create_string_buffer((directory / 'kernels.ptx').read_bytes())
        check(self.driver.cuModuleLoadDataEx(C.byref(self.module), ptx, 0, None, None), 'cuModuleLoadDataEx')
        self.functions = {}
        for name in ('dct_pack_even', 'dct_unpack_forward', 'dct_pack_inverse', 'dct_unpack_inverse'):
            function = C.c_void_p()
            check(self.driver.cuModuleGetFunction(C.byref(function), self.module, name.encode()), 'cuModuleGetFunction')
            self.functions[name] = function
        self.identity = dict(build=str(directory.relative_to(ROOT)), receipt_sha256=digest(directory / 'receipt.json'), **receipt)


class FFT3DDCT(FFTDCT):
    def __init__(self, build='runs/full-ai-data-diversity-v1/fft3d-layout-build-01'):
        super().__init__()
        self.layout = LayoutKernels(build)
        self.constants = {}
        self.identity.update(transform='3D_even_extension_rfftn_irfftn_with_four_fused_layouts',
                             layout=self.layout.identity, constant_origin='geometry-only phases and orthonormal scales')

    def prepare(self, shape):
        shape = tuple(shape)
        if shape in self.prepared:
            return
        if len(shape) != 3 or any(n < 1 for n in shape) or torch.cuda.is_current_stream_capturing():
            raise ValueError('Prepare positive 3D geometry outside capture')
        # Forward spectrum: positive frequencies in all dimensions.
        phase = torch.zeros(shape, device='cuda', dtype=torch.float64)
        scale = torch.ones(shape, device='cuda', dtype=torch.float64)
        for axis, n in enumerate(shape):
            k = torch.arange(n, device='cuda', dtype=torch.float64)
            view = [1] * 3; view[axis] = n
            phase.add_((k * (-math.pi / (2 * n))).reshape(view))
            s = torch.full((n,), math.sqrt(2 / n) / 2, device='cuda', dtype=torch.float64)
            s[0] = 1 / (2 * math.sqrt(n))
            scale.mul_(s.reshape(view))
        forward_real = (torch.cos(phase) * scale).contiguous()
        forward_imag = (torch.sin(phase) * scale).contiguous()
        # Inverse: signed z/y frequencies, nonnegative x; Nyquist planes zeroed
        # by the layout kernel. Reflect coefficients, never physical responses.
        nz, ny, nx = shape
        spectrum_shape = (2 * nz, 2 * ny, nx + 1)
        phase = torch.zeros(spectrum_shape, device='cuda', dtype=torch.float64)
        scale = torch.ones(spectrum_shape, device='cuda', dtype=torch.float64)
        for axis, n in enumerate(shape):
            length = spectrum_shape[axis]
            k = torch.arange(length, device='cuda', dtype=torch.float64)
            if axis != 2:
                k = torch.where(k <= n, k, k - 2 * n)
            view = [1] * 3; view[axis] = length
            phase.add_((k * (math.pi / (2 * n))).reshape(view))
            s = torch.full((length,), math.sqrt(2 * n), device='cuda', dtype=torch.float64)
            s[0] = 2 * math.sqrt(n)
            scale.mul_(s.reshape(view))
        inverse_real = (torch.cos(phase) * scale).contiguous()
        inverse_imag = (torch.sin(phase) * scale).contiguous()
        self.constants[shape] = forward_real, forward_imag, inverse_real, inverse_imag
        dummy = torch.zeros(shape, device='cuda', dtype=torch.float64)
        for _ in range(2):
            self.inverse(self.forward(dummy))
        torch.cuda.synchronize()
        self.prepared.add(shape)
        cache = torch.backends.cuda.cufft_plan_cache[torch.cuda.current_device()]
        self.setup_observations.append(dict(shape_zyx=list(shape), actual_cached_plans=cache.size,
            cache_max_size=cache.max_size, warmups=2,
            geometry_constant_bytes=sum(v.numel() * v.element_size() for v in self.constants[shape])))

    def check_value(self, value):
        if (value.device.type != 'cuda' or value.dtype != torch.float64 or value.ndim != 3
                or tuple(value.shape) not in self.constants):
            raise ValueError('Prepared complete 3D CUDA double field required')
        return value.contiguous()

    def launch(self, name, elements, tensors, shape):
        self.layout.launch(name, elements, [*[C.c_void_p(v.data_ptr()) for v in tensors],
                                           *[C.c_int(n) for n in shape]])

    def forward(self, value):
        value = self.check_value(value); shape = tuple(value.shape)
        extended = torch.empty(tuple(2 * n for n in shape), device='cuda', dtype=torch.float64)
        self.launch('dct_pack_even', extended.numel(), [value, extended], shape)
        spectrum = torch.fft.rfftn(extended, dim=(0, 1, 2), norm='backward')
        output = torch.empty_like(value)
        fr, fi, _, _ = self.constants[shape]
        self.launch('dct_unpack_forward', output.numel(), [spectrum, fr, fi, output], shape)
        return output

    def inverse(self, value):
        value = self.check_value(value); shape = tuple(value.shape)
        nz, ny, nx = shape
        spectrum = torch.empty((2 * nz, 2 * ny, nx + 1), device='cuda', dtype=torch.complex128)
        _, _, ir, ii = self.constants[shape]
        self.launch('dct_pack_inverse', spectrum.numel(), [value, ir, ii, spectrum], shape)
        extended = torch.fft.irfftn(spectrum, s=tuple(2 * n for n in shape), dim=(0, 1, 2), norm='backward')
        output = torch.empty_like(value)
        self.launch('dct_unpack_inverse', output.numel(), [extended, output], shape)
        return output

    def close(self):
        self.layout.close()
