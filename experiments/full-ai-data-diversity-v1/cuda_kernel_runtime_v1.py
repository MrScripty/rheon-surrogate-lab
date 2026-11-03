"""Pinned NVRTC compilation and CUDA-driver launch of generic learned kernels."""
import ctypes as C
import ctypes.util
import hashlib
import json
from pathlib import Path
import torch
from support import ROOT,read,digest,write


def check(status,label):
    if status!=0:raise RuntimeError(label+' failed with CUDA/NVRTC status '+str(status))


def compile_kernels(protocol,output):
    if output.exists():raise ValueError('Fresh source-bound CUDA kernel build required')
    reg=read(protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Frozen CUDA build source changed: '+path)
    for row in reg['compiler_libraries']:
        if digest(row['path'])!=row['sha256']:raise ValueError('Qualified NVRTC library changed')
    library=next(row['path'] for row in reg['compiler_libraries'] if Path(row['path']).name=='libnvrtc.so.13')
    lib=C.CDLL(library);pointer=C.c_void_p;charp=C.c_char_p
    lib.nvrtcCreateProgram.argtypes=[C.POINTER(pointer),charp,charp,C.c_int,C.POINTER(charp),C.POINTER(charp)]
    lib.nvrtcCompileProgram.argtypes=[pointer,C.c_int,C.POINTER(charp)]
    for name in ('nvrtcGetProgramLogSize','nvrtcGetPTXSize'):getattr(lib,name).argtypes=[pointer,C.POINTER(C.c_size_t)]
    for name in ('nvrtcGetProgramLog','nvrtcGetPTX'):getattr(lib,name).argtypes=[pointer,pointer]
    lib.nvrtcDestroyProgram.argtypes=[C.POINTER(pointer)]
    lib.nvrtcVersion.argtypes=[C.POINTER(C.c_int),C.POINTER(C.c_int)]
    torch.cuda.init();device=torch.cuda.get_device_properties(torch.cuda.current_device())
    arch=str(device.major)+str(device.minor)
    options=[*reg['compile_options'],'--gpu-architecture=compute_'+arch]
    source=ROOT/reg['kernel_source'];program=pointer();major=C.c_int();minor=C.c_int()
    check(lib.nvrtcVersion(C.byref(major),C.byref(minor)),'nvrtcVersion')
    check(lib.nvrtcCreateProgram(C.byref(program),source.read_bytes(),source.name.encode(),0,None,None),'nvrtcCreateProgram')
    output.mkdir(parents=True)
    try:
        encoded=(charp*len(options))(*(option.encode() for option in options))
        status=lib.nvrtcCompileProgram(program,len(options),encoded)
        size=C.c_size_t();check(lib.nvrtcGetProgramLogSize(program,C.byref(size)),'nvrtcGetProgramLogSize')
        log=C.create_string_buffer(size.value);check(lib.nvrtcGetProgramLog(program,log),'nvrtcGetProgramLog')
        (output/'compiler.log').write_bytes(log.value)
        check(status,'nvrtcCompileProgram')
        check(lib.nvrtcGetPTXSize(program,C.byref(size)),'nvrtcGetPTXSize')
        if size.value>4*1024**2:raise ValueError('Finite compiled-kernel artifact cap exceeded')
        ptx=C.create_string_buffer(size.value);check(lib.nvrtcGetPTX(program,ptx),'nvrtcGetPTX')
        (output/'kernels.ptx').write_bytes(ptx.raw);(output/'source.cu').write_bytes(source.read_bytes())
        for path,expected in reg['source_sha256'].items():
            if digest(ROOT/path)!=expected:raise ValueError('CUDA build source changed during compilation: '+path)
        write(output/'receipt.json',dict(status='compiled_source_bound_cuda_kernels',source=reg['kernel_source'],source_sha256=digest(source),
            ptx_sha256=digest(output/'kernels.ptx'),compiler_libraries=reg['compiler_libraries'],compiler_nvrtc_version=[major.value,minor.value],
            options=options,compute_capability=[device.major,device.minor],gpu_model=device.name,
            torch_version=torch.__version__,cuda_version=torch.version.cuda,protocol_sha256=digest(protocol),
            optimizer_updates=0,model_forward_executed=False,fma_contraction=False,fast_math=False))
        print('Compiled generic learned CUDA kernels withNVRTC',major.value,minor.value,'compute',arch,flush=True)
    finally:check(lib.nvrtcDestroyProgram(C.byref(program)),'nvrtcDestroyProgram')


class CudaKernels:
    def __init__(self,directory):
        directory=ROOT/directory;receipt=read(directory/'receipt.json')
        if (receipt['status']!='compiled_source_bound_cuda_kernels' or receipt['fma_contraction'] or receipt['fast_math']
                or digest(directory/'kernels.ptx')!=receipt['ptx_sha256'] or digest(ROOT/receipt['source'])!=receipt['source_sha256']):
            raise ValueError('Exact precise source-bound CUDA kernel build required')
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        if [props.major,props.minor]!=receipt['compute_capability']:raise ValueError('Compiled CUDA architecture differs')
        self.driver=C.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
        self.driver.cuInit.argtypes=[C.c_uint]
        self.driver.cuModuleLoadDataEx.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_uint,C.c_void_p,C.c_void_p]
        self.driver.cuModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p]
        self.driver.cuLaunchKernel.argtypes=[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p]
        self.driver.cuModuleUnload.argtypes=[C.c_void_p]
        check(self.driver.cuInit(0),'cuInit');self.module=C.c_void_p()
        ptx=C.create_string_buffer((directory/'kernels.ptx').read_bytes())
        check(self.driver.cuModuleLoadDataEx(C.byref(self.module),ptx,0,None,None),'cuModuleLoadDataEx')
        self.functions={}
        for name in ('learned_transport','learned_linear'):
            function=C.c_void_p();check(self.driver.cuModuleGetFunction(C.byref(function),self.module,name.encode()),'cuModuleGetFunction')
            self.functions[name]=function
        self.identity=dict(build=str(directory.relative_to(ROOT)),receipt_sha256=digest(directory/'receipt.json'),**receipt)

    def launch(self,name,elements,arguments):
        pointers=(C.c_void_p*len(arguments))(*(C.addressof(value) for value in arguments))
        stream=C.c_void_p(torch.cuda.current_stream().cuda_stream)
        check(self.driver.cuLaunchKernel(self.functions[name],(elements+127)//128,1,1,128,1,1,0,stream,pointers,None),'cuLaunchKernel '+name)

    def close(self):
        if self.module is not None:
            torch.cuda.synchronize();check(self.driver.cuModuleUnload(self.module),'cuModuleUnload');self.module=None
