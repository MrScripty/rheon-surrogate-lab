"""CPU-only frozen-weight native smoother; no labels and no reused output buffer."""
import ctypes
from v6_support import ROOT,HERE,BASE,read,sha
import torch
class NativeLocal:
    def __init__(self,network,normalization,streaming=False):
        import historical_physical
        build=BASE/('local-native-stream-build-01'if streaming else'local-native-build-01');receipt=read(build/'build.json')
        if receipt['source_sha256']!=sha(ROOT/receipt['source'])or receipt['binary_sha256']!=sha(build/'local_native.so'):raise ValueError('Native kernel identity differs')
        if sum(p.numel()for p in network.parameters())!=648:raise ValueError('Native implementation is specific to the 648-parameter architecture')
        self.weights=[layer.weight.detach().cpu().contiguous().numpy().copy()for layer in network.layers]
        if [tuple(w.shape)for w in self.weights]!=[(4,1,3,3,3),(4,4,3,3,3),(1,4,3,3,3)]:raise ValueError('Exact captured convolution shapes required')
        self.scale=historical_physical.scales(normalization)['laplacian_rhs']
        self.lib=ctypes.CDLL(str(build/'local_native.so'))
        self.lib.rheon_local_create.restype=ctypes.c_void_p
        self.lib.rheon_local_destroy.argtypes=[ctypes.c_void_p]
        self.lib.rheon_local_apply.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_double,ctypes.c_double,*([ctypes.c_void_p]*3),*([ctypes.c_int]*3),ctypes.c_void_p]
        self.lib.rheon_local_apply.restype=ctypes.c_int
        self.handle=self.lib.rheon_local_create()
        if not self.handle:raise MemoryError('Native workspace creation failed')
    def close(self):
        if getattr(self,'handle',None):self.lib.rheon_local_destroy(self.handle);self.handle=None
    def __del__(self):self.close()
    def __call__(self,r,h,rho):
        if r.device.type!='cpu'or r.dtype!=torch.float64 or r.ndim!=5 or r.shape[:2]!=(1,1)or min(r.shape[2:])<1 or max(r.shape[2:])>256:raise ValueError('Bounded CPU f64 field required')
        if any(x.shape!=(1,)or x.dtype!=torch.float64 or x.device.type!='cpu'or not bool(torch.isfinite(x).all()and(x>0).all())for x in [h,rho]):raise ValueError('Positive CPU spacing and fluid density required')
        r=r.contiguous();out=torch.empty_like(r)
        code=self.lib.rheon_local_apply(self.handle,r.data_ptr(),float(rho[0]/h[0]),self.scale,*[w.ctypes.data for w in self.weights],*r.shape[2:],out.data_ptr())
        if code:raise ValueError('Native local inference failed with code '+str(code))
        return out
