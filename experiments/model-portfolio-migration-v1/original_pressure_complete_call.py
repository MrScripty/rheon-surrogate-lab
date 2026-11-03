"""Same complete CPU-array/CUDA/owned-CPU boundary as the127named pressure fits."""
import hashlib
import math

FIELDS = ('base','native_input_residual','cell_size_m','rho','dt')


def make_complete(state):
    torch,np = state.torch,state.np
    def complete(native):
        if set(native) != set(FIELDS):
            raise ValueError('Exactly five native input fields required; no reference labels')
        packed = {}
        for name in FIELDS[:2]:
            array = np.asarray(native[name])
            if array.dtype != np.dtype('<f8') or array.ndim != 3 or array.shape != native['base'].shape or not np.isfinite(array).all():
                raise ValueError('Finite F64 native pressure arrays required')
            packed[name] = torch.from_numpy(np.ascontiguousarray(array))[None,None]
        for name in FIELDS[2:]:
            value = float(native[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Positive finite native scalar required')
            packed[name] = torch.tensor([value],dtype=torch.float64)
        device = {name:value.to('cuda') for name,value in packed.items()}
        prediction = state.model.predict_pressure(state.network,device,state.normalization)
        if prediction.device.type != 'cuda' or prediction.dtype != torch.float64 or tuple(prediction.shape) != (1,1,*native['base'].shape):
            raise ValueError('Captured complete CUDA F64 pressure output differs')
        owned = prediction.detach().to('cpu',copy=True).numpy()[0,0]
        torch.cuda.synchronize()
        return owned
    return complete


def check_output(np,value,expected,native):
    if value.dtype != np.dtype('<f8') or value.shape != native['base'].shape or not np.isfinite(value).all() \
            or hashlib.sha256(value.tobytes()).hexdigest() != expected \
            or any(np.shares_memory(value,native[name]) for name in FIELDS[:2]):
        raise ValueError('Owned pressure output differs from independently retained accuracy')
