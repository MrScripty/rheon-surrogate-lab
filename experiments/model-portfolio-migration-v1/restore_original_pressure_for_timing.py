"""Original train checkpoint restoration with explicit observer ownership.

The qualified accuracy loader remains unchanged. This timing adapter shares its
metadata contract and exposes hook lifecycle rather than patching Torch globals.
"""
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from original_train_pressure_cuda import ROOT,verify_entry,require
from cuda_forward_observer import CUDAForwardObserver


@contextmanager
def restore(material,fit_id):
    require(os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda','Explicit CUDA timing environment required')
    require('model' not in sys.modules and 'data' not in sys.modules,'One original checkpoint per fresh timing process required')
    verified = verify_entry(material,fit_id)
    import torch
    require(torch.cuda.is_available(),'CUDA required; CPU neural fallback forbidden')
    observer = CUDAForwardObserver(torch)
    observer.enable('construction')
    try:
        project = ROOT/verified.entry['project']
        sys.path.insert(0,str(project))
        with torch.device('cuda'):
            import model
            import data
            require(Path(model.__file__).resolve() == project/'model.py' and Path(data.__file__).resolve() == project/'data.py',
                    'Captured timing model/data ownership differs')
            model.configure('cuda')
            if verified.config.get('conditioning') is not None:
                model.select(verified.config)
            require(model.ARCHITECTURE == verified.receipt['architecture'],'Captured timing architecture/selection differs')
            network = model.PressureCorrectionNet().to('cuda').eval()
            data.helpers().load_weights(network,ROOT/verified.entry['outputs']['weights.npz']['path'],verified.inventory)
        require(all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in network.parameters())
                and all(b.device.type == 'cuda' for b in network.buffers()),'All original F32 parameters and buffers must remain CUDA')
        require(data.helpers().state_inventory(network) == verified.inventory,'Original final timing tensors differ')
        architecture = copy.deepcopy(model.ARCHITECTURE)
        state = SimpleNamespace(identifier=fit_id,entry=verified.entry,torch=torch,np=data.np,model=model,data=data,network=network,
            normalization=verified.normalization,config=verified.config,inventory=verified.inventory,observer=observer)
        observer.set_phase('setup')
        with torch.inference_mode():
            yield state
        require(observer.handle is not None and observer.forbidden_attempts == 0,'Timing must restore the CUDA observer and have no forbidden attempts')
        require(data.helpers().state_inventory(network) == verified.inventory and model.ARCHITECTURE == architecture,
                'Timing changed original learned tensors or architecture')
        verify_entry(material,fit_id)
    finally:
        observer.close()
