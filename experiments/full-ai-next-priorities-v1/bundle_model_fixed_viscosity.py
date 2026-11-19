"""Fresh full-state family with the exact managed viscosity calibration frozen."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from bundle_model_displacement import BundleNet as DisplacementBundleNet


class BundleNet(DisplacementBundleNet):
    def __init__(self,*args,asset_directory,**kwargs):
        super().__init__(*args,asset_directory=asset_directory,**kwargs)
        assets=Path(asset_directory);meta=json.loads((assets/'viscosity-initialization.json').read_text())
        path=assets/'viscosity-weights.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['weights_sha256']or not meta['native_qualified_float32']:
            raise ValueError('Exact managed native-qualified viscosity coefficients required')
        with np.load(path,allow_pickle=False)as arrays:
            if arrays.files!=['projection.weight']or arrays['projection.weight'].shape!=(3,3,3,3,3)or arrays['projection.weight'].dtype!=np.float32:
                raise ValueError('Managed243-coefficient viscosity tensor required')
            with torch.no_grad():self.viscous_head.weight.copy_(torch.from_numpy(arrays['projection.weight'].copy()))
        self.viscous_head.weight.requires_grad_(False)
