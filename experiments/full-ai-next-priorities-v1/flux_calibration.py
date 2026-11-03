"""Offline SGD calibration of the linear neural layer on native-qualified labels."""
import hashlib
import json
import time
import numpy as np
import torch
from torch import nn


def calibrate(network,archive,device,updates=1000):
    with np.load(archive,allow_pickle=False)as arrays:
        if arrays['velocity'].shape!=(80,3,13,13,13)or arrays['velocity'].dtype!=np.float32 or arrays['target'].shape!=(80,1,12,12,12)or arrays['target'].dtype!=np.float64:
            raise ValueError('Native-qualified synthetic flux archive contract differs')
        velocity=torch.from_numpy(arrays['velocity'].copy()).double().to(device)
        target=torch.from_numpy(arrays['target'].copy()).to(device)
    layer=nn.Conv3d(3,1,3,padding=1,padding_mode='replicate',bias=False).double().to(device)
    with torch.no_grad():layer.weight.zero_()
    optimizer=torch.optim.SGD(layer.parameters(),lr=.05);history=[];tick=time.perf_counter()
    for update in range(1,updates+1):
        optimizer.zero_grad(set_to_none=True)
        predicted=layer(velocity[:64])[:,:,:-1,:-1,:-1]
        objective=(predicted-target[:64]).square().mean();objective.backward()
        if not bool(torch.isfinite(objective))or not bool(torch.isfinite(layer.weight.grad).all()):
            raise ValueError('Offline neural flux calibration gradient failed')
        optimizer.step()
        if update==1 or update%100==0 or update==updates:
            history.append(dict(update=update,loss=float(objective.detach())))
            print(json.dumps(dict(format='tuldok_progress_v1',stage='neural-linear-flux-calibration',
                completed=update,total=updates,metrics=dict(loss=float(objective.detach())))),flush=True)
    with torch.no_grad():
        predicted=layer(velocity[64:])[:,:,:-1,:-1,:-1]
        errors=(predicted-target[64:]).flatten(1).norm(dim=1)/target[64:].flatten(1).norm(dim=1)
        network.flux.weight.copy_(layer.weight.float())
        network.flux.weight.requires_grad_(False)
    if updates==1000 and float(errors.max())>1e-10:raise ValueError('Learned flux layer fails held-out synthetic accuracy')
    return dict(updates=updates,optimizer='SGDfloat64',learning_rate=.05,batch_size=64,train_patterns=64,
        development_patterns=16,worst_development_relative_error=float(errors.max()),history=history,
        seconds=time.perf_counter()-tick,learned_weight_sha256=hashlib.sha256(network.flux.weight.detach().cpu().numpy().tobytes()).hexdigest(),
        pressure_solves=0,temporal_scenes_used=False)
