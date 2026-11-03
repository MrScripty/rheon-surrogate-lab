"""Deploy only exact managed-CUDA-trained, managed-CPU-restored sampler heads."""
import hashlib
from profile_frozen_components import ROOT,read,digest
from recurrent_sampler_constructor import construct
from recurrent_sampler_integer_v2 import library
import recurrent_sampler_native as native
import numpy as np
import torch


def load_model(pair,library_path):
    pair=ROOT/pair;receipt=read(pair/'receipt.json');restored=read(pair/'restore/receipt.json')
    if receipt['status']!='completed'or receipt['optimizer_updates']!=440 or not all(receipt['artifact_parity'].values()):
        raise ValueError('Completed registered managed CUDA fit and exact CPU restore required')
    if restored['optimizer_updates']!=0 or restored['execution']['cuda_initialized']or not restored['full_cpu_probe_bitwise_verified']:
        raise ValueError('Actual fresh zero-update typed CPU restore required')
    for key in ('training_run_id','restore_run_id'):
        terminal=read(pair/('terminal-'+receipt[key]+'.json'))
        if terminal['state']!='completed'or terminal['custody_phase']!='drained':raise ValueError('Both managed owners must drain')
    archive=pair/'restore/weights.npz'
    if digest(archive)!=restored['outputs']['weights.npz']['sha256']:raise ValueError('Restored learned head archive changed')
    native.library=library;model=construct(library_path)
    try:
        with np.load(archive,allow_pickle=False)as weights,torch.no_grad():
            if set(weights.files)!={'mid_gate','mix'}:raise ValueError('Exact 48-parameter inventory required')
            for name in weights.files:
                value=weights[name];spec=restored['final_inventory'][name];parameter=getattr(model,name)
                if value.dtype.str!='<f8'or list(value.shape)!=spec['shape']or value.shape!=tuple(parameter.shape)or hashlib.sha256(value.tobytes()).hexdigest()!=spec['sha256']:
                    raise ValueError('Exact trained double coefficient identity changed')
                parameter.copy_(torch.from_numpy(value.copy()))
        model.eval().requires_grad_(False)
        model.frozen_versions={name:p._version for name,p in model.named_parameters()}
        model.initialization='managed_cuda_trained_exact_cpu_restored'
        model.asset_identity.update(transport_pair=str(pair.relative_to(ROOT)),transport_pair_receipt_sha256=digest(pair/'receipt.json'),
            transport_weights_sha256=digest(archive),transport_training_updates=440)
        return model
    except BaseException:
        model.close();raise
