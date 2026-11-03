"""Fresh CPU loader for honestly labeled local assemblies of managed checkpoints."""
from pathlib import Path
import sys
from types import SimpleNamespace
from study_support import read,sha

_CACHE=None


def load(root,entry,identifier):
    global _CACHE
    if _CACHE is not None:
        if _CACHE.identifier!=identifier:raise ValueError('One checkpoint per numerical process required')
        return _CACHE
    for name,digest in entry['identities'].items():
        if sha(root/name)!=digest:raise ValueError('Routed inference artifact changed')
    project=root/entry['project'];artifacts=root/entry['restore'];witness=read(project/'source-witness.json')
    for name,digest in witness['sources'].items():
        if sha(project/name)!=digest:raise ValueError('Routed inference source changed')
    if 'model'in sys.modules:raise ValueError('A different model namespace was already imported')
    sys.path.insert(0,str(project));import torch,model,data
    model.configure('cpu');torch.set_num_threads(1);config=read(project/'config.json');model.select(config)
    receipt=read(artifacts/'assembly-receipt.json')
    if receipt['mode']!='local_weight_assembly'or receipt['optimizer_updates']!=0 or receipt['managed_training_run_claimed']:
        raise ValueError('Local assembly must not masquerade as a training Run')
    if receipt['architecture']!=model.ARCHITECTURE or witness['architecture']!=model.ARCHITECTURE or receipt['source_witness_sha256']!=sha(project/'source-witness.json'):
        raise ValueError('Routed model architecture/source closure differs')
    for name in ('weights.npz','normalization.json'):
        if sha(artifacts/name)!=receipt['outputs'][name]['sha256']:raise ValueError('Assembled artifact bytes differ')
    network=model.PressureCorrectionNet().eval();data.helpers().load_weights(network,artifacts/'weights.npz',receipt['inventory'])
    if sum(p.numel()for p in network.parameters())!=94956 or torch.cuda.is_initialized()or any(p.dtype!=torch.float32 or p.device.type!='cpu'for p in network.parameters()):
        raise ValueError('Expected CPU float32 shared-CNN/two-head model')
    _CACHE=SimpleNamespace(identifier=identifier,torch=torch,np=data.np,model=model,data=data,network=network,
        normalization=read(artifacts/'normalization.json'),config=config,entry=entry)
    return _CACHE
