"""Restore captured pressure training outputs without inventing a CPU restore.

Metadata checks import only stdlib. CUDA execution is a separate, process-owned
context; callers must finish native publication before qualifying this adapter.
"""
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def require(condition,message):
    if not condition:
        raise ValueError(message)


def bound_bytes(spec,root):
    relative = Path(spec['path'])
    require(not relative.is_absolute() and '..' not in relative.parts,'Material paths must stay within their bound root')
    path = root/relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),'Ordinary bound material file required')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == spec['sha256']
            and ('bytes' not in spec or len(raw) == spec['bytes']),'Original pressure material bytes changed')
    return raw


def verify_entry(material,fit_id,*,root=ROOT):
    """Return exact original metadata/inventory; do not deserialize an array."""
    require(material['root'] == str(root),'Original material root differs')
    entries = [r for r in material['models'] if r['fit_item_id'] == fit_id]
    require(len(entries) == 1 and entries[0]['checkpoint_origin'] == 'original_train_output',
            'Exactly one original training-output fit required')
    entry = entries[0]
    receipt = json.loads(bound_bytes(entry['original_receipt'],root))
    config = json.loads(bound_bytes(entry['original_configuration'],root))
    witness = json.loads(bound_bytes(entry['source_witness'],root))
    require(receipt['schema'] == 'rheon-water-learning-training-v1' and receipt['mode'] == 'train'
            and receipt['status'] == 'complete' and receipt['optimizer_updates'] == entry['original_optimizer_updates'] > 0,
            'Original completed training receipt required; no restore substitution')
    project = Path(entry['project'])
    artifacts = Path(entry['artifacts'])
    require(Path(entry['original_configuration']['path']) == project/'config.json'
            and Path(entry['source_witness']['path']) == project/'source-witness.json'
            and Path(entry['original_receipt']['path']) == artifacts/'receipt.json','Original project/output ownership differs')
    closure = receipt['closure']
    hashes = [closure[k] for k in ('experiment_sha256','config_sha256') if k in closure]
    require(hashes and all(h == entry['original_configuration']['sha256'] for h in hashes)
            and closure['source_witness_sha256'] == entry['source_witness']['sha256']
            and receipt['architecture'] == entry['captured_architecture'],'Original training closure/architecture differs')
    expected = {str(project/name):value for name,value in witness['sources'].items()}
    require({r['path']:r['sha256'] for r in entry['sources']} == expected
            and len(entry['sources']) == len(expected),'Complete captured source witness required')
    for spec in entry['sources']:
        bound_bytes(spec,root)
    outputs = {}
    for name in ('weights.npz','normalization.json','history.json'):
        spec = entry['outputs'][name]
        require(Path(spec['path']) == artifacts/name and spec['sha256'] == receipt['outputs'][name]['sha256']
                and spec['bytes'] == receipt['outputs'][name]['bytes'],'Original training output binding differs')
        outputs[name] = bound_bytes(spec,root)
    require(entry['outputs']['normalization.json']['sha256'] == closure['normalization_sha256'],
            'Original training normalization differs')
    history = json.loads(outputs['history.json'])
    selected = [c for c in history['checkpoints'].values() if c['sha256'] == entry['outputs']['weights.npz']['sha256']]
    require(len(selected) == 1 and selected[0]['update'] == receipt['optimizer_updates']
            and isinstance(selected[0]['inventory'],dict) and selected[0]['inventory'],
            'Selected original final tensor inventory is unavailable or ambiguous')
    return SimpleNamespace(entry=entry,receipt=receipt,config=config,inventory=selected[0]['inventory'],
        normalization=json.loads(outputs['normalization.json']))


@contextmanager
def restore(material,fit_id):
    """Yield original model/state with every neural forward guarded on CUDA."""
    require(os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda','Explicit CUDA environment required; CPU fallback forbidden')
    require('model' not in sys.modules and 'data' not in sys.modules,'One captured checkpoint per fresh process required')
    verified = verify_entry(material,fit_id)
    import torch
    require(torch.cuda.is_available(),'CUDA is unavailable; no CPU neural fallback')
    observed = dict(phase='construction',CPU_neural_forwards=0,forbidden_forward_attempts=0,scoped_guard_removed=False,
        original_receipt_mode='train',restore_receipt_fabricated=False)
    calls = Counter()

    def guard(module,args):
        tensors = []
        def collect(value):
            if isinstance(value,torch.Tensor):
                tensors.append(value)
            elif isinstance(value,(tuple,list)):
                for item in value:
                    collect(item)
            elif isinstance(value,dict):
                for item in value.values():
                    collect(item)
        collect(args)
        local = [*module.parameters(recurse=False),*module.buffers(recurse=False)]
        if not tensors or any(t.device.type != 'cuda' for t in [*tensors,*local]):
            observed['forbidden_forward_attempts'] += 1
            raise ValueError('CPU/unclassified neural forward blocked before execution')
        calls[(observed['phase'],type(module).__name__)] += 1

    handle = torch.nn.modules.module.register_module_forward_pre_hook(guard)
    try:
        project = ROOT/verified.entry['project']
        sys.path.insert(0,str(project))
        with torch.device('cuda'):
            import model
            import data
            require(Path(model.__file__).resolve() == project/'model.py' and Path(data.__file__).resolve() == project/'data.py',
                    'Captured model/data import ownership differs')
            model.configure('cuda')
            if verified.config.get('conditioning') is not None:
                model.select(verified.config)
            require(model.ARCHITECTURE == verified.receipt['architecture'],'Captured inference architecture/selection differs')
            network = model.PressureCorrectionNet().to('cuda').eval()
            data.helpers().load_weights(network,ROOT/verified.entry['outputs']['weights.npz']['path'],verified.inventory)
        require(all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in network.parameters())
                and all(b.device.type == 'cuda' for b in network.buffers()),'Original F32 parameters and all buffers must be CUDA')
        require(data.helpers().state_inventory(network) == verified.inventory,'Restored original learned tensor inventory differs')
        architecture = copy.deepcopy(model.ARCHITECTURE)
        state = SimpleNamespace(identifier=fit_id,entry=verified.entry,torch=torch,np=data.np,model=model,data=data,network=network,
            normalization=verified.normalization,config=verified.config,inventory=verified.inventory,device_observation=observed)
        observed['phase'] = 'prediction'
        with torch.inference_mode():
            yield state
        require(data.helpers().state_inventory(network) == verified.inventory and model.ARCHITECTURE == architecture,
                'Inference changed original tensors or captured architecture')
        verify_entry(material,fit_id)
    finally:
        handle.remove()
        observed['scoped_guard_removed'] = True
        observed['neural_calls_by_phase'] = {phase:{name:count for (p,name),count in calls.items() if p == phase}
            for phase in {p for p,_ in calls}}
