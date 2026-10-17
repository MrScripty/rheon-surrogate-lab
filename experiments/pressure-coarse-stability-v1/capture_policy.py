"""Exact allowed clipping-owner change against the explicit historical capture."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repaired_trainer(original):
    old_import = b'from prepared import PreparedSample\n'
    old_call = b'torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)'
    if original.count(old_import) != 1 or original.count(old_call) != 1:
        raise ValueError('Original clipping owner seam differs')
    return original.replace(old_import, old_import+b'from gradient_clipping import clip_l2_\n').replace(
        old_call, b'clip_l2_(network.parameters(), 1.0)')


def verify_sources(project, reference):
    baseline = project/'baseline_train.py'
    if digest(baseline) != reference['original_sources']['train.py']:
        raise ValueError('Explicit original trainer identity differs')
    if (project/'train.py').read_bytes() != repaired_trainer(baseline.read_bytes()):
        raise ValueError('Only clipping import and call replacement permitted')
    for name, identity in reference['original_sources'].items():
        if Path(name).name != name or digest(baseline if name=='train.py' else project/name) != identity:
            raise ValueError('Original numerical source changed: '+name)
    for name, identity in reference['repair_sources'].items():
        if Path(name).name != name or digest(project/name) != identity:
            raise ValueError('Qualified clipping screen source changed: '+name)
    if digest(project/'config.json') != reference['configuration_sha256']:
        raise ValueError('Original full-length configuration differs')
    return True
