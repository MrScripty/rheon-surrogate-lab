"""Classify real lab Python files separately from pinned Torch virtual scopes."""
from pathlib import Path

VIRTUAL_TORCH_FILES = {'torch.classes':'_classes.py','torch.ops':'_ops.py'}


def collect(modules,root):
    root = root.resolve()
    sources,virtual = set(),[]
    for name,module in list(modules.items()):
        if name in VIRTUAL_TORCH_FILES:
            filename = getattr(module,'__file__',None)
            if filename != VIRTUAL_TORCH_FILES[name]:
                raise ValueError('Pinned Torch virtual namespace metadata changed')
            virtual.append(dict(module=name,synthetic_file=filename))
            continue
        filename = getattr(module,'__dict__',{}).get('__file__')
        if not isinstance(filename,str) or Path(filename).suffix != '.py':
            continue
        path = Path(filename)
        if not path.is_absolute():
            path = root/path
        resolved = path.resolve()
        if resolved.is_relative_to(root):
            if not resolved.is_file():
                raise ValueError('Real imported lab Python source is unavailable: '+str(resolved.relative_to(root)))
            sources.add(str(resolved.relative_to(root)))
    return dict(lab_python=sorted(sources),runtime_virtual_modules=sorted(virtual,key=lambda row:row['module']))
