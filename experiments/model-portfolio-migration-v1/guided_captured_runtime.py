"""Captured Python workspace with in-place scientific payload bindings.

Only small Python source is materialized, preserving original __file__/ROOT
semantics. Weights, compiled kernels, JSON assets and data retain their existing
keepers. This is a temporary execution view, never another project or dataset.
"""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import zipfile

from import_common_generator_measurements import ROOT, RUN, require


class GuidedCapturedRuntime:
    def __init__(self,item,meta,contract,output,extra_read_paths=()):
        self.item,self.meta,self.contract=item,meta,contract
        self.output=Path(output).resolve()
        self.allowed={Path(ROOT/spec['path']).resolve() for spec in contract['original_material_observations']}
        self.allowed.update(Path(ROOT/name).resolve() for name in extra_read_paths)
        self.active=False;self.denied=[];self.code_bytes=0;self.links=0

    def __enter__(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='guided-captured-python-',dir=RUN)
        self.root=Path(self.temporary.name)
        self.cwd,self.paths=Path.cwd(),list(sys.path)
        try:
            members=self.meta['source_manifest']['members']
            with zipfile.ZipFile(ROOT/self.item['model_source']['path']) as archive:
                require(len(archive.infolist())==len(members) and set(archive.namelist())==set(members),
                        'Exact captured source membership required')
                for name,sha in members.items():
                    relative=Path(name)
                    require(not relative.is_absolute() and '..' not in relative.parts, 'Safe original source path required')
                    path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                    if relative.suffix=='.py':
                        raw=archive.read(name)
                        require(hashlib.sha256(raw).hexdigest()==sha, 'Captured Python source changed')
                        path.write_bytes(raw);self.code_bytes+=len(raw)
                    else:
                        keeper=ROOT/self.item['non_Python_keepers'][name]['path']
                        require(keeper.is_file() and not keeper.is_symlink()
                                and hashlib.sha256(keeper.read_bytes()).hexdigest()==sha,
                                'In-place original scientific keeper changed')
                        path.symlink_to(keeper);self.links+=1
            require(self.code_bytes==self.item['temporary_Python_source_bytes']
                    and self.links==len(self.item['non_Python_keepers']), 'Exact source-view footprint required')
            # Match the original portable source search order. Current task
            # helpers are already imported; ambient scientific modules cannot
            # fill a missing captured module through the old repository paths.
            sys.path[:]=[p for p in self.paths if p and not Path(p).absolute().is_relative_to(ROOT)]
            sys.path[:0]=[str(self.root/'experiments'/name) for name in
                ('full-ai-data-diversity-v1','full-ai-tight-targets-v1','full-ai-next-priorities-v1','pressure-methods-v2')]
            os.chdir(self.root)
            self.active=True
            def restrict(event,args):
                if not self.active or event!='open' or not isinstance(args[0],(str,bytes)):return
                path=Path(os.fsdecode(args[0])).absolute()
                actual=path.resolve()
                mode,flags=args[1],args[2]
                writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
                output=self.output==actual or self.output in actual.parents
                forbidden=(writing and actual.is_relative_to(ROOT) and not output)
                forbidden=forbidden or (actual.is_relative_to(ROOT) and not actual.is_relative_to(self.root)
                    and not output and actual not in self.allowed)
                forbidden=forbidden or actual.is_relative_to('/tmp/rheon-surrogate-lab-long-reference-240-v1')
                if forbidden:
                    self.denied.append(str(actual));raise PermissionError('Unbound ambient scientific access in captured guided runtime: '+str(actual))
            sys.addaudithook(restrict)
            return self
        except BaseException:
            self.__exit__();raise

    def imported_sources(self):
        captured={};members=self.meta['source_manifest']['members']
        for name,module in list(sys.modules.items()):
            filename=getattr(module,'__dict__',{}).get('__file__')
            if not isinstance(filename,str) or not filename.endswith('.py'):continue
            path=Path(filename).absolute()
            if path.is_relative_to(self.root):
                key=str(path.relative_to(self.root));sha=hashlib.sha256(path.read_bytes()).hexdigest()
                require(sha==members[key], 'Imported captured Python source differs')
                captured[name]=dict(member=key,sha256=sha)
        return captured

    def __exit__(self,*unused):
        self.active=False
        if hasattr(self,'cwd'):os.chdir(self.cwd);sys.path[:]=self.paths
        if hasattr(self,'temporary'):self.temporary.cleanup()


def restore_network(torch,np,model,item):
    expected=item['inventory']
    require(set(model.network.state_dict())==set(expected), 'Original residual architecture/state membership required')
    values={}
    with np.load(ROOT/item['weights']['path'],allow_pickle=False) as archive:
        require(set(archive.files)=={'w4.'+name for name in expected}, 'Original selected width4 archive keys required')
        for name,spec in expected.items():
            value=archive['w4.'+name]
            require(value.dtype.str==spec['dtype']=='<f4' and list(value.shape)==spec['shape']
                    and np.isfinite(value).all() and hashlib.sha256(value.tobytes()).hexdigest()==spec['sha256'],
                    'Original selected residual parameter changed')
            values[name]=torch.from_numpy(value.copy()).to('cuda')
    model.network.load_state_dict(values,strict=True)
    model.network.eval().requires_grad_(False)
    require(all(v.device.type=='cuda' for v in model.state_dict().values() if v.is_floating_point()),
            'All learned composition floating tensors must be CUDA')


def guard_direct_methods(torch,network):
    """These methods bypass nn.Module.__call__;qualify operands explicitly."""
    counters={name:0 for name in ('encode','endpoint','interior')}
    def tensors(value):
        if isinstance(value,torch.Tensor):return [value]
        if isinstance(value,(list,tuple)):return [v for x in value for v in tensors(x)]
        if isinstance(value,dict):return [v for x in value.values() for v in tensors(x)]
        return []
    for name in counters:
        original=getattr(network,name)
        def guarded(*args,_name=name,_method=original,**kwargs):
            operands=[*tensors(args),*tensors(kwargs)]
            require(operands and all(v.device.type=='cuda' for v in [*operands,*network.parameters(),*network.buffers()]),
                    'Direct learned '+_name+' must consume only CUDA operands')
            output=_method(*args,**kwargs)
            require(tensors(output) and all(v.device.type=='cuda' for v in tensors(output)), 'Direct learned output must remain CUDA')
            counters[_name]+=1
            return output
        setattr(network,name,guarded)
    return counters
