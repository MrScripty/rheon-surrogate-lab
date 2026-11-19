"""Reuse the complete fixed learned pressure map through neural unit responses.

No inverse or reference field constructs these coefficients. Small-grid maps
compose the existing learned stages in float64; qualified modal bounds justify
their linear accepted branch in real arithmetic. Rounding requires separate
qualification. The unchanged component handles the larger registered geometry.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from frozen_temporal_pressure_core import FrozenPressureCore,index_laplacian,restrict,prolong


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FullLearnedPressureMap(nn.Module):
    _cache={}
    _cache_bytes=0
    cache_limit_bytes=512*1024**2

    def __init__(self,asset_directory,bound_path):
        super().__init__();assets=Path(asset_directory);bound_path=Path(bound_path)
        bindings=json.loads(bound_path.with_name(bound_path.stem+'.source.json').read_text())
        if digest(bound_path)!=bindings['result_sha256']or digest(assets/'configuration.json')!=bindings['asset_configuration_sha256']or digest(assets/'geometry-gains.npz')!=bindings['learned_gain_asset_sha256']:
            raise ValueError('Learned modal bound/asset identity changed')
        bounds=json.loads(bound_path.read_text())
        if bounds['status']!='completed_offline_bound_probe'or bounds['numerical_inverse_constructed']:
            raise ValueError('Learned-gain-only bound qualification required')
        self.preparation=[];self.routes={}
        with torch.inference_mode(False):
            self.source=FrozenPressureCore(assets).eval()
            key_prefix=tuple(digest(assets/name)for name in('configuration.json','fine-weights.npz','geometry-gains.npz'))
            key_prefix+=(digest(__file__),digest(Path(__file__).with_name('frozen_temporal_pressure_core.py')),digest(bound_path))
            spatial=None
            for counts,(prefix,tail) in self.source.geometry.items():
                if max(counts)>16:
                    self.routes[counts]=None;continue
                for branch in ('low','tail')if tail else('low',):
                    bound=next(row for row in bounds['rows']if tuple(row['counts_zyx'])==counts and row['branch']==branch)
                    if not bound['strict_contraction_in_nonDC_learned_subspace']or not 0<bound['minimum_learned_gain_times_eigenvalue']<=bound['maximum_learned_gain_times_eigenvalue']<2:
                        raise ValueError('Unqualified learned modal contraction')
                key=key_prefix+(counts,);cached=self._cache.get(key)
                if cached is None:
                    cells=math.prod(counts);needed=cells*cells*8
                    if needed+self._cache_bytes>self.cache_limit_bytes:raise ValueError('Complete learned-map cache exceeds512MiB')
                    if spatial is None:spatial=copy.deepcopy(self.source.fine).double().eval()
                    tick=time.perf_counter()
                    with torch.no_grad():
                        matrix=torch.empty((cells,cells),dtype=torch.float64)
                        for start in range(0,cells,4):
                            batch=min(4,cells-start);q=torch.zeros((batch,1,*counts),dtype=torch.float64)
                            q.flatten(1)[torch.arange(batch),start+torch.arange(batch)]=1.
                            matrix[:,start:start+batch]=self.linear_graph(q,prefix,tail,spatial).flatten(1).T
                    if not bool(torch.isfinite(matrix).all())or torch.count_nonzero(matrix[0]):
                        raise ValueError('Finite cell-zero learned unit-response map required')
                    preparation=dict(counts_zyx=counts,cells=cells,matrix_bytes=needed,seconds=time.perf_counter()-tick,
                        unit_queries=cells,unit_batch_size=4,spatial_parameters_unchanged=True,
                        method='Complete float64 composition of trained neural coefficients on unit inputs; no numerical inverse or reference labels')
                    cached=(matrix,preparation,matrix._version);self._cache[key]=cached
                    type(self)._cache_bytes+=needed
                    print('Constructed full learned pressure map',counts,'bytes',needed,flush=True)
                    cache_hit=False
                else:cache_hit=True
                matrix,preparation,version=cached
                if matrix._version!=version:raise ValueError('Cached neural unit-response coefficients were modified')
                name=prefix+'_full_matrix';self.register_buffer(name,matrix,persistent=False);self.routes[counts]=name
                self.preparation.append(dict(preparation,cache_hit=cache_hit,constructor_seconds=0. if cache_hit else preparation['seconds']))
        self.requires_grad_(False)

    def proposal(self,residual,prefix,branch):
        axes=tuple(getattr(self.source,f'{prefix}_{branch}_axis{i}')for i in range(3))
        gain=getattr(self.source,prefix+'_'+branch)[:axes[0].shape[0],:axes[1].shape[0],:axes[2].shape[0]]
        pressure=prolong(restrict(residual,axes)*gain,axes)
        return pressure-pressure[:,:,:1,:1,:1]

    def linear_graph(self,q,prefix,tail,spatial):
        pressure=torch.zeros_like(q);fine_matrix=getattr(self.source,prefix+'_fine_matrix',None)
        for stage in range(self.source.steps):
            residual=q if stage==0 else q-index_laplacian(pressure)
            proposal=self.proposal(residual,prefix,'low');remaining=residual-index_laplacian(proposal)
            values=remaining/self.source.rhs_scale
            raw=spatial(values)if fine_matrix is None else F.linear(values.flatten(1),fine_matrix).reshape_as(values)
            delta=raw*self.source.output_scale;delta=delta-delta[:,:,:1,:1,:1]
            pressure=pressure+proposal+delta
        if tail:pressure=pressure+self.proposal(q-index_laplacian(pressure),prefix,'tail')
        return pressure

    def forward(self,q):
        if q.ndim!=5 or q.shape[1]!=1 or not 1<=q.shape[0]<=4 or q.dtype not in(torch.float32,torch.float64):
            raise ValueError('One to four colocated pressure-unit requests required')
        counts=tuple(q.shape[2:])
        if counts not in self.routes:raise ValueError('Pressure geometry not exported/qualified')
        route=self.routes[counts]
        if route is None:return self.source(q)
        matrix=getattr(self,route)
        if matrix.dtype!=torch.float64 or matrix.device!=q.device or not bool(torch.isfinite(q).all()):
            raise ValueError('Finite input and colocated float64 neural map required')
        return F.linear(q.flatten(1).double(),matrix).reshape_as(q)
