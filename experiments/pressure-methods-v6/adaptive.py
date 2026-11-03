"""Deterministic reuse admission from frozen development costs and input sketches."""
import math
from v6_support import read,sha
import torch
class AdaptiveRule:
    def __init__(self,path):
        self.config=read(path);self.path=path;self.identity=sha(path);self.history=None;self.pending=None
        c=self.config
        if not math.isfinite(c['overhead_margin'])or c['overhead_margin']<1:raise ValueError('Conservative finite overhead margin required')
        if c['status']!='frozen_from_consumed_development'or not c['bands']or c['minimum_cosine']!=.9:raise ValueError('Frozen development cost model required')
        for band in c['bands']:
            if not 0<band['baseline_iterations']<=8 or not 0<=band['saved_fraction']<=1 or not band['cycle_seconds_per_cell']>0 or not band['overhead_seconds_per_cell']>=0:raise ValueError('Finite positive development costs required')
            if not all(math.isfinite(band[k])for k in ['saved_fraction','cycle_seconds_per_cell','overhead_seconds_per_cell']):raise ValueError('Finite cost model required')
    def decide(self,initial,key):
        if self.pending is not None:raise ValueError('Commit or reject the previous adaptive decision')
        stride=[max(1,(n+7)//8)for n in initial.shape[2:]]
        sketch=initial[:,:,::stride[0],::stride[1],::stride[2]].flatten().clone()
        cells=initial.numel();band=min(self.config['bands'],key=lambda b:abs(math.log(cells/b['cells'])))
        same=self.history is not None and self.history['key']==key and self.history['sketch'].shape==sketch.shape
        cosine=None
        if same:
            old=self.history['sketch'];den=torch.linalg.vector_norm(old)*torch.linalg.vector_norm(sketch)
            cosine=float(torch.dot(old,sketch)/den)if float(den)>0 else 0.
        baseline=self.history['baseline_iterations']if same else band['baseline_iterations']
        saved_iterations=baseline*band['saved_fraction'];savings=saved_iterations*band['cycle_seconds_per_cell']*cells
        overhead=band['overhead_seconds_per_cell']*cells
        economic=savings>self.config['overhead_margin']*overhead
        admitted=bool(economic and cosine is not None and cosine>=self.config['minimum_cosine'])
        decision=dict(admitted=admitted,collect_basis=economic,sketch_cosine=cosine,
          expected_saved_iterations=saved_iterations,expected_savings_seconds=savings,expected_overhead_seconds=overhead,
          overhead_margin=self.config['overhead_margin'],calibration_sha256=self.identity,calibration_band_cells=band['cells'],
          extrapolated_grid=cells!=band['cells'],baseline_iteration_estimate=baseline)
        self.pending=dict(key=key,sketch=sketch,band=band,admitted=admitted,old_baseline=baseline)
        return decision
    def observe(self,work):
        if self.pending is None:raise ValueError('Adaptive decision must precede observation')
        new=work['coarse_calls'];p=self.pending
        # Reused solves do not observe counterfactual fresh work. Keep the
        # estimate anchored to the last fresh work, bounded by observed new rays.
        estimate=max(new,p['old_baseline'])if p['admitted']else new
        p['baseline_iterations']=.75*p['old_baseline']+.25*estimate
    def commit(self,advanced):
        if advanced and self.pending is not None:
            if 'baseline_iterations'not in self.pending:raise ValueError('Missing completed adaptive observation')
            self.history=self.pending
        self.pending=None
