import json,hashlib,math,sys
from pathlib import Path
import numpy as np
import torch
root=Path.cwd();assets=root/'runs/full-ai-next-priorities-v1/temporal-pressure-core-assets-01'
config=json.loads((assets/'configuration.json').read_text());rows=[]
with np.load(assets/'geometry-gains.npz',allow_pickle=False)as arrays:
 for geometry in config['geometries']:
  counts=geometry['counts_zyx']
  for branch,modes in [('low',4),('tail',8)]:
   key=geometry[branch+'_gain_key']
   if key is None:continue
   lambdas=[4*torch.sin(torch.arange(min(modes,n),dtype=torch.float64)*math.pi/(2*n)).square()for n in counts]
   eigen=lambdas[0][:,None,None]+lambdas[1][None,:,None]+lambdas[2][None,None,:]
   gain=torch.from_numpy(arrays[key].copy())[:eigen.shape[0],:eigen.shape[1],:eigen.shape[2]]
   response=(eigen*gain).flatten()[1:]
   rows.append(dict(counts_zyx=counts,branch=branch,minimum_learned_gain_times_eigenvalue=float(response.min()),maximum_learned_gain_times_eigenvalue=float(response.max()),strict_contraction_in_nonDC_learned_subspace=bool(((response>0)&(response<2)).all())))
result=dict(status='completed_offline_bound_probe',rows=rows,learned_gains_unchanged=True,analytical_eigenvalues_only_in_offline_proof=True,numerical_inverse_constructed=False,temporal_candidate=False,CPU_latency_claim=False)
output=root/'runs/full-ai-next-priorities-v1/learned-pressure-gate-bounds-01.json'
if output.exists():raise ValueError('Fresh bound probe required')
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))
