"""Minimal CUDA-only learned pressure adjoint check; no native labels or fits."""
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(root/"experiments"/n) for n in ("full-ai-data-diversity-v1","full-ai-tight-targets-v1","full-ai-next-priorities-v1","pressure-methods-v2")]
import torch
from guided_pressure_fit_core_v1 import make_model
from coupled_macrostep_v1 import _RestoredPressure

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
configuration=json.loads(Path('runs/full-ai-data-diversity-v1/guided-pressure-source-package-02/manifest.json').read_text())['configuration']
model=make_model(configuration,4)
try:
 counts=(64,64,64);model.model.prepare_geometry(counts)
 generator=torch.Generator(device='cuda').manual_seed(7218)
 q=torch.randn((1,1,*reversed(counts)),generator=generator,device='cuda',dtype=torch.float64,requires_grad=True)
 z=torch.randn(q.shape,generator=generator,device='cuda',dtype=torch.float64)
 direction=torch.randn(q.shape,generator=generator,device='cuda',dtype=torch.float64)
 p=_RestoredPressure.apply(q,model.model,counts)
 gradient=torch.autograd.grad((p*z).sum(),q)[0]
 with torch.no_grad():
  left=(model.model.pressure(direction,counts)*z).sum()
  right=(direction*gradient).sum()
  relative=float((left-right).abs()/torch.maximum(left.abs(),right.abs()).clamp_min(1e-300))
 print(json.dumps(dict(status='passed' if relative<1e-10 else 'failed',counts=list(counts),relative_adjoint_error=relative,optimizer_updates=0,native_labels_read=0)))
 assert relative<1e-10, 'Actual learned pressure adjoint disagrees with forward directional response'
finally:model.close()
