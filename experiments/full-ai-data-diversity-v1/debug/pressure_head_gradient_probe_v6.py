"""Minimal CUDA-only learned pressure adjoint check; no native labels or fits."""
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(root/"experiments"/n) for n in ("full-ai-data-diversity-v1","full-ai-tight-targets-v1","full-ai-next-priorities-v1","pressure-methods-v2")]
import torch
from guided_pressure_fit_core_v1 import make_model
from captured_long_training_bank_v1 import CapturedLongTrainingBank
from pressure_head_geometry_v2 import full_pressure_geometry
from types import SimpleNamespace

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
configuration=json.loads(Path('runs/full-ai-data-diversity-v1/guided-pressure-source-package-02/manifest.json').read_text())['configuration']
transport=json.loads(Path('runs/full-ai-data-diversity-v1/long-training-transport-01/results.json').read_text())['archive']
normalization=json.loads(Path('docs/plans/full-ai-tight-targets-v1/sampler-accuracy-normalization-01.json').read_text())
model=make_model(configuration,4)
# Change only arithmetic of the two pressure-head forward reductions. Stored
# parameters remain F32 and unchanged. Module hooks retain the same inputs.
import types
from torch.nn import functional as F
for layer in (model.network.endpoint_pressure, model.network.interior_pressure):
 def double_forward(self, value):
  return F.conv3d(value.double(), self.weight.double())
 layer.forward=types.MethodType(double_forward,layer)

bank=CapturedLongTrainingBank(transport['path'],{n:transport[n] for n in ('bytes','sha256')})
try:
 window=dict(bank.windows[0]);window['configurations']=window['configurations'][1:2]
 proxy=SimpleNamespace(windows=[window],cuda_window=bank.cuda_window)
 diagnostic=[]
 try:
  try:
   full_pressure_geometry(model,proxy,normalization,lambda:None,diagnostic_receiver=diagnostic)
  except ValueError as error:
   if str(error) != 'Incomplete/nonfinite full-population pressure geometry':raise
   assert len(diagnostic)==1
 finally:print(json.dumps(dict(scope='Single first training window/30outputsK2/F64head arithmetic',diagnostic=diagnostic,optimizer_updates=0)),flush=True)
finally:model.close();bank.close()
