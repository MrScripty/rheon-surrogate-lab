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
bank=CapturedLongTrainingBank(transport['path'],{n:transport[n] for n in ('bytes','sha256')})
try:
 window=dict(bank.windows[0])
 proxy=SimpleNamespace(windows=[window],cuda_window=bank.cuda_window)
 diagnostic=[]
 try:
  full_pressure_geometry(model,proxy,normalization,lambda:None,diagnostic_receiver=diagnostic)
 finally:print(json.dumps(dict(scope='Single first training window/all12',diagnostic=diagnostic,optimizer_updates=0)),flush=True)
finally:model.close();bank.close()
