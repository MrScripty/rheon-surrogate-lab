"""Minimal CUDA-only learned pressure adjoint check; no native labels or fits."""
import json
import argparse
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(root/"experiments"/n) for n in ("full-ai-data-diversity-v1","full-ai-tight-targets-v1","full-ai-next-priorities-v1","pressure-methods-v2")]
import torch
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--heads',choices=('f32','f64'),required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise ValueError('Fresh regression result required')
if args.heads=='f64':
 from guided_pressure_fit_core_v2 import make_model
else:
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
 window=dict(bank.windows[0]);window['configurations']=window['configurations'][1:2]
 proxy=SimpleNamespace(windows=[window],cuda_window=bank.cuda_window)
 diagnostic=[]
 try:
  try:
   full_pressure_geometry(model,proxy,normalization,lambda:None,diagnostic_receiver=diagnostic)
  except ValueError as error:
   if str(error) != 'Incomplete/nonfinite full-population pressure geometry':raise
   assert len(diagnostic)==1
 finally:
  agreement=(len(diagnostic)==1 and (diagnostic[0]['relative_disagreement']<=1e-4
    or diagnostic[0]['maximum_absolute_disagreement']<=1e-14))
  result=dict(status='passed' if agreement else 'failed_gradient_agreement',
    scope='Single first training window/30outputsK2',head_arithmetic=args.heads,
    diagnostic=diagnostic,optimizer_updates=0,parameter_fits=0)
  args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
  print(json.dumps(result),flush=True)
finally:model.close();bank.close()
