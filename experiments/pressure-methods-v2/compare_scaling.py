"""Compare saved large-grid native pressures only when initial requests match exactly."""
import hashlib,json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];RUNS=ROOT/'runs/pressure-methods-v2';DOCS=ROOT/'docs/plans/pressure-methods-v2'
os.sched_setaffinity(0,{22})
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(path.read_text())
paths=[]
for folder in ['scaling-baseline-01','scaling-controls-128-01','scaling-selected-128-01','scaling-256-01','scaling-256-02','scaling-warm-start-01']:
 paths.extend((RUNS/folder).glob('native-*/results.json'))
 paths.extend((RUNS/folder).glob('warm-native-*/results.json'))
references={}
for path in paths:
 d=read(path)
 if d['kind']=='spectral':references[(tuple(d['grid_dimensions']),d['cells_per_metre'],d['scene']['warmup_steps'],d['scene']['steps'])]=(path,d)
rows=[];skipped=[]
for path in sorted(paths):
 d=read(path)
 if d['prediction'] is None or d['request_identity'] is None:
  skipped.append(dict(case=path.parent.name,result=str(path.relative_to(ROOT)),reason='Original in-process native solver does not export the pressure field or external request; retain its host residual and frame measurements without inventing a reference error.'))
  continue
 key=(tuple(d['grid_dimensions']),d['cells_per_metre'],d['scene']['warmup_steps'],d['scene']['steps'])
 if key not in references:raise ValueError('Exact spectral reference missing')
 refpath,reference=references[key]
 if d['request_identity']!=reference['request_identity']:raise ValueError('Native initial requests differ')
 values=[]
 for p,x in [(path,d),(refpath,reference)]:
  descriptor=x['prediction'];artifact=p.parent/descriptor['path']
  if sha(artifact)!=descriptor['sha256']:raise ValueError('Saved pressure changed')
  array=np.load(artifact,mmap_mode='r',allow_pickle=False)
  if list(array.shape)!=descriptor['shape'] or array.dtype.str!=descriptor['dtype']:raise ValueError('Pressure descriptor mismatch')
  values.append(array)
 pressure,truth=values;error=pressure-truth;centered=error-error.mean();target=truth-truth.mean()
 meanfree=float(np.linalg.norm(centered.ravel()))/max(float(np.linalg.norm(target.ravel())),1e-300)
 gauged=float(np.linalg.norm(error.ravel()))/max(float(np.linalg.norm(truth.ravel())),1e-300)
 rows.append(dict(case=path.parent.name,status=d['status'],size=d['grid_dimensions'][0],cells_per_metre=d['cells_per_metre'],
  spectral_mean_free_pressure_relative_l2=meanfree,spectral_gauged_pressure_relative_l2=gauged,
  result=str(path.relative_to(ROOT)),result_sha256=sha(path),reference=str(refpath.relative_to(ROOT)),reference_sha256=sha(refpath),
  request_identity=d['request_identity']))
result=dict(status='completed',scope='One matched native pressure request per grid, cell spacing and warm-up context, exact spectral pressure reference; rejected model pressure still compared, never advanced.',rows=rows,skipped_field_comparisons=skipped,source_sha256=sha(Path(__file__)),numpy=np.__version__)
(DOCS/'native-scaling-accuracy.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('Compared',len(rows),'saved native pressure fields')
