import sys,os,json
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root/'experiments/full-ai-data-diversity-v1'))
from support import read
from legacy_exact_forcing import forcing
from native_forcing_client import NativeForcingClient
import torch,numpy as np
os.sched_setaffinity(0,{4});torch.set_num_threads(1);torch.set_num_interop_threads(1)
reg=read(root/'docs/plans/full-ai-data-diversity-v1/strict-legacy-force-qualification-protocol-01.json');case=reg['cases'][0];scene=case['scene']
client=NativeForcingClient(scene,root/'runs/full-ai-data-diversity-v1/strict-legacy-force-difference-01')
try:
 inc=forcing(scene,0,1)[0];base=[(-v).float() for v in inc];actual=client.preview_force(0,[v.numpy() for v in base]);merged=[(v.double()+a).float().numpy() for v,a in zip(base,inc)];rows=[]
 for a,b in zip(actual,merged):
  at=a.view('u4')!=b.reshape(-1).view('u4');bv=b.reshape(-1);rows.append({'byte_differences':int(at.sum()),'nonzero_numeric_differences':int(np.count_nonzero(a-bv)),'all_differences_are_signed_zero':bool(np.all((a[at]==0)&(bv[at]==0))),'max_absolute':float(np.max(np.abs(a-bv)))})
finally:owner=client.close()
result={'identifier':case['identifier'],'step':0,'base':'canceling','rows':rows,'native_process':owner}
p=root/'runs/full-ai-data-diversity-v1/strict-legacy-force-difference-01/result.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
