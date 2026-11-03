"""All original small-grid frames must have finite three-step backward passes."""
import sys
from v6_support import ROOT,BASE,read,write,sha
project=BASE/'v6-unrolled-project-01';sys.path.insert(0,str(project))
import torch,model,data,train,fine_tuning
model.configure('cpu');torch.set_num_threads(1);model.select(read(project/'config.json'))
slots={r['id']:ROOT/r['path']for r in read(project/'project.json')['data']}
_,parents=data.load(tuple(slots[k].read_bytes()for k in ['dataset','dataset_extra',*train.VIEW_SLOTS]),read(slots['splits']))
n=model.PressureCorrectionNet();fine_tuning.initialize(n,slots['initial_weights'],data.helpers());norm=model.fit_normalization(parents,data.fit_normalization);rows=[]
for parent in parents:
    if max(parent['counts_xyz'])>16:continue
    for frame in range(len(parent['frames'])):
        n.zero_grad(set_to_none=True);s=data.sample(parent,frame);loss=model.training_loss(n,s,norm).mean();loss.backward()
        if not bool(torch.isfinite(loss))or any(p.grad is None or not bool(torch.isfinite(p.grad).all())for p in n.parameters()):raise ValueError('Nonfinite small-grid unrolled backward: '+str((parent['entry']['trajectory_id'],frame)))
        rows.append(dict(trajectory_id=parent['entry']['trajectory_id'],frame=frame,loss=float(loss.detach())))
write(BASE/'unrolled-all-small-backward-01.json',dict(status='passed',rows=rows,source_sha256=sha(__file__),objective_sha256=sha(project/'fine_tuning.py'),optimizer_updates=0))
print('All',len(rows),'original small-grid training frames have finite unrolled gradients',flush=True)
