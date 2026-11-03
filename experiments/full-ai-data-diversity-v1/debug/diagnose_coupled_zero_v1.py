import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'experiments/full-ai-data-diversity-v1'))
import support
import numpy as np,torch,json
from coupled_macrostep_v1 import CoupledMacroBackbone,wall
from recurrent_sampler_native import dimensions
from cuda_native_transfer_v1 import FIELDS
D=Path('docs/plans/full-ai-data-diversity-v1');reg=json.loads((D/'coupled-operator-qualification-protocol-01.json').read_text());bank=json.loads(Path(reg['case_selection']).read_text());case=bank['cases'][0]
scene=json.loads(Path(case['receipt']).read_text())['scene'];spec=next(s for s in case['states'] if s['step']==32)
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
a=np.load(spec['path']);fields={n:torch.from_numpy(a[n].copy()).to('cuda') for n in FIELDS}
m=CoupledMacroBackbone(*[reg[n] for n in ['transport_pair','shared_pair','build','control_build']]);c=m.context(fields,scene,32);core=m.model
h,dt,rho=c['h'],c['dt'],c['rho'];shape=c['shape'];counts=c['counts'];prev=c['previous'];pv=c['provisional']
def check(n,a,b):
 d=a.double()-b.double();print(n,'different',int((a.contiguous().view(torch.int64 if a.dtype==torch.float64 else torch.int32)!=b.contiguous().view(torch.int64 if b.dtype==torch.float64 else torch.int32)).sum()),'max_abs',float(d.abs().max()),'norm',float(d.norm()),flush=True)
r0=core.gather_ops.ordered_rhs(core,pv,shape,'flux',0,h*h,dt);r1=m.rhs(pv,c);check('rhs',r0,r1)
h0=core.gather_ops.three(core,[prev[0,0]],None,counts,'momentum',(h*h)*(1/(rho*h)),multiply_only=True,as_double=True)
h1=[wall(m.linear([prev[0,0]],s,'momentum',i)*((h*h)*(1/(rho*h))),i) for i,s in enumerate(dimensions(counts)[:3])]
for i in range(3):check('history'+str(i),h0[i],h1[i])
q0=core.gather_ops.difference(core,h0,r0,shape,'flux',0,-1.,rho/h)
act=m.linear(h1,shape,'flux',0,sign=-1);q1=(r1-act)*(rho/h);check('request',q0,q1)
ps0=core.affine_ops.binary('sum_gauge',prev,core.pressure(q0.reshape(1,1,*shape),counts));em=m.emit(c);check('pressure',ps0.flatten(),em['pressure'])
vv=core.gather_ops.three(core,[ps0[0,0]],pv,counts,'momentum',dt*(1/(rho*h)))
for i,n in enumerate(['vx','vy','vz']):check(n,vv[i].flatten(),em[n])
rr=core.affine_ops.ordered_three([m.linear(pv,shape,'flux',0,i).contiguous() for i in range(3)],h*h,dt);check('ordered_three_repair_rhs',r0,rr)
m.close()
