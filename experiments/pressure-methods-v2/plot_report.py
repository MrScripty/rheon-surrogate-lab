"""Export standalone scientific figures using Matplotlib, without the training runtime."""
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/rheon-v2-matplotlib')
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];DOCS=ROOT/'docs/plans/pressure-methods-v2';RUNS=ROOT/'runs/pressure-methods-v2'
os.sched_setaffinity(0,{22})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def frontier(rows,key):return sorted([d for d in rows if not any(o is not d and o['median_ms']<=d['median_ms'] and o[key]<=d[key] and (o['median_ms']<d['median_ms'] or o[key]<d[key]) for o in rows)],key=lambda d:d['median_ms'])
d=json.loads((DOCS/'results.json').read_text())
if d['status']!='completed':raise ValueError('Final measured report required for exported figures')
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
artifacts=[]
def point_label(label):
 if label=='Full spectral':return 'SP'
 if label=='Numerical two-level PCG6':return 'MG6'
 if label=='v2-num6 adaptive':return 'NA'
 if label=='v2-small16 adaptive':return 'SA'
 words=label.split();prefix='C' if label.startswith('C53') else 'S' if label.startswith('v2-small16') else 'H' if label.startswith('v2-headshape') else 'N'
 return prefix+words[-2]
fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
for ax,pop,key in [(axes[0,0],'consumed128','worst_pressure_relative_l2'),(axes[0,1],'consumed128','worst_relative_residual'),(axes[1,0],'all-development','worst_pressure_relative_l2'),(axes[1,1],'all-development','worst_relative_residual')]:
 rows=[r for r in d['chart'] if r['population']==pop];valid=[r for r in rows if r['qualified']==r['cases']]
 for index,r in enumerate(rows):
  ax.scatter(r['median_ms'],r[key],marker='o' if r in valid else 'x',s=34,color='#276da8' if r['ai'] else '#008c65')
  ax.annotate(point_label(r['label']),(r['median_ms'],r[key]),xytext=(5,[-9,4,11][index%3]),textcoords='offset points',fontsize=7,bbox=dict(facecolor='white',edgecolor='none',alpha=.65,pad=.1))
 for group,color,label in [(valid,'#008c65','All-method frontier'),([r for r in valid if r['ai']],'#276da8','Neural-policy frontier')]:
  f=frontier(group,key)
  ax.plot([r['median_ms'] for r in f],[r[key] for r in f],color=color,lw=1,alpha=.65,label=label)
 if key=='worst_relative_residual':ax.axhline(1e-6,color='#be5b51',ls=':',lw=1,label='Equation tolerance')
 ax.set(xscale='log',yscale='log',xlabel='Median warm prepared-policy time (ms)',ylabel='Worst mean-free pressure relative L2' if key=='worst_pressure_relative_l2' else 'Worst relative equation residual',title=('Original 128 requests' if pop=='consumed128' else 'All 156 development requests'))
 ax.set_xlim(min(r['median_ms'] for r in rows)*.8,max(r['median_ms'] for r in rows)*1.25)
 ax.grid(which='major',alpha=.2);ax.legend(fontsize=7,loc='best')
fig.supxlabel('C: C53 extended · S: small control · H: shape-informed learned · N: trained numerical6 · number: passes\nNA/SA: adaptive trained/small control + numerical6 · MG6: numerical PCG6 · SP: spectral\nPressure errors near 10⁻⁸ reflect finite native-CG reference error; interpret alongside equation residuals.',fontsize=8)
fig.suptitle('Pressure accuracy versus policy cost — three quiet warm repetitions\nCrosses fail an equation gate; loading, IPC and simulation excluded; C53 uses the qualified extended wrapper')
for suffix in ['svg','png']:
 path=DOCS/('policy-frontiers.'+suffix);fig.savefig(path,dpi=180);artifacts.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
plt.close(fig)
series=[('harder-drift-long-01/scene0-neural-8','C53, 8 passes'),('harder-drift-long-01/scene0-periodic-8','C53 + spectral every 16'),('harder-drift-long-01/scene0-spectral-8','Full spectral'),('harder-drift-long-01/scene0-triggered-4','C53 4 + residual trigger'),('cost-native-01/scene0-cost-hybrid-7200','Numerical6 + small control, adaptive'),('finalist-native-long-01/v2-headshape-scene0-7','Shape-informed learned, 7 passes'),('finalist-native-long-01/v2-num6-scene0-7','Trained numerical6, 7 passes')]
fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True,layout='constrained');sources={}
for folder,label in series:
 path=RUNS/folder/'results.json';sources[str(path.relative_to(ROOT))]=sha(path);raw=json.loads(path.read_text());points=raw['points']
 for ax,key in zip(axes,['pressure_trajectory_relative_l2','velocity_trajectory_relative_l2','tracer_trajectory_relative_l2']):
  x=[p['time_s'] for p in points];y=[max(p[key],1e-16) for p in points];ax.plot(x,y,lw=.8,label=label,alpha=.85)
for ax,name in zip(axes,['Cell-zero-gauged pressure','Staggered velocity','Cell tracer concentration']):
 ax.set(yscale='log',ylabel='Trajectory relative L2',title=name);ax.axhline(1e-6,ls=':',lw=1,color='#555');ax.grid(alpha=.2)
axes[-1].set_xlabel('Simulated time (s)');axes[0].legend(ncol=2,fontsize=8,loc='lower right');fig.suptitle('120-second varied-force 16³ scene — diagnostic drift, no application acceptance limits\nZero errors are displayed at 10⁻¹⁶ on the logarithmic axis')
for suffix in ['svg','png']:
 path=DOCS/('drift-120s.'+suffix);fig.savefig(path,dpi=180);artifacts.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
plt.close(fig)
(DOCS/'figure-evidence.json').write_text(json.dumps(dict(status='completed',matplotlib=matplotlib.__version__,python=sys.version,source_sha256=sha(Path(__file__)),report_sha256=sha(DOCS/'results.json'),drift_sources=sources,artifacts=artifacts),indent=2)+'\n')
print('Exported',len(artifacts),'standalone scientific figures')
