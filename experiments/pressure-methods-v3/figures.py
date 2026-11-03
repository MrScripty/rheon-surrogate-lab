"""Standalone scientific plots from common-population retained measurements."""
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/rheon-v3-mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/pressure-methods-v3'
DOCS=ROOT/'docs/plans/pressure-methods-v3'

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def plot():
    destination=DOCS/'figures';destination.mkdir(exist_ok=True)
    rows=[read(p)for p in sorted((BASE/'quiet').glob('*.json'))]
    if len(rows)!=11 or any(r['status']!='completed'or r['cpu']!=23 or r['summary']['cases']!=128 for r in rows):
        raise ValueError('All 11 matched historical-population benchmarks required')
    fig,ax=plt.subplots(figsize=(12.5,8))
    fig.subplots_adjust(left=.10,right=.96,bottom=.19,top=.86)
    inset=ax.inset_axes([.035,.53,.32,.40])
    offsets={'v3-native':(-60,16),'v3-replay':(-70,-28),'v2-headshape':(10,-2),
        'C37-num6':(-95,15),'C44-num6':(8,10),'C53-num6':(8,-14),
        'v3-c37half':(10,20),'v3-c44half':(10,-25),'v2-num6':(8,-20),'C53-fine':(8,10)}
    for index,row in enumerate(rows):
        summary=row['summary'];x=summary['median_ms'];y=max(summary['worst_pressure_relative_l2']*100,1e-14)
        qualified=summary['qualified']==128
        color='#17818c'if row['identifier'].startswith('v3-')else'#6c5298'
        marker='o'if qualified else'X'
        if row['kind']=='spectral':color='#253e52';marker='*'
        inset.scatter(x,y,s=75 if marker=='*'else 15,marker=marker,color=color,zorder=4)
        if row['kind']=='spectral':
            inset.annotate('spectral',(x,y),xytext=(5,9),textcoords='offset points',fontsize=7)
            continue
        ax.scatter(x,y,s=65,marker=marker,color=color,zorder=4)
        ax.annotate(row['identifier'],(x,y),xytext=offsets[row['identifier']],textcoords='offset points',fontsize=9,
            arrowprops=dict(arrowstyle='-',lw=.5,color='#7a8790'))
    eligible=[r for r in rows if r['summary']['qualified']==128 and r['kind']!='spectral']
    frontier=[];best=float('inf')
    for row in sorted(eligible,key=lambda r:r['summary']['median_ms']):
        error=row['summary']['worst_pressure_relative_l2']
        if error<best:frontier.append(row);best=error
    if frontier:
        ax.plot([r['summary']['median_ms']for r in frontier],[r['summary']['worst_pressure_relative_l2']*100 for r in frontier],
            color='#de8e22',lw=1.5,alpha=.8,label='Neural / hybrid Pareto frontier (all 128 gates passed)')
    ax.set(yscale='log',xlim=(9.5,20),ylim=(3e-5,2),xlabel='Median pressure-method inference time (ms, CPU 23, one thread)',
        ylabel='Worst mean-free pressure relative L2 error (%)',
        title='Neural / hybrid comparison: same 128 historical requests (5–16 cells per axis)\nAdaptive stopping; at most eight neural corrections. Inset includes the spectral control.')
    inset.set(xscale='log',yscale='log',title='Full comparison',xlim=(.20,25))
    inset.tick_params(labelsize=7);inset.title.set_fontsize(9);inset.grid(True,which='both',alpha=.15)
    ax.grid(True,which='both',alpha=.18)
    if frontier:ax.legend(loc='upper right',fontsize=8)
    fig.text(.10,.035,'Circles: all physical gates passed. X: at least one failed. Star: full spectral control (inset).\nThe line compares measured neural/hybrid operating points. Small-grid timings do not establish 128³ performance.',fontsize=9)
    artifacts=[]
    for extension in ['svg','png','pdf']:
        path=destination/f'historical-frontier.{extension}';fig.savefig(path,dpi=180,bbox_inches='tight');artifacts.append(path)
    plt.close(fig)
    native=[read(p)for p in sorted((BASE/'evaluation').glob('*.json'))]
    ids=[r['entry']['id']for r in native]
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    positions=list(range(len(ids)))
    for offset,passes,color in [(-.18,8,'#68adbc'),(.18,16,'#28576e')]:
        values=[next(s['qualified']for s in r['summary']if s['population']=='native-development'and s['max_passes']==passes)for r in native]
        ax.bar([x+offset for x in positions],values,width=.34,label=f'Ceiling {passes}',color=color)
    ax.set(xticks=positions,xticklabels=ids,ylim=(0,8.8),ylabel='Native development requests passing both physical gates',
        title='64³ / 128³ cold and warm native requests (eight total)')
    ax.tick_params(axis='x',rotation=35);ax.set_yticks(range(9));ax.legend();ax.grid(axis='y',alpha=.15)
    for extension in ['svg','png','pdf']:
        path=destination/f'native-coverage.{extension}';fig.savefig(path,dpi=180,bbox_inches='tight');artifacts.append(path)
    plt.close(fig)
    evidence=dict(status='completed',source_sha256=sha(Path(__file__)),matplotlib=matplotlib.__version__,
        inputs={str(p.relative_to(ROOT)):sha(p)for p in [*(BASE/'quiet').glob('*.json'),*(BASE/'evaluation').glob('*.json')]},
        artifacts=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p))for p in artifacts])
    (DOCS/'figure-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('Exported matched-population frontier and native-coverage SVG/PNG/PDF',flush=True)

if __name__=='__main__':plot()
