"""Standalone scientific figures; data populations and timing scopes stay separate."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/rheon-v4-mpl-cache')
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from v4_support import ROOT,BASE,DOCS,read,write,sha

def build():
    if read(BASE/'all-timing-batch-02/execution.json')['status']!='completed':raise ValueError('Complete serial timing required before figures')
    output=DOCS/'figures';output.mkdir(parents=True,exist_ok=True);inputs={};artifacts={}
    def data(path):inputs[str(path.relative_to(ROOT))]=sha(path);return read(path)
    def save(fig,name):
        for suffix in ['svg','png','pdf']:
            path=output/(name+'.'+suffix)
            options=dict(dpi=160,bbox_inches='tight')
            if suffix=='pdf':options['metadata']={'CreationDate':None,'ModDate':None}
            fig.savefig(path,**options);artifacts[str(path.relative_to(ROOT))]=sha(path)
        plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':11,'svg.hashsalt':'rheon-pressure-methods-v4','savefig.facecolor':'white'})
    quiet=[data(p)for p in sorted((BASE/'quiet').glob('*.json'))]
    if len(quiet)!=19:raise ValueError('All19 common-population timing points required')
    def pareto(rows):
        result=[];best=float('inf')
        for d in sorted(rows,key=lambda d:(d['summary']['median_ms'],d['summary']['worst_pressure_relative_l2'])):
            y=d['summary']['worst_pressure_relative_l2']
            if y<best:result.append(d);best=y
        return result
    neural=[d for d in quiet if d['parameters']>0]
    qualified=[d for d in neural if d['summary']['qualified']==128]
    prior=[d for d in qualified if d['identifier']in ['C53-fine','C53-num6','v2-num6']]
    fig,ax=plt.subplots(figsize=(14,8));fig.subplots_adjust(right=.70,bottom=.18,top=.88)
    ax.set_yscale('log');ax.grid(True,which='both',alpha=.16)
    for rows,color,style,label in [(pareto(prior),'#7862a6','--','Prior matched neural/hybrid frontier controls'),(pareto(qualified),'#168b8d','-','Frontier among evaluated neural/hybrid policies')]:
        ax.plot([d['summary']['median_ms']for d in rows],[100*d['summary']['worst_pressure_relative_l2']for d in rows],style,color=color,lw=3.5 if style=='--'else 2,zorder=3 if style=='--'else 2,label=label)
    labels=[]
    for index,d in enumerate(quiet,1):
        s=d['summary'];good=s['qualified']==128;is_neural=d['parameters']>0
        color='#168b8d'if d['identifier'].startswith('v4-')else'#7862a6'
        if is_neural:
            x=s['median_ms'];y=100*s['worst_pressure_relative_l2']
            ax.scatter(x,y,color=color,marker='o'if good else'X',s=65,zorder=4)
            offset={11:(-18,-16),14:(8,10)}.get(index,(5,5 if index%2 else-12))
            ax.annotate(str(index),(x,y),xytext=offset,textcoords='offset points',fontsize=8,color=color)
        labels.append(f"{index:2d}  {d['identifier']}"+(' [rejected]'if not good else''))
    inset=inset_axes(ax,width='42%',height='35%',loc='upper left',borderpad=1.2)
    for d in quiet:
        s=d['summary'];pure=d['parameters']==0
        inset.scatter(s['median_ms'],100*s['worst_pressure_relative_l2'],s=35,marker='*'if d['policy']=='spectral'else'D'if pure else'o'if s['qualified']==128 else'X',color='#253f50'if pure else'#168b8d')
        if d['policy']=='spectral':inset.annotate('spectral',(s['median_ms'],100*s['worst_pressure_relative_l2']),xytext=(5,4),textcoords='offset points',fontsize=8)
    inset.set_xscale('log');inset.set_yscale('log');inset.grid(True,which='both',alpha=.12);inset.tick_params(labelsize=7);inset.set_title('All methods, including numerical controls',fontsize=9)
    lo=min(d['summary']['median_ms']for d in neural);hi=max(d['summary']['median_ms']for d in neural)
    ax.set_xlim(lo*.85,hi*1.08)
    ax.set_xlabel('Median pressure-policy time (ms; CPU 23, one thread)')
    ax.set_ylabel('Worst mean-free pressure relative L2 error (%)')
    ax.legend(loc='lower left',fontsize=8)
    fig.suptitle('Matched accuracy and cost: 128 historical requests on 5–16-cell grids',fontsize=15)
    fig.text(.72,.87,'Point identifiers',weight='bold');fig.text(.72,.835,'\n'.join(labels),va='top',family='monospace',fontsize=8.4,linespacing=1.65)
    fig.text(.08,.07,'Circles: both physical gates passed on every request. X: at least one failed; excluded from the frontier.\nOne population warmup and three repeats per request. At most eight corrections. These are selected matched policies, not all historical cohorts.',fontsize=9)
    save(fig,'historical-frontier')

    selection=data(DOCS/'selection.json');records={r['identifier']:r for r in selection['selected']+selection['references']}
    ordered=[selection['selected'][1]['identifier'],selection['selected'][2]['identifier'],selection['selected'][0]['identifier']]+[r['identifier']for r in selection['references']]
    ordered=list(dict.fromkeys(ordered));cells=[]
    for key in ordered:
        d=data(BASE/'evaluation'/(key+'.json'));b=data(BASE/'blind-evaluation'/(key+'.json'))
        groups=[[r for r in d['rows']if r['population']not in ['native-development','v4-development']],
                [r for r in d['rows']if r['population']=='native-development'],[r for r in d['rows']if r['population']=='v4-development'],b['rows']]
        cells.append([key,*[f"{sum(r['qualified']for r in group)}/{len(group)}"for group in groups]])
    fig,ax=plt.subplots(figsize=(13,7));ax.axis('off')
    tab=ax.table(cellText=cells,colLabels=['Policy','Retained prior','Known native','New development','Reserved blind'],colWidths=[.48,.13,.13,.13,.13],cellLoc='center',loc='center')
    tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,2.1)
    for (row,col),cell in tab.get_celld().items():
        cell.set_edgecolor('#d1dce0')
        if row==0:cell.set_facecolor('#e1edf2');cell.set_text_props(weight='bold')
        elif col:
            a,b=map(int,cells[row-1][col].split('/'));cell.set_facecolor('#d1e9e0'if a==b else'#f2d5c4')
        else:cell.set_text_props(ha='left')
    ax.set_title('Request qualification: coverage is distinct from a completed training fit',pad=18)
    fig.text(.07,.07,'Both original pressure-residual and predicted-divergence gates, at eight corrections. New blind data: five independent forcing scenes, cold/warm snapshots.\nGreen cells pass every request in that population. Application accuracy and long-term drift acceptance remain separate.',fontsize=9)
    save(fig,'native-coverage')

    timing=data(DOCS/'timing-reservation.json');chosen=selection['selected']
    methods=[('spectral','Spectral'),('numerical-gcr-multilevel','Numerical multilevel + GCR'),
        (chosen[2]['identifier'],'Multilevel + CNN'),(chosen[1]['identifier'],'Full model trained in GCR'),
        (chosen[0]['identifier'],'Student 75%, control'),('v3-native-baseline','Previous learned policy'),('v3-native-layout','Previous policy, channels-last')]
    fig,axes=plt.subplots(1,3,figsize=(16,6),sharey=True);fig.subplots_adjust(left=.19,bottom=.22,wspace=.12,top=.84)
    for ax,row in zip(axes,timing['rows']):
        for y,(key,label)in enumerate(methods):
            medians=[];accepted=True
            for rep in range(3):
                d=data(BASE/'coupled-quiet'/(row['trajectory_id']+'-'+key+f'-r{rep}')/'receipt.json')
                medians.append(statistics.median(f['frame_seconds']for f in d['frames']));accepted &=d['status']=='completed'
            value=statistics.median(medians);color='#168b8d'if accepted else'#c78d70'
            ax.barh(y,value,xerr=[[value-min(medians)],[max(medians)-value]],color=color,alpha=.85,hatch=None if accepted else'///',error_kw=dict(capsize=3,linewidth=1))
            ax.text(max(medians)*1.04,y,f'{value:.3f} s'+(' rejected'if not accepted else''),va='center',fontsize=7)
        ax.set_title(' × '.join(map(str,row['scene']['counts'])));ax.set_xlabel('Median headless fluid-step seconds');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.margins(x=.30)
    axes[0].set_yticks(range(len(methods)),[label for _,label in methods]);axes[0].invert_yaxis()
    fig.suptitle('Native fluid cost: matched four-step prefixes of reserved blind scenes',fontsize=15)
    fig.text(.08,.08,'Three serial repeats per method; bars are medians of per-run medians, whiskers show their observed range. CPU 23, one thread.\nHatched bars are rejected attempts, not usable solver timings. Binary exchange, preparation, projection and host gates count; GUI, loading and label/export work do not.',fontsize=9)
    save(fig,'native-fluid-time')

    drift=[('v3 baseline',BASE/'drift-development-64-01.json','#7b8793'),('Replay continuation',BASE/'drift-v4-replay-64-01.json','#7862a6'),('On-policy data',BASE/'drift-v4-onpolicy-64-01.json','#168b8d')]
    fig,axes=plt.subplots(1,3,figsize=(14,4.8));fig.subplots_adjust(bottom=.24,top=.80,wspace=.25)
    for label,path,color in drift:
        d=data(path)
        for ax,field in zip(axes,['pressure','velocity','tracer']):
            times=[r['time_s']for r in d['rows']];values=[max(r[field]['relative_l2']or 0,1e-12)for r in d['rows']]
            ax.plot(times,values,label=label,color=color,lw=1.4)
    for ax,field in zip(axes,['pressure','velocity','tracer']):
        ax.set_yscale('log');ax.set_title(field.capitalize());ax.set_xlabel('Physical simulated seconds');ax.grid(True,which='both',alpha=.15)
    axes[0].set_ylabel('Trajectory relative L2 difference');axes[2].legend(fontsize=8,loc='best')
    fig.suptitle('Pressure, velocity and tracer drift on matched 64³ trajectories',fontsize=13)
    fig.text(.07,.07,'Independent spectral reference and model-driven native tracks, 120 accepted steps at dt = 1/60 s. Pressure gauges are removed.\nZeros are drawn at 1e-12 for log plotting. One scene, diagnostic limits only; no application acceptance or longer safe horizon is inferred.',fontsize=9)
    save(fig,'trajectory-drift')
    write(DOCS/'figure-evidence.json',dict(status='generated_from_retained_results',inputs=inputs,artifacts=artifacts,
        source_sha256=sha(__file__),matplotlib_version=matplotlib.__version__,promotion=False))
    print('Generated four scientific figures in SVG, PNG and PDF',flush=True)

if __name__=='__main__':build()
