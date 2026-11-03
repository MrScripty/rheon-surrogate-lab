"""Standalone scientific comparisons from matched, retained current evidence."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/rheon-v5-mpl-cache')
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v5_support import ROOT,HERE,BASE,DOCS,read,write,sha

def build():
    for name in ['all-timing-batch-03','prior-controls-timing-batch-04']:
        if read(BASE/name/'execution.json')['status']!='completed':raise ValueError('Complete all serial timing before figures')
    output=DOCS/'figures';output.mkdir(exist_ok=True);inputs={};artifacts={};names=[]
    def data(path):inputs[str(path.relative_to(ROOT))]=sha(path);return read(path)
    def save(fig,name):
        names.append(name)
        for suffix in ['svg','png','pdf']:
            p=output/(name+'.'+suffix);opts=dict(dpi=160,bbox_inches='tight')
            if suffix=='pdf':opts['metadata']={'CreationDate':None,'ModDate':None}
            if suffix=='svg':opts['metadata']={'Date':None}
            fig.savefig(p,**opts);artifacts[str(p.relative_to(ROOT))]=sha(p)
        plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'svg.hashsalt':'rheon-v5-pressure','savefig.facecolor':'white'})
    quiet=[data(p)for p in sorted((BASE/'quiet-pressure').glob('*.json'))]
    if len(quiet)!=26:raise ValueError('All thirteen matched points on both timing populations required')
    ids=sorted({d['identifier']for d in quiet});number={key:i+1 for i,key in enumerate(ids)}
    def frontier(rows):
        best=float('inf');result=[]
        for d in sorted(rows,key=lambda d:(d['summary']['median_ms'],d['summary']['worst_pressure_relative_l2'])):
            y=d['summary']['worst_pressure_relative_l2']
            if y<best:result.append(d);best=y
        return result
    fig,grid=plt.subplots(2,2,figsize=(17,11),gridspec_kw={'height_ratios':[3,1.2]});axes=grid[0];fig.subplots_adjust(right=.76,bottom=.16,top=.91,wspace=.28,hspace=.43)
    for ax,pop,title in zip(axes,['historical128','native-blind'],['128 retained requests; 5–16 cells per axis','10 fresh native requests; 32³ through 128³']):
        group=[d for d in quiet if d['population']==pop];neural=[d for d in group if d['parameters']>0];passing=[d for d in neural if d['summary']['qualified']==d['summary']['cases']]
        previous=[d for d in passing if d['identifier']in ['C53-num6','v2-num6','v3-legacy']]
        for source,color,style,label,z in [(frontier(passing),'#168b8d','-','Evaluated neural/hybrid frontier',2),(frontier(previous),'#8064a2','--','Matched prior controls',3)]:
            ax.plot([d['summary']['median_ms']for d in source],[100*d['summary']['worst_pressure_relative_l2']for d in source],style,color=color,lw=2.5,zorder=z,label=label)
        for d in neural:
            s=d['summary'];good=s['qualified']==s['cases'];x=s['median_ms'];y=max(100*s['worst_pressure_relative_l2'],1e-13);color='#8064a2'if d['identifier'].startswith(('C53','v2','v3'))else'#168b8d'
            ax.scatter(x,y,s=55,marker='o'if good else'X',edgecolors=color,facecolors='white'if d['initialization_control']else color,zorder=4)
            ax.annotate(str(number[d['identifier']]),(x,y),xytext=(4,5 if number[d['identifier']]%2 else-11),textcoords='offset points',fontsize=8,color=color)
        ax.set_xscale('log');ax.set_yscale('log');ax.grid(True,which='both',alpha=.14);ax.set_title(title);ax.set_xlabel('Median complete pressure-policy milliseconds')
        inset=grid[1,list(axes).index(ax)]
        for d in group:
            s=d['summary'];numeric=d['parameters']==0
            inset.scatter(s['median_ms'],max(100*s['worst_pressure_relative_l2'],1e-13),s=30,marker='*'if d['method']=='spectral'else'D'if numeric else'o',color='#344f5b'if numeric else'#168b8d')
            if d['method']=='spectral':inset.annotate('spectral',(s['median_ms'],max(100*s['worst_pressure_relative_l2'],1e-13)),xytext=(4,3),textcoords='offset points',fontsize=7)
        inset.set_xscale('log');inset.set_yscale('log');inset.grid(True,alpha=.12);inset.tick_params(labelsize=8);inset.set_title('Same population: all methods, including numerical',fontsize=10);inset.set_xlabel('Median milliseconds',fontsize=9)
    axes[0].set_ylabel('Worst mean-free pressure relative L2 error (%)');axes[0].legend(loc='lower left',fontsize=7)
    fig.suptitle('Accuracy and inference cost on two separate matched populations',fontsize=15)
    fig.text(.78,.84,'Point identifiers',weight='bold');fig.text(.78,.80,'\n'.join(f'{number[key]:2d} {key}'for key in ids),va='top',family='monospace',fontsize=8.5,linespacing=1.8)
    grid[1,0].set_ylabel('Worst pressure error (%)',fontsize=9)
    fig.text(.07,.04,'Both physical gates must pass every request for a point to enter a frontier. X marks a rejected policy; open circles are the untrained initialization control.\nCPU 23, one thread; one complete warmup and three repeats per request. Conversions, stopping and a common independent returned-field gate count.\nLower panels retain numerical controls. Zero reference-relative differences, if any, are drawn at 1e-13%; this does not imply zero equation error.',fontsize=9)
    save(fig,'accuracy-cost')

    blind={p.stem:data(p)for p in sorted((BASE/'blind-evaluation').glob('*.json'))}
    chosen=data(DOCS/'selection.json');bands=[(1,4),(5,9),(10,15)];models=[('v5-initial-gcr','Untrained'),('v5-raw-gcr','Raw-trained'),('v5-multilevel-gcr','Post-coarse'),('v5-spectrum-gcr','Post-coarse + spectra'),('numerical-gcr','Numerical GCR')]
    fig,axes=plt.subplots(1,2,figsize=(14,5.5));fig.subplots_adjust(bottom=.27,wspace=.25)
    colors=['#9aadb4','#8064a2','#247b9c','#168b8d','#505e68']
    for i,((key,label),color)in enumerate(zip(models,colors)):
        work=[];errors=[]
        for band in bands:
            group=[r for r in blind[key]['rows']if r['population']=='v5-spectrum-blind'and tuple(r['mode_band'])==band]
            if len(group)!=8:raise ValueError('Eight equal spatial patterns per band required')
            work.append(statistics.median(r['coarse_calls']for r in group));errors.append(max(r['pressure_relative_l2']for r in group))
        axes[0].bar(np.arange(3)+(i-2)*.15,work,width=.14,color=color,label=label)
        axes[1].plot(range(3),errors,'o-',color=color,label=label)
    for ax in axes:ax.set_xticks(range(3),['1–4','5–9','10–15']);ax.set_xlabel('Maximum axis-mode band');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Median hierarchy cycles to both gates');axes[1].set_ylabel('Worst mean-free pressure relative L2');axes[1].set_yscale('log')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.08),fontsize=9)
    fig.suptitle('Fresh spatial-band generalization: training work and pressure error',fontsize=14)
    fig.text(.07,.015,'Four reserved cube/rectangle shapes, two independent six-mode pressure patterns per band. All cases pass both gates. These are pressure equations, not evolved fluid trajectories.',fontsize=8.5)
    save(fig,'spatial-bands')

    reservation=data(DOCS/'scene-reservation.json');roll=[r for r in reservation['rows']if r['population']=='rollout-blind']
    methods=[('tiny-fresh','Tiny, fresh'),('tiny-tail4','Tiny, newest4'),('tiny-recycled','Tiny, useful4'),('numerical-fresh','Numerical, fresh'),('numerical-recycled','Numerical, useful4')]
    fig,ax=plt.subplots(figsize=(12,6));fig.subplots_adjust(bottom=.25)
    for i,(key,label)in enumerate(methods):
        values=[]
        for row in roll:
            d=data(BASE/'rollouts-blind'/(row['trajectory_id']+'-'+key)/'receipt.json')
            if d['status']!='completed':raise ValueError('Rejected rollout cannot be presented as full work')
            values.append(sum(c['work']['coarse_calls']for c in d['callbacks']))
        bars=ax.bar(np.arange(3)+(i-2)*.15,values,width=.14,label=label,color=['#247b9c','#8064a2','#168b8d','#aab8bf','#566b77'][i])
        ax.bar_label(bars,padding=3,fontsize=8)
    ax.set_xticks(range(3),[' × '.join(map(str,r['scene']['counts']))for r in roll]);ax.set_ylabel('Total hierarchy cycles over32 accepted native steps');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('Useful-direction retention on independently advanced blind scenes');ax.legend(loc='upper left',fontsize=8,ncol=2)
    fig.text(.08,.07,'Same initial scene and forcing, existing pressure warm starts, dt=1/60 s. Every track advances all32 steps.\nNewest4 retains the latest solver rays; useful4 retains directions by their pressure contribution. Work reduction alone is not a speedup.\nFour directions store sixteen MiB of direction/image fields at64³; ordinary solver workspace is additional.',fontsize=9)
    save(fig,'recycling-work')

    timing=data(DOCS/'timing-reservation.json');methods=[('spectral','Spectral'),('numerical-optimized','Numerical, optimized'),('numerical-recycled','Numerical, recycled4'),
        ('selected-optimized','Tiny, optimized'),('selected-recycled','Tiny, optimized + reuse4'),('selected-original','Tiny, original execution'),('legacy-v3','Previous full-model hybrid'),('selected-duplicate-audit','Tiny + duplicate callback audit')]
    fig,axes=plt.subplots(1,3,figsize=(17,6.5),sharey=True);fig.subplots_adjust(left=.22,bottom=.23,top=.83,wspace=.12)
    for ax,row in zip(axes,timing['rows']):
        for i,(key,label)in enumerate(methods):
            paths=[BASE/'quiet-native'/(row['trajectory_id']+'-'+key+f'-r{rep}')/'receipt.json'for rep in range(3)]
            if not all(p.exists()for p in paths):
                ax.text(.03,i,'Not timed on this grid',transform=ax.get_yaxis_transform(),va='center',fontsize=7,color='#7e8a92');continue
            ds=[data(p)for p in paths];medians=[statistics.median(f['frame_seconds']for f in d['frames'])for d in ds];value=statistics.median(medians);good=all(d['status']=='completed'for d in ds)
            ax.barh(i,value,xerr=[[value-min(medians)],[max(medians)-value]],color='#168b8d'if good else'#c58b70',hatch=None if good else'///',alpha=.85,error_kw=dict(capsize=3,linewidth=1))
            ax.text(max(medians)*1.04,i,f'{value:.3f} s'+(' rejected'if not good else''),va='center',fontsize=7)
        ax.set_title(' × '.join(map(str,row['scene']['counts']))+f"\n{row['scene']['steps']} native steps per repeat");ax.set_xlabel('Median headless fluid-step seconds');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.margins(x=.35)
    axes[0].set_yticks(range(len(methods)),[label for _,label in methods]);axes[0].invert_yaxis();fig.suptitle('Final native-fluid cost: all preparation, pressure exchange, gates and projection',fontsize=14)
    fig.text(.07,.075,'Three serial repeats, CPU 23 and one thread; bars are medians of per-run medians and whiskers their observed range.\nNo GUI, labels or state exports. Model loading/setup is separate. The128³ result is a four-step prefix, not a32-step qualification.\nHatching marks rejected attempts. The duplicate-audit comparison is reserved only on128³.',fontsize=9)
    save(fig,'native-fluid-time')

    fig,axes=plt.subplots(1,3,figsize=(15,5));fig.subplots_adjust(bottom=.25,top=.82,wspace=.3)
    for i,row in enumerate(roll):
        label='×'.join(map(str,row['scene']['counts']));color=['#247b9c','#8064a2','#168b8d'][i]
        for key,style in [('tiny-fresh','-'),('tiny-recycled','--')]:
            d=data(BASE/'drift-blind'/(row['trajectory_id']+'-'+key+'.json'))
            for ax,field in zip(axes,['pressure','velocity','tracer']):
                ax.plot([r['time_s']for r in d['rows']],[max(r[field]['relative_l2']or 0,1e-12)for r in d['rows']],style,color=color,lw=1.3,label=label+(' reuse4'if key=='tiny-recycled'else' fresh'))
    for ax,field in zip(axes,['pressure','velocity','tracer']):ax.set_title(field.capitalize());ax.set_yscale('log');ax.grid(True,which='both',alpha=.15);ax.set_xlabel('Simulated physical seconds')
    axes[0].set_ylabel('Trajectory relative L2 difference');axes[2].legend(fontsize=7,loc='best');fig.suptitle('Short blind trajectory drift against independent spectral tracks',fontsize=14)
    fig.text(.07,.07,'All32 steps accepted at dt=1/60 s, covering0.533 simulated seconds. Pressure gauges are removed; velocities retain all staggered faces.\nSolid=fresh solves; dashed=useful-direction recycling. Zeros are shown at1e-12. Application drift limits remain unset.',fontsize=9)
    save(fig,'blind-drift')
    write(DOCS/'figure-evidence.json',dict(status='generated_from_retained_results',names=names,inputs=inputs,artifacts=artifacts,source_sha256=sha(__file__),matplotlib_version=matplotlib.__version__,promotion=False))
    print('Generated five scientific figures in SVG, PNG and PDF',flush=True)
if __name__=='__main__':build()
