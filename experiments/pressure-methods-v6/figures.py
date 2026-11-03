"""Standalone scientific figures; refuse to render before serial timing drains."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/rheon-v6-mpl-cache')
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v6_support import ROOT,HERE,BASE,DOCS,read,write,sha
def build():
    for name in ['quiet-development-batch-03','final-native-timing-batch-01']:
        d=read(BASE/name/'execution.json')
        if d['status']!='completed'or not d['owned_child_reaped']:raise ValueError('Finish and drain every serial timing task before plotting')
    output=DOCS/'figures';output.mkdir(exist_ok=True);inputs={};artifacts={};names=[]
    def data(p):inputs[str(p.relative_to(ROOT))]=sha(p);return read(p)
    def save(fig,name):
        names.append(name)
        for suffix in ['svg','png','pdf']:
            p=output/(name+'.'+suffix);options=dict(dpi=160,bbox_inches='tight')
            if suffix=='svg':options['metadata']={'Date':None}
            if suffix=='pdf':options['metadata']={'CreationDate':None,'ModDate':None}
            fig.savefig(p,**options);artifacts[str(p.relative_to(ROOT))]=sha(p)
        plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'svg.hashsalt':'rheon-pressure-v6','savefig.facecolor':'white'})
    selection=data(DOCS/'selection.json');calibration=data(DOCS/'calibration-points-03.json');selected={x['point']['identifier']for x in selection['selected']}
    timing=[data(BASE/'quiet-development'/(p['identifier']+'.json'))for p in calibration['points']]
    if len(timing)!=19 or len({d['population_sha256']for d in timing})!=1:raise ValueError('Nineteen matched development timing points required')
    number={d['identifier']:i+1 for i,d in enumerate(timing)}
    def frontier(group):
        best=float('inf');result=[]
        for d in sorted(group,key=lambda d:(d['summary']['median_ms'],d['summary']['worst_pressure_relative_l2'])):
            if d['summary']['qualified']!=d['summary']['cases']:continue
            error=d['summary']['worst_pressure_relative_l2']
            if error<best:result.append(d);best=error
        return result
    fig,axes=plt.subplots(1,2,figsize=(17,8));fig.subplots_adjust(right=.75,bottom=.22,wspace=.27,top=.86)
    for ax,group,title in [(axes[0],[d for d in timing if not d['point']['numerical']],'Learned / conditional policies'),(axes[1],timing,'All policies, including numerical controls')]:
        for d in group:
            s=d['summary'];good=s['qualified']==s['cases'];numeric=d['point']['numerical'];inactive=not any(r['network_calls']for r in d['rows']);color='#526b78'if numeric else'#188b8e'
            ax.scatter(s['median_ms'],max(100*s['worst_pressure_relative_l2'],1e-13),marker='X'if not good else'*'if d['point']['method']=='spectral'else'D'if numeric else'o',s=75 if d['identifier']in selected else 45,
              edgecolors=color,facecolors='white'if inactive and not numeric else color,zorder=3)
            if ax is axes[0]or numeric:
                ax.annotate(str(number[d['identifier']]),(s['median_ms'],max(100*s['worst_pressure_relative_l2'],1e-13)),xytext=(4,5 if number[d['identifier']]%2 else-10),textcoords='offset points',fontsize=8)
        eligible=[d for d in group if any(r['network_calls']for r in d['rows'])]if ax is axes[0]else group
        f=frontier(eligible);ax.plot([d['summary']['median_ms']for d in f],[max(100*d['summary']['worst_pressure_relative_l2'],1e-13)for d in f],color='#c17745',lw=2,marker='s',markersize=12,markerfacecolor='none',zorder=4)
        ax.set_xscale('log');ax.set_yscale('log');ax.grid(True,which='both',alpha=.14);ax.set_xlabel('Median complete pressure-policy milliseconds');ax.set_title(title)
    axes[0].set_ylabel('Worst pressure relative L2 error (%)')
    fig.suptitle('Matched cost and accuracy on 20 consumed native development requests',fontsize=15)
    fig.text(.77,.83,'Point identifiers',weight='bold');fig.text(.77,.79,'\n'.join(f'{number[d["identifier"]]:2d} {d["identifier"]}'for d in timing),family='monospace',fontsize=8,va='top',linespacing=1.5)
    fig.text(.06,.06,'Same 20 cold/warm requests on 32³–128³ and rectangular grids. CPU 23, one thread; whole-population warmup and three repeats per request.\nAll correction, conversion, stopping and independent returned-field gate work counts. Orange outlines mark qualified frontier points.\nOpen learned markers mean no CNN calls and are excluded from the active-neural frontier on the left. The right panel labels numerical controls only.\nZero reference differences use a 1e-13% display floor. These are matched policies, not the complete historical cohort catalog.',fontsize=8.5)
    save(fig,'development-accuracy-cost')

    blind={p['identifier']:data(BASE/'blind-evaluation'/(p['identifier']+'.json'))for p in selection['blind_points']}
    if len({d['population_sha256']for d in blind.values()})!=1:raise ValueError('Blind input populations differ')
    ids=list(blind);fig,axes=plt.subplots(1,2,figsize=(16,7));fig.subplots_adjust(bottom=.35,wspace=.25)
    xs=np.arange(len(ids))
    for offset,pop,color in [(-.18,'blind','#247b9c'),(.18,'large-blind','#188b8e')]:
        work=[];errors=[];labels=[]
        for key in ids:
            rows=[r for r in blind[key]['rows']if r['population']==pop];work.append(statistics.median(r['coarse_calls']for r in rows));errors.append(max(100*r['pressure_relative_l2']for r in rows));labels.append(f'{sum(r["qualified"]for r in rows)}/{len(rows)}')
        bars=axes[0].bar(xs+offset,work,width=.34,color=color,label='64³–128³ / rectangles'if pop=='blind'else'256³ / large rectangle')
        axes[0].bar_label(bars,labels,padding=3,fontsize=7);axes[1].scatter(xs+offset,np.maximum(errors,1e-13),color=color,s=45)
    for ax in axes:ax.set_xticks(xs,ids,rotation=55,ha='right',fontsize=8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Median global calls\n(hierarchy cycles / direct spectral solves)');axes[0].legend(fontsize=8);axes[1].set_ylabel('Worst same-grid pressure relative L2 (%)');axes[1].set_yscale('log')
    fig.suptitle('Four trained contrasts and frozen controls on 14 fresh held-out equations',fontsize=14)
    fig.text(.05,.025,'Eight requests from four smaller scenes and six requests from three large scenes; two snapshots per scene. Labels above bars show gates passed / requests.\nLarge scenes include two 256³ cubes at 16/32 cells per metre and one 256×193×129 rectangle. Errors on rejected requests remain visible.\nA global call is one hierarchy cycle or one direct spectral solve; these are not equal-cost units. Each error uses its own discrete-grid reference.\nNo application drift or continuum-accuracy claim is implied.',fontsize=9)
    save(fig,'blind-work-error')

    reservation=data(DOCS/'timing-reservation.json');native={};method_labels={'selected-fresh':'Selected, fresh','selected-fixed4':'Selected, fixed reuse4','selected-adaptive4':'Selected, adaptive reuse4','baseline-v5':'Previous tiny model','numerical':'Selected numerical control','spectral':'Spectral'}
    fig,axes=plt.subplots(1,3,figsize=(17,6));fig.subplots_adjust(left=.19,bottom=.24,top=.82,wspace=.15)
    for ax,row in zip(axes,reservation['rows']):
        for i,method in enumerate(reservation['methods']):
            label=method['label'];ds=[data(BASE/'quiet-native'/(row['trajectory_id']+'-'+label+f'-r{rep}')/'receipt.json')for rep in range(3)]
            medians=[statistics.median(f['frame_seconds']for f in d['frames'])for d in ds];complete=all(d['status']=='completed'for d in ds);median=statistics.median(medians)
            native[(tuple(row['scene']['counts']),label)]=dict(median=median,min=min(medians),max=max(medians),complete=complete,receipts=ds)
            ax.barh(i,median,color='#188b8e'if complete else'#d4966c',alpha=.85,hatch=''if complete else'//')
            ax.errorbar(median,i,xerr=[[median-min(medians)],[max(medians)-median]],fmt='none',ecolor='#263d48',capsize=3,lw=1)
            ax.text(max(medians)*1.025,i,f'{median:.3f} s'+(' attempt'if not complete else''),va='center',fontsize=8)
        ax.set_yticks(range(len(reservation['methods'])),[method_labels[m['label']]for m in reservation['methods']]if ax is axes[0]else[]);ax.invert_yaxis();ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
        ax.set_xlabel('Median native fluid-step seconds');ax.set_title(' × '.join(map(str,row['scene']['counts']))+'\n'+str(row['scene']['steps'])+' steps per repeat');ax.set_xlim(right=ax.get_xlim()[1]*1.25)
    fig.suptitle('Complete native fluid-step cost, including exchange, gates and projection',fontsize=14)
    fig.text(.04,.045,'Three balanced-order serial repeats per method/grid, CPU 23 and one thread. Bars are medians of per-run frame medians; whiskers show their observed range.\nNo GUI, labels or state exports. Model loading/setup is separate. Hatched attempted cost does not establish a usable speedup.\nThe numerical control was selected using development only; its exact point is retained in the selection and timing reservation.',fontsize=9)
    save(fig,'native-fluid-time')

    long=[r for r in data(DOCS/'scene-reservation.json')['rows']if r['population']=='long-blind'];fig,axes=plt.subplots(2,3,figsize=(16,9));fig.subplots_adjust(bottom=.19,hspace=.35,wspace=.26)
    colors={'numerical':'#526b78','selected-fresh':'#247b9c','selected-fixed4':'#8064a2','selected-adaptive4':'#188b8e'}
    styles={'numerical':'-','selected-fresh':'-','selected-fixed4':'--','selected-adaptive4':':'}
    for row,axs in zip(long,axes):
        for policy in colors:
            d=data(BASE/'long-drift'/(row['trajectory_id']+'-'+policy+'.json'))
            for ax,field in zip(axs,['pressure','velocity','tracer']):
                values=[max(r[field]['relative_l2'],1e-12)if r[field]['relative_l2']is not None else np.nan for r in d['rows']]
                ax.plot([r['time_s']for r in d['rows']],values,color=colors[policy],ls=styles[policy],lw=1.6,label=policy+f' ({d["matched_accepted_steps"]} steps)')
                ax.set_yscale('log');ax.grid(True,which='both',alpha=.13);ax.set_xlabel('Simulated seconds');ax.set_title(field.capitalize())
        axs[0].set_ylabel('Relative L2 drift\n'+'×'.join(map(str,row['scene']['counts']))+' at '+str(row['scene']['cells_per_metre'])+' cells/m');axs[2].legend(fontsize=7,loc='best')
    fig.suptitle('Independent full-field trajectories against the spectral reference',fontsize=14)
    fig.text(.05,.04,'Each track requests 256 steps, dt = 1/60 s, covering 4.267 simulated seconds if every step advances. Curves stop at matched accepted prefixes.\nPressure gauges are removed; velocity retains every staggered face. Each full field is decoded and hash-verified. Zeros use a 1e-12 display floor.\nCoincident curves remain identified by line style. These are diagnostic differences; no application limits or universal safe horizon have been set.',fontsize=9)
    save(fig,'long-trajectory-drift')

    cal=data(DOCS/'reuse-calibration.json');fig,axes=plt.subplots(1,2,figsize=(14,6));fig.subplots_adjust(bottom=.26,wspace=.3)
    bands=sorted(cal['bands'],key=lambda b:b['cells']);x=np.arange(len(bands));expected=[b['baseline_iterations']*b['saved_fraction']*b['cycle_seconds_per_cell']*b['cells']for b in bands];overhead=[b['overhead_seconds_per_cell']*b['cells']for b in bands]
    saved_bars=axes[0].bar(x-.17,np.array(expected)*1000,width=.32,label='Expected saved work',color='#247b9c');overhead_bars=axes[0].bar(x+.17,np.array(overhead)*1000,width=.32,label='Measured reuse overhead',color='#8064a2')
    axes[0].bar_label(saved_bars,fmt='%.2f',padding=3,fontsize=7);axes[0].bar_label(overhead_bars,fmt='%.2f',padding=3,fontsize=7)
    axes[0].scatter(x,np.array(overhead)*1000*cal['overhead_margin'],marker='_',s=220,color='#ad5947',zorder=4,label='Required savings: overhead × 1.15')
    axes[0].set_xticks(x,['×'.join(map(str,b['counts_xyz']))for b in bands]);axes[0].set_ylabel('Development milliseconds per pressure request');axes[0].legend(fontsize=8);axes[0].set_title('Frozen development cost model')
    shapes=[tuple(r['scene']['counts'])for r in reservation['rows']]
    for i,shape in enumerate(shapes):
        fresh=native[(shape,'selected-fresh')];fixed=native[(shape,'selected-fixed4')];adaptive=native[(shape,'selected-adaptive4')]
        for off,d,color,label in [(-.17,fixed,'#8064a2','Fixed4'),(.17,adaptive,'#188b8e','Adaptive4')]:
            value=100*(fresh['median']-d['median'])/fresh['median']
            axes[1].bar(i+off,value,width=.32,color=color,label=label if i==0 else None,hatch=''if fresh['complete']and d['complete']else'//')
    axes[1].axhline(0,color='#526b78',lw=.8);axes[1].set_xticks(range(len(shapes)),['×'.join(map(str,s))for s in shapes]);axes[1].set_ylabel('Native fluid-step time reduction versus fresh (%)');axes[1].set_title('Final held-out repeated timing');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
    fig.suptitle('Does reusable solver work pay for its maintenance?',fontsize=14)
    fig.text(.05,.045,'The admission rule requires predicted savings to exceed measured overhead by 15%, plus a similar residual sketch. Calibration uses consumed development only.\nRemoving a numerical-only tail does not claim savings from earlier CNN calls. Cost extrapolation and actual runtime benefits are hypotheses tested after freezing.\nPositive bars mean lower time; negative bars mean a slowdown. Independently advanced trajectories do not provide per-request counterfactual timings.',fontsize=8.5)
    save(fig,'adaptive-reuse-cost')
    write(DOCS/'figure-evidence.json',dict(status='generated_from_retained_results',names=names,inputs=inputs,artifacts=artifacts,source_sha256=sha(__file__),matplotlib_version=matplotlib.__version__))
    print('Generated five scientific figures as SVG, PNG and PDF',flush=True)
if __name__=='__main__':build()
