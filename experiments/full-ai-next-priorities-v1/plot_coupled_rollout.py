"""Plot matched own-history errors without extending incomplete scene prefixes."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/rheon-full-ai-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    result=read(DOCS/'temporal-coupled-results.json')
    if result['status']!='completed':raise ValueError('Completed coupled comparison required')
    roots=[BASE/'temporal-decoder-evaluation-01/b2-endpoint-decoder',BASE/'temporal-coupled-evaluation-01/b2-coupled']
    sources={};curves=[]
    for directory,label in zip(roots,('Previous decoder','Coupled state updates')):
        summary_path=directory/'summary.json';summary=read(summary_path);sources[str(summary_path.relative_to(ROOT))]=sha(summary_path)
        scenes=[s for s in summary['scenes']if max(s['counts_xyz'])<=16]
        samples=[]
        for scene in scenes:
            path=directory/(scene['identifier']+'.json');raw=read(path)
            sources[str(path.relative_to(ROOT))]=sha(path)
            samples.append({r['step']:r['metrics']['relative']for r in raw['rows']if r['mode']=='own_history'})
        common=sorted(set.intersection(*(set(s)for s in samples)))
        # A scene missing a state ends the pooled curve; later partial maxima
        # cannot masquerade as maxima over the complete scene population.
        prefix=[]
        for expected,step in enumerate(common,1):
            if step!=expected:break
            prefix.append(step)
        curves.append(dict(label=label,steps=prefix,scene_count=len(samples),
            values={field:[max(s[t][field]for s in samples)for t in prefix]for field in ('pressure','equation','velocity','tracer')}))
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10})
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),sharex=True,layout='constrained')
    for axis,field in zip(axes.flat,('pressure','equation','velocity','tracer')):
        for curve in curves:
            axis.plot(curve['steps'],[max(v,1e-12)for v in curve['values'][field]],label=curve['label'])
            if len(curve['steps'])<240 and curve['steps']:
                axis.axvline(curve['steps'][-1],color='#777777',alpha=.35,linestyle=':')
        axis.axhline(.004,color='#c23434',linestyle='--',linewidth=1,label='Required limit')
        axis.set(title=field.capitalize(),yscale='log',xlim=(1,240),ylabel='Worst relative error')
        axis.grid(True,which='both',alpha=.2)
    for axis in axes[-1]:axis.set_xlabel('Physical step using predicted history')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Two-step blocks on six small-grid development scenes\nEvery intermediate output; predicted history throughout',fontsize=11)
    directory=DOCS/'figures';directory.mkdir(exist_ok=True)
    paths=[]
    for extension in ('png','svg'):
        path=directory/('temporal-coupled-rollout.'+extension);fig.savefig(path,dpi=160);paths.append(path)
    plt.close(fig)
    write(BASE/'temporal-coupled-rollout-figure-01.json',dict(status='completed',source_sha256=sha(__file__),
        result_sha256=sha(DOCS/'temporal-coupled-results.json'),source_evidence=sources,
        common_complete_scene_prefix_steps={c['label']:len(c['steps'])for c in curves},
        artifacts={str(p.relative_to(ROOT)):sha(p)for p in paths},
        display_floor=1e-12,display_floor_changes_metrics=False,candidate_CPU_latency_claim=False))
    print('Plotted every-step coupled/decoder own-history errors on the complete common scene prefixes',flush=True)


if __name__=='__main__':run()
