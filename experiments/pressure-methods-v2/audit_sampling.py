"""Audit recorded optimizer work and paired replay streams across actual fits."""
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha
import shape_order

def audit():
    base=ROOT/'runs/pressure-methods-v2';docs=ROOT/'docs/plans/pressure-methods-v2'
    manifest=read(base/'shape-training-view-01/manifest.json')
    parents=[dict(p,entry=dict(split=p['split'])) for p in manifest['parents']]
    orders={};configs={};rows=[]
    arms=['v2-small16','v2-small24','v2-mixed16','v2-retain24','v2-headbase','v2-headshape','v2-num4','v2-num6']
    for arm in arms:
        project=base/(arm+'-project-02');config=read(project/'config.json');configs[arm]=config
        selected=parents[:16] if arm in ['v2-small16','v2-small24'] else parents
        sequence=shape_order.order(selected,config);orders[arm]=sequence
        history_path=base/(arm+'-study-02/restore/history.json');history=read(history_path)
        observed=history['current_feedback']['sampling'];replayed=shape_order.STATS
        if observed!=replayed:raise ValueError('Recorded training sampling differs from deterministic replay')
        qualification=read(docs/(arm+'-qualification-02.json'))
        digest=hashlib.sha256(sequence.tobytes()).hexdigest()
        if digest!=qualification['matched_control']['sample_order_sha256']:
            raise ValueError('Actual replay differs from the admitted qualification')
        work=history['optimizer_work']
        if work['network_forwards']!=2*len(sequence) or work['forward_cells']!=2*observed['sampled_cells']:
            raise ValueError('Actual optimizer-loss network work differs from recorded sampled fields')
        rows.append(dict(arm=arm,updates=len(sequence),sampling=observed,optimizer_work=work,
            order_sha256=digest,history_sha256=sha(history_path),sampler_source_sha256=sha(project/'mixed_order.py')))
    mixed=orders['v2-mixed16'];retained=orders['v2-retain24'];small=orders['v2-small16']
    large_mixed=mixed[mixed[:,0,0]>=16];large_retained=retained[retained[:,0,0]>=16]
    small_retained=retained[retained[:,0,0]<16]
    if not np.array_equal(large_mixed,large_retained) or not np.array_equal(small_retained,small):
        raise ValueError('Counterfactual streams changed instead of preserving exposures')
    def rate(config,update):
        result=config['training']['learning_rate']
        for change in config['training']['schedule']:
            if update>=change['update']:result=change['learning_rate']
        return result
    def rates(arm,mask):
        order=orders[arm];positions=np.nonzero(mask(order))[0]+1
        return [rate(configs[arm],int(position)) for position in positions]
    small_rates=rates('v2-small16',lambda o:np.ones(len(o),dtype=bool))
    retained_small_rates=rates('v2-retain24',lambda o:o[:,0,0]<16)
    mixed_large_rates=rates('v2-mixed16',lambda o:o[:,0,0]>=16)
    retained_large_rates=rates('v2-retain24',lambda o:o[:,0,0]>=16)
    if small_rates!=retained_small_rates or mixed_large_rates!=retained_large_rates:
        raise ValueError('Per-stream learning-rate alignment changed')
    write(docs/'sampling-audit.json',dict(status='passed',rows=rows,source_sha256=sha(__file__),
        training_manifest_sha256=sha(base/'shape-training-view-01/manifest.json'),
        exact_large_sample_stream_preserved_updates=len(large_mixed),
        exact_small_control_stream_preserved_updates=len(small),
        per_occurrence_fine_learning_rates_match_both_controls=True,
        optimizer_state_scope='Additional updates intentionally change Adam moments; matching streams/rates is not a claim of identical optimizer trajectories',
        work_scope='Actual optimizer-loss neural forwards and fine-grid cell visits; numerical-stage and diagnostic work are separate'))
    print('Audited all eight actual optimizer-work logs and matched small/large streams',flush=True)

if __name__=='__main__':audit()
