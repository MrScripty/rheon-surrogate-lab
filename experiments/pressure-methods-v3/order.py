"""Retain the original 1600-small sample stream; balance declared larger tiers."""
import hashlib
import math
import numpy as np

STATS = {}

def order(parents,config):
    training=config['training'];updates=training['updates']
    if training['batch_size']!=1 or updates not in (1600,2400): raise ValueError('Declared replay budget required')
    if any(p['entry']['split']!='train' for p in parents): raise ValueError('Training-only sampling required')
    small=[i for i,p in enumerate(parents) if max(p['counts_xyz'])<=16]
    tiers={n:[i for i,p in enumerate(parents) if (16 if max(p['counts_xyz'])<=16 else 32 if max(p['counts_xyz'])<=32 else 64 if max(p['counts_xyz'])<=64 else 128)==n] for n in (32,64,128)}
    tiers={n:indices for n,indices in tiers.items() if indices}
    if not small or len(small)!=16: raise ValueError('Exact original 16 small parents required')
    rng=np.random.Generator(np.random.PCG64(training['sampling_seed']))
    other=np.random.Generator(np.random.PCG64(training['sampling_seed']+1))
    def choose(indices,generator):
        parent=indices[int(generator.integers(len(indices)))];return parent,int(generator.integers(len(parents[parent]['frames'])))
    small_stream=[choose(small,rng) for _ in range(1600)]
    result=[];large_count=0
    if updates==1600:
        result=[[item] for item in small_stream]
    else:
        if not tiers: raise ValueError('Larger replay tiers required')
        names=sorted(tiers)
        for index,item in enumerate(small_stream):
            result.append([item])
            if index%2==1:
                tier=names[large_count%len(names)];large_count+=1
                result.append([choose(tiers[tier],other)])
    result=np.asarray(result,dtype='<i8')
    if result.shape!=(updates,1,2): raise ValueError('Exact update count required')
    exposures={};parent_exposures=[0]*len(parents);frame_exposures={};cells=0
    for ((p,f),) in result:
        p=int(p);f=int(f);n=max(parents[p]['counts_xyz']);tier=16 if n<=16 else 32 if n<=32 else 64 if n<=64 else 128
        exposures[str(tier)]=exposures.get(str(tier),0)+1;parent_exposures[p]+=1
        key=f'{parents[p]["trajectory_id"]}:{parents[p]["frames"][f]["index"]}';frame_exposures[key]=frame_exposures.get(key,0)+1
        cells+=math.prod(parents[p]['counts_xyz'])
    STATS.clear();STATS.update(tier_exposures=exposures,parent_exposures=parent_exposures,frame_exposures=frame_exposures,
        updates=updates,sampled_cells=cells,small_subsequence_sha256=hashlib.sha256(np.asarray(small_stream,dtype='<i8').tobytes()).hexdigest(),
        sampling='1600 original small samples plus 800 larger samples, balanced across present size tiers. Report actual forward cells separately; no equal-cell-budget claim.')
    return result
