"""Independent streams preserve the exact small-control sample subsequence."""
import math
import numpy as np

STATS={}

def order(parents,config):
    training=config['training'];updates=training['updates']
    if training['batch_size']!=1:raise ValueError('Single-case sampling required')
    small=[i for i,p in enumerate(parents) if max(p['counts_xyz'])<=16]
    large=[i for i,p in enumerate(parents) if max(p['counts_xyz'])>16]
    if not small or any(p['entry']['split']!='train' for p in parents):
        raise ValueError('Small replay and train-only parents required')
    rng=np.random.Generator(np.random.PCG64(training['sampling_seed']))
    def choose(indices,generator):
        parent=indices[int(generator.integers(len(indices)))]
        return parent,int(generator.integers(len(parents[parent]['frames'])))
    if large:
        retained=updates==2400
        if updates not in (1600,2400):raise ValueError('Declared expansion update budget required')
        small_count=1600 if retained else 800
        small_stream=[choose(small,rng) for _ in range(small_count)]
        other=np.random.Generator(np.random.PCG64(training['sampling_seed']+1))
        tiers={32:[i for i in large if max(parents[i]['counts_xyz'])<=32],
               64:[i for i in large if max(parents[i]['counts_xyz'])>32]}
        result=[]
        for index,item in enumerate(small_stream):
            result.append([item])
            if not retained or index%2==1:
                tier=32 if (len(result)//(3 if retained else 2))%2==0 else 64
                result.append([choose(tiers[tier],other)])
    else:
        small_stream=[choose(small,rng) for _ in range(updates)]
        result=[[item] for item in small_stream]
    result=np.asarray(result,dtype='<i8')
    if result.shape!=(updates,1,2):raise ValueError('Exact declared update count required')
    exposures=[0]*len(parents);cells=0;tiers={}
    for ((parent,frame),) in result:
        exposures[int(parent)]+=1
        count=math.prod(parents[int(parent)]['counts_xyz']);cells+=count
        tier=16 if int(parent) in small else 32 if max(parents[int(parent)]['counts_xyz'])<=32 else 64
        tiers[str(tier)]=tiers.get(str(tier),0)+1
    import hashlib
    STATS.clear();STATS.update(tier_exposures=tiers,parent_exposures=exposures,updates=updates,
        sampled_cells=cells,small_subsequence_sha256=hashlib.sha256(np.asarray(small_stream,dtype='<i8').tobytes()).hexdigest(),
        sampling='Independent small replay stream; original exposure preserved in 2400-update expansion. Report update and cell budgets separately.')
    return result
