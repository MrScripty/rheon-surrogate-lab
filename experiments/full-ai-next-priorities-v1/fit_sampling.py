"""Paired 800 small / 400 larger draws with frozen train-only tail weights."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

STATS={}

def order(parents,config):
    spec=json.loads(Path(__file__).with_name('fit-sampling.json').read_text())
    if config['training']['updates']!=1200 or config['training']['batch_size']!=1:
        raise ValueError('Registered 1200 serial updates required')
    if any(p['entry']['split']!='train'for p in parents):raise ValueError('Train-only sampler required')
    tiers={}
    for i,parent in enumerate(parents):
        n=max(parent['counts_xyz']);band=16 if n<=16 else 32 if n<=32 else 64 if n<=64 else 128
        tiers.setdefault(band,[]).append(i)
    if 16 not in tiers or len(tiers[16])!=16:raise ValueError('Original 16 small parents required')
    weights={(r['parent'],r['frame']):r['multiplier']for r in spec['ranking_records']}
    choices={};probabilities={}
    for band,indices in tiers.items():
        pairs=[];mass=[]
        for index in indices:
            parent=parents[index]
            for frame in range(len(parent['frames'])):
                pairs.append((index,frame))
                multiplier=weights[(index,frame)]if spec['emphasize_tail']else 1.
                if not 1<=multiplier<=4:raise ValueError('Registered multiplier bounds failed')
                mass.append(multiplier/len(indices)/len(parent['frames']))
        mass=np.asarray(mass,dtype='<f8');mass/=mass.sum()
        choices[band]=pairs;probabilities[band]=np.cumsum(mass)
    rng=np.random.Generator(np.random.PCG64(config['training']['sampling_seed']))
    large=sorted(n for n in tiers if n>16)
    if not large:raise ValueError('Larger tiers required')
    result=[];exposures={};frames={};cells=0
    for index in range(1200):
        band=16 if index%3!=2 else large[(index//3)%len(large)]
        uniform=float(rng.random());position=min(int(np.searchsorted(probabilities[band],uniform,side='right')),len(choices[band])-1)
        parent,frame=choices[band][position];result.append([(parent,frame)])
        exposures[str(band)]=exposures.get(str(band),0)+1
        key=f'{parents[parent]["trajectory_id"]}:{parents[parent]["frames"][frame]["index"]}'
        frames[key]=frames.get(key,0)+1;cells+=math.prod(parents[parent]['counts_xyz'])
    result=np.asarray(result,dtype='<i8')
    if exposures['16']!=800 or sum(exposures.values())!=1200:raise ValueError('Spatial tier proportions changed')
    STATS.clear();STATS.update(updates=1200,tier_exposures=exposures,frame_exposures=frames,
        sampled_cells=cells,sample_order_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
        emphasize_tail=spec['emphasize_tail'],ranking_sha256=spec['ranking_sha256'],
        sampling='800 small draws and 400 round-robin larger draws; same uniform random stream across paired arms')
    return result
