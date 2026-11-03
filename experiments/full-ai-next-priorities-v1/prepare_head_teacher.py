"""Generate offline low-mode labels for a possible neural-head calibration follow-up.

This file is a teacher/data generator. Candidate inference must never import it.
No optimizer updates, physical trajectories or reserved states are used here.
"""
import hashlib
import itertools
import math
from pathlib import Path
import numpy as np
from study_support import HERE,BASE,read,write,sha


if __name__=='__main__':
    output=BASE/'head-teacher-bank-01'
    if output.exists():raise ValueError('Fresh offline teacher bank required')
    output.mkdir();seed=2026101941;rng=np.random.Generator(np.random.PCG64(seed))
    small=list(itertools.product(range(5,17),repeat=3));large=set()
    while len(large)<5000:
        counts=tuple(int(n)for n in rng.integers(5,129,size=3))
        if max(counts)>16:large.add(counts)
    counts=np.asarray(small+sorted(large),dtype='<i4')
    modes=np.asarray([m for m in itertools.product(range(4),repeat=3)if sum(m)],dtype='<i4')
    eigenvalues=(2*(1-np.cos(math.pi*modes[None,:,:]/counts[:,None,:]))).sum(axis=2)
    gains=1/eigenvalues;cap=counts.max(axis=1).astype('f8')**2/2
    log_fraction=np.log(gains/cap[:,None])
    if not np.isfinite(log_fraction).all()or not ((gains>0)&(gains<cap[:,None])).all():
        raise ValueError('Bounded teacher gain contract failed')
    split=[]
    for row in counts:
        key=f'{seed}:{row[0]}:{row[1]}:{row[2]}'.encode()
        split.append(int(int.from_bytes(hashlib.sha256(key).digest()[:8],'little')%10==0))
    arrays=dict(counts_zyx=counts,features=np.log(counts.astype('f8')/16).astype('<f4'),
        log_gain_fraction=log_fraction.astype('<f4'),teacher_gain=gains.astype('<f8'),
        mode_indices_zyx=modes,tier_small=(counts.max(axis=1)<=16).astype('u1'),split=np.asarray(split,dtype='u1'))
    path=output/'teacher.npz';np.savez_compressed(path,**arrays)
    manifest=dict(status='generated_offline_teacher_only',seed=seed,source_sha256=sha(__file__),
        archive_sha256=sha(path),archive_bytes=path.stat().st_size,geometry_count=len(counts),
        original_primary_registration_changed=False,optimizer_updates=0,
        counts_order='Z,Y,X; matches pressure tensor dimensions and learned shape features',
        features='log(Nz/16),log(Ny/16),log(Nx/16)',teacher='Neumann cosine low-mode gain: 1/[sum_axis 2(1-cos(pi*k/N))]; exclude DC',
        label='log(teacher_gain / (max_axis_cells^2/2))',
        numerical_inverse_used_only_offline=True,candidate_inference_must_remain_neural=True,
        split='Hash whole geometry: 90% teacher fitting /10% teacher development; unrelated to physical scene splits',
        counts={f'{phase}_{tier}':int(((arrays['split']==phase_index)&(arrays['tier_small']==tier_index)).sum())
            for phase,phase_index in [('train',0),('development',1)]for tier,tier_index in [('small',1),('large',0)]},
        inventory={name:dict(shape=list(value.shape),dtype=value.dtype.str,sha256=hashlib.sha256(value.tobytes()).hexdigest())for name,value in arrays.items()},
        physical_reference_fields_used=False,reserved_scenes_consumed=False,model_promotion=False)
    write(output/'manifest.json',manifest)
    print('Generated offline low-mode teacher labels for',len(counts),'geometries; zero fitting updates',flush=True)
