"""Offline eight-mode gain labels; no physical snapshots and no fitting."""
import hashlib
import itertools
import math
import numpy as np
from study_support import BASE,write,sha


if __name__=='__main__':
    output=BASE/'tail-teacher-bank-01'
    if output.exists():raise ValueError('Fresh offline tail teacher required')
    output.mkdir()
    seed=2026101941
    counts=np.asarray(list(itertools.product(range(5,17),repeat=3)),dtype='<i4')
    modes=np.asarray([row for row in itertools.product(range(8),repeat=3)if sum(row)],dtype='<i4')
    eigenvalues=(2*(1-np.cos(math.pi*modes[None,:,:]/counts[:,None,:]))).sum(2)
    gains=1/eigenvalues
    cap=counts.max(1).astype('f8')**2/2
    valid=(modes[None,:,:]<counts[:,None,:]).all(2)
    labels=np.log(gains/cap[:,None])
    split=np.asarray([int(int.from_bytes(hashlib.sha256(f'{seed}:{z}:{y}:{x}'.encode()).digest()[:8],'little')%10==0)
                      for z,y,x in counts],dtype='u1')
    if not np.isfinite(labels).all()or not ((gains>0)&(gains<cap[:,None])).all():
        raise ValueError('Bounded finite offline labels required')
    arrays=dict(counts_zyx=counts,features=np.log(counts.astype('f8')/16).astype('<f4'),
        log_gain_fraction=labels.astype('<f4'),teacher_gain=gains.astype('<f8'),valid_mode=valid.astype('u1'),
        mode_indices_zyx=modes,tier_small=np.ones(len(counts),dtype='u1'),split=split)
    path=output/'teacher.npz';np.savez_compressed(path,**arrays)
    write(output/'manifest.json',dict(status='generated_offline_teacher_only',seed=seed,source_sha256=sha(__file__),
        archive_sha256=sha(path),archive_bytes=path.stat().st_size,geometry_count=len(counts),modes_per_axis=8,
        counts_order='Z,Y,X',train_geometries=int((split==0).sum()),development_geometries=int((split==1).sum()),
        valid_modes_only_in_training_loss=True,split='Same whole-geometry hash convention as original head calibration',
        teacher='1/sum_axis[2(1-cos(pi*k/N))], DC excluded, labels are log(gain/(maxN^2/2))',
        numerical_inverse_used_only_offline=True,physical_reference_fields_used=False,reserved_scenes_consumed=False,
        optimizer_updates=0,inventory={name:dict(shape=list(value.shape),dtype=value.dtype.str,
        sha256=hashlib.sha256(value.tobytes()).hexdigest())for name,value in arrays.items()}))
    print('Generated offline eight-mode tail teacher',len(counts),'geometries; zero updates',flush=True)
