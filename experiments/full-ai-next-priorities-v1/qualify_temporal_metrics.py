"""Check independent native equation alignment, field sensitivity and gauge removal."""
import argparse
from pathlib import Path
import numpy as np
import torch
from study_support import ROOT,HERE,BASE,read,write,sha
from temporal_metrics import metrics,FIELDS

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh metric qualification required')
    torch.set_num_threads(1);bank=read(BASE/'temporal-development-240-01/bank.json')
    norm=read(BASE/'temporal-training-view-02/normalization.json');rows=[]
    for row in bank['trajectories']:
        receipt_path=ROOT/row['receipt'];receipt=read(receipt_path)
        for step in (8,128,240):
            with np.load(receipt_path.parent/receipt['states'][step]['path'],allow_pickle=False)as source:
                reference={name:source[name].copy()for name in (*FIELDS,'rhs')}
            exact={name:reference[name]for name in FIELDS}
            value=metrics(exact,reference,reference['rhs'],receipt['scene'],norm,0.)
            if any(v!=0. for v in value['absolute_rms'].values())or value['relative']['equation']>1e-6:
                raise ValueError('Reference identity or equation timestamp alignment failed')
            changed=dict(exact,pressure=exact['pressure']+123.)
            gauge=metrics(changed,reference,reference['rhs'],receipt['scene'],norm,0.)
            if gauge['absolute_rms']['pressure']>1e-12:raise ValueError('Pressure gauge removal failed')
            changed=dict(exact,tracer=exact['tracer']+.01)
            altered=metrics(changed,reference,reference['rhs'],receipt['scene'],norm,0.)
            if altered['criterion_pass']['tracer']or abs(altered['absolute_rms']['tracer']-.01)>1e-7:
                raise ValueError('Tracer criterion is insensitive to a physical error')
            rows.append(dict(identifier=row['identifier'],step=step,reference_equation_error=value['relative']['equation'],
                identity_field_errors_zero=True,gauge_invariance=True,tracer_perturbation_rejected=True))
    write(args.output,dict(status='passed',rows=rows,source_sha256=sha(__file__),metrics_sha256=sha(HERE/'temporal_metrics.py'),
        bank_sha256=sha(BASE/'temporal-development-240-01/bank.json'),optimizer_updates=0,reserved_scenes_consumed=False))
    print('Qualified offline full-state metrics and equation time alignment on8development scenes',flush=True)
