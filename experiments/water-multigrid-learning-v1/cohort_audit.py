"""Audit the prepared real-simulation cohort without training or test inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

LAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/water-conditioning-v1/training'))
import data
import model
import policy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();model.configure('cpu');reservation,parents=data.read_view(args.packet)
    train=[p for p in parents if p['entry']['split']=='train'];records=[]
    for parent in train:
        pressure_delta=parent['arrays']['reference']-parent['arrays']['base']
        force=parent['scene']['force'];counts=parent['counts_xyz'];h=parent['cell_size_m']
        identity=hashlib.sha256(policy.canonical({k:parent['scene'][k] for k in ('counts_xyz','cells_per_metre','force')})).hexdigest()
        records.append({'trajectory_id':parent['entry']['trajectory_id'],'setup_sha256':identity,
            'counts_xyz':counts,'cell_size_m':h,'extents_m':[n*h for n in counts],
            'native_frames':[f['index'] for f in parent['frames']],'force':force,
            'correction_rms_pa':float((pressure_delta**2).mean()**.5),
            'states_differ':len(pressure_delta)>1 and bool((parent['arrays']['reference'][0]!=parent['arrays']['reference'][-1]).any())})
    if not train or len({r['setup_sha256'] for r in records})!=len(records):raise ValueError('distinct real training setups required')
    cells=sum(p['arrays']['base'].size for p in train)
    correction_rms=math.sqrt(math.fsum(float(((p['arrays']['reference']-p['arrays']['base'])**2).sum()) for p in train)/cells)
    result={'schema':'rheon-prepared-cohort-audit-v1','training_simulations':len(train),
        'training_states':sum(len(p['frames']) for p in train),'distinct_actual_setups':len({r['setup_sha256'] for r in records}),
        'training_cell_states':cells,'global_training_correction_rms_pa':correction_rms,
        'normalization':data.fit_normalization(parents),'records':records,
        'packet_splits_sha256':hashlib.sha256((args.packet/'splits.json').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'prepared train cohort only; no optimizer, reserved-test inference or new CFD generation'}
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('training_simulations','training_states','distinct_actual_setups','global_training_correction_rms_pa')}),flush=True)


if __name__=='__main__':main()
