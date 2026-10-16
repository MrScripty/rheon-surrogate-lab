"""Prepare train-only late-residual diagnostics without changing either fit."""
import argparse
import fcntl
from pathlib import Path
import statistics
import sys
import time

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/water-shape-repeatability-v1'))
from execution import command,read,utc,write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',type=Path,required=True)
    parser.add_argument('--reports',type=Path,required=True)
    args=parser.parse_args()
    sequence=read(args.sequence)
    root=LAB/sequence['root']
    output=root/'prepared-late-residuals'
    project=LAB/sequence['reference']['project']
    artifacts=LAB/sequence['reference']['artifacts']
    started=time.monotonic()
    with (root/'cpu-qualification.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        timing=command(['taskset','-c','1-23',sys.executable,
            LAB/'experiments/water-stability-learning-v1/learned_iteration.py',
            '--project',project,'--artifacts',artifacts,'--packet',LAB/sequence['training_packet'],
            '--population','train','--proposals','8','--single-evaluation','--output',output],
            root,'late-residual-predictions')
        sys.path.insert(0,str(LAB/'experiments/water-residual-learning-v1'))
        import audit
        audit.initialize(project,artifacts,[23])
        import numpy as np
        receipt=read(output/'receipt.json')
        _,parents=audit.DATA.read_view(LAB/sequence['training_packet'])
        records=[]
        with np.load(output/'predictions.npz',allow_pickle=False) as arrays:
            for index,case in enumerate(receipt['cases']):
                parent=parents[case['parent']]
                if parent['entry']['split']!='train':
                    raise ValueError('late preparation must use training parents only')
                sample=audit.DATA.sample(parent,case['frame'])
                pressure=audit.TORCH.from_numpy(arrays[f'case_{index:03d}']).reshape_as(sample['base'])
                initial=sample['native_input_residual']
                residual=initial-audit.MODEL.full_operator(pressure-sample['base'],sample['cell_size_m'],sample['rho'])
                defect=sample['rhs']-audit.MODEL.full_operator(sample['reference'],sample['cell_size_m'],sample['rho'])
                norm=audit.vector_norm(residual)
                records.append({'trajectory_id':case['trajectory_id'],'frame':case['frame'],
                    'counts_xyz':case['counts_xyz'],'network_calls':8,
                    'initial_residual_l2':audit.vector_norm(initial),'late_residual_l2':norm,
                    'remaining_pressure_correction_rms_pa':float((sample['reference']-pressure).square().mean().sqrt()),
                    'teacher_defect_over_late_residual':audit.vector_norm(defect)/max(norm,1e-30),
                    'late_over_initial_residual':norm/max(audit.vector_norm(initial),1e-30)})
        result={'recorded_at_utc':utc(),'status':'prepared_not_trained','training_states':len(records),
            'training_parents':len({r['trajectory_id'] for r in records}),'records':records,
            'median_late_over_initial_residual':statistics.median(r['late_over_initial_residual'] for r in records),
            'maximum_teacher_defect_over_late_residual':max(r['teacher_defect_over_late_residual'] for r in records),
            'weights_sha256':receipt['weights_sha256'],'used_cpus':receipt['used_cpus'],
            'cuda_initialized':receipt['runtime']['cuda_initialized'],'optimizer_updates':0,
            'prediction_command':timing,'elapsed_seconds':time.monotonic()-started,
            'scope':'fixed eight learned corrections on training snapshots; no new CFD simulations, final examples, new fit or inference-policy change. Teacher label defect is measured before any future late-residual training.'}
        write(args.reports/'prepared-late-residuals.json',result)
        print({k:result[k] for k in ('training_states','training_parents','median_late_over_initial_residual',
                                    'maximum_teacher_defect_over_late_residual','elapsed_seconds')})


if __name__=='__main__':
    main()
