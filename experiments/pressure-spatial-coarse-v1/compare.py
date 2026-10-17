"""Verify matched actual coarse/control fits; never assert learned accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent/'pressure-current-policy-v1'),
               str(HERE.parent/'pressure-throughput-feedback-v1')]
from compare_arms import project
from compare_training import measured


def inventory(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as arrays:
        return {name: {'shape':list(arrays[name].shape), 'dtype':arrays[name].dtype.str,
                       'bytes':arrays[name].nbytes,
                       'sha256':hashlib.sha256(arrays[name].tobytes()).hexdigest(),
                       'nonzero':int(np.count_nonzero(arrays[name]))}
                for name in arrays.files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','control','candidate','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    root = a.root.resolve(strict=True)
    if a.output.exists():
        raise ValueError('fresh report required')
    rows, configs, witnesses, histories, closures, initial = {}, {}, {}, {}, {}, {}
    for name,folder in (('control',a.control),('candidate',a.candidate)):
        folder = folder.resolve(strict=True)
        receipt, rows[name] = measured(folder,root)
        configs[name], witnesses[name] = project(folder,root,receipt)
        histories[name] = json.loads((folder/'train/history.json').read_text())
        closures[name] = receipt['training_receipt']['closure']
        initial[name] = inventory(folder/'train/checkpoint-000.npz')
        declared = histories[name]['checkpoints']['checkpoint-000.npz']
        if rows[name]['initial_checkpoint_sha256'] != declared['sha256']:
            raise ValueError('initial archive identity differs from actual history')
        for key,value in initial[name].items():
            if {k:v for k,v in value.items()if k!='nonzero'} != declared['inventory'][key]:
                raise ValueError('initial tensor identity differs from actual archive')
        if rows[name]['optimizer_updates'] != 5000:
            raise ValueError('full matched update budget required')
    if {k:v for k,v in configs['control'].items()if k!='id'} != {k:v for k,v in configs['candidate'].items()if k!='id'}:
        raise ValueError('configuration changed beyond identity')
    left,right = witnesses['control']['sources'],witnesses['candidate']['sources']
    if set(right) != set(left)|{'base_model.py','coarse.py'} or left['model.py'] != right['base_model.py']:
        raise ValueError('unexpected architecture source inventory')
    if any(left[name]!=right[name] for name in left if name not in ('model.py','config.json')):
        raise ValueError('shared numerical source changed')
    for name in ('inputs','normalization_sha256','runtime_manifest_sha256','sample_order_sha256'):
        if closures['control'][name] != closures['candidate'][name]:
            raise ValueError('matched fitting material changed: '+name)
    for name in ('analytical_feedback','supplemental_feedback','direction_feedback','current_feedback','optimizer_work'):
        if histories['control'][name] != histories['candidate'][name]:
            raise ValueError('sampling or actual network work changed: '+name)
    work = histories['control']['optimizer_work']
    if work['network_forwards'] != 50000 or work['forward_cells'] != 425703026:
        raise ValueError('actual network work differs from frozen matched budget')
    left,right = initial['control'],initial['candidate']
    if set(right) != set(left)|{'spatial_coarse.weight'} or any(left[k]!=right[k] for k in left):
        raise ValueError('surviving seeded parameters differ')
    if right['spatial_coarse.weight']['nonzero'] != 0:
        raise ValueError('coarse initialization is not zero')
    result = {'format':'pressure-spatial-coarse-matched-training-v1','status':'passed',
        'models':rows,'actual_optimizer_work':histories['control']['optimizer_work'],
        'all_shared_sources_inputs_streams_exact':True,'all_surviving_initial_tensors_exact':True,
        'new_initial_tensor':right['spatial_coarse.weight'],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scientific_accuracy_asserted':False,'optimizer_updates_executed_by_report':0,
        'scope':'Two fresh managed CUDA fits and exact CPU restores; same update, network-call and forward-cell budgets. Candidate adds coarse arithmetic, so FLOP budgets differ. Independent trajectories own accuracy and selection.'}
    with a.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()
