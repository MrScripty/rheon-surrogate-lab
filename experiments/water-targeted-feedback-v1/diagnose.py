"""Minimized actual-solve replay; diagnostic failures remain retained evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

LAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/water-residual-learning-v1'))
import audit
sys.path.insert(0,str(LAB/'experiments/water-stability-learning-v1'))
from learned_iteration import correct,stopping_policy


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('capture','replay'))
    for name in ('project','artifacts','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('scene','binary','sample'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--maximum-calls',type=int,default=13)
    args=parser.parse_args();started=time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=False)
    policy=stopping_policy(read(LAB/'experiments/water-inference-budget-v1/configs/adaptive.json'))
    if args.mode=='capture':
        import rollout
        original=rollout.correct
        def capture(*values,**options):
            sample=values[3]
            import numpy as np
            with (args.output/'sample.npz').open('xb') as stream:
                np.savez(stream,**{k:v.detach().cpu().numpy() for k,v in sample.items()})
            return original(*values,**options)
        rollout.correct=capture
        scene=read(args.scene);scene['steps']=scene['warmup_steps']+1
        receipt=rollout.run(args.binary.absolute(),scene,args.output/'native',
            args.project.absolute(),args.artifacts.absolute(),proposals=32,
            stopping=policy,single_evaluation=True)
        assert receipt['native_process']['reaped'] and receipt['native_process']['returncode']==0
        result={'scene':scene,'original_scene_sha256':digest(args.scene),
            'binary_sha256':digest(args.binary),'sample_sha256':digest(args.output/'sample.npz'),
            'native_receipt_sha256':digest(args.output/'native/receipt.json')}
    else:
        import numpy as np
        audit.initialize(args.project.absolute(),args.artifacts.absolute(),sorted(os.sched_getaffinity(0)))
        torch=audit.TORCH
        with np.load(args.sample,allow_pickle=False) as arrays:
            sample={k:torch.from_numpy(arrays[k].copy()) for k in arrays.files}
        trace=[]
        with torch.no_grad():
            correct(audit.MODEL,audit.NETWORK,audit.NORM,sample,32,policy,trace,single_evaluation=True)
        calls=trace[-1]['proposals']
        result={'sample_sha256':digest(args.sample),'network_calls':calls,
            'maximum_calls':args.maximum_calls,'passed':calls<=args.maximum_calls and trace[-1]['equation_passed'],
            'trace':trace,'counts_xyz':list(sample['base'].shape[2:][::-1])}
    result.update(mode=args.mode,elapsed_seconds=time.perf_counter()-started,
        project=str(args.project),weights_sha256=digest(args.artifacts/'weights.npz'),
        source_sha256=digest(Path(__file__)),cuda_initialized=audit.TORCH.cuda.is_initialized(),
        optimizer_updates=0,argv=[sys.executable,*sys.argv])
    write(args.output/'report.json',result)
    print({k:result[k] for k in ('mode','elapsed_seconds','network_calls','passed') if k in result})
    if result.get('passed') is False:raise SystemExit(1)


if __name__=='__main__':main()
