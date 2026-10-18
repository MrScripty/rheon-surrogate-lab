"""One owned native-control or fixed-pass independent simulation episode."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

from common import ROOT, read, sha, write


def episode(material_path,scene_path,output,cpu,identifier=None,passes=None):
    selected=read(material_path);os.sched_setaffinity(0,{cpu});started=time.perf_counter()
    sys.path[:0]=[str(ROOT/'experiments/pressure-pass-sweep-v1'),
                  str(ROOT/selected['config']['source_lab']/'experiments/water-residual-learning-v1')]
    import rollout
    scene=read(scene_path)
    output.mkdir(parents=True,exist_ok=False)
    manifest=None
    if identifier is not None:
        if passes not in selected['config']['passes']:
            raise ValueError('Explicit frozen pass policy required')
        shutil.copyfile(ROOT/'experiments/pressure-pass-sweep-v1/model_adapter.py',output/'model_adapter.py')
        manifest=output/'manifest.json'
        write(manifest,dict(schema='rheon-pressure-pass-sweep-adapter-v1',material=str(material_path),id=identifier,passes=passes))
    def stop(*_):
        raise KeyboardInterrupt('Episode cancelled; native process owner drains its group')
    signal.signal(signal.SIGTERM,stop)
    receipt=rollout.run(ROOT/selected['config']['binary'],scene,output/'native',adapter_manifest=manifest,
                       control=identifier is None,timeout=600,threads=1)
    if not receipt['native_process']['reaped']:
        raise ValueError('Native child did not reap')
    frames=[json.loads(line)for line in(output/'native/frames.jsonl').read_text().splitlines()]
    ai=[f for f in frames if f.get('ai_quality')]
    attempts_path=output/'native/rejected-attempts.jsonl'
    attempts=[json.loads(line)['attempt']for line in attempts_path.read_text().splitlines()]if attempts_path.exists()else[]
    if any(not f['reference_quality']['qualified']for f in frames):
        raise ValueError('Native reference failed physical qualification')
    completed=bool(ai)and ai[-1]['step']==scene['steps']and not receipt['result']['hard_stop']
    if identifier is None and not completed:
        raise ValueError('Native-only control did not qualify complete episode')
    sys.path.insert(0,str(ROOT/'docs/plans'))
    from drift_pressure_metrics import pressure_magnitudes
    points=[]
    for frame in ai:
        magnitudes,_=pressure_magnitudes(frame,scene['counts'])
        points.append(dict(step=frame['step'],time_s=frame['time_s'],relative_pressure_difference=frame['relative_pressure_difference'],
             relative_equation_residual=frame['ai_quality']['relative_residual'],qualified=frame['ai_quality']['qualified'],magnitudes=magnitudes))
    result=dict(status='completed'if completed else'pressure_rejection',model=identifier or'native-control',passes=passes,
         scene=scene,points=points,recorded_ai_steps=len(points),qualified_ai_steps=sum(p['qualified']for p in points),
         rejected=[dict(step=a['generation'],time_s=a['interval_end_s'],outcome=a['outcome'],
             same_request_pressure_comparison=a.get('same_request_pressure_comparison'))for a in attempts],
         native_reaped=True,elapsed_s=time.perf_counter()-started,
         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,source_sha256=sha(__file__),
         native_frames_sha256=sha(output/'native/frames.jsonl'),material_sha256=sha(material_path),scene_sha256=sha(scene_path))
    write(output/'episode.json',result)
    print(json.dumps({k:result[k]for k in ['status','model','passes','recorded_ai_steps','elapsed_s']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['material','scene','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cpu',type=int,required=True)
    parser.add_argument('--id');parser.add_argument('--passes',type=int)
    args=parser.parse_args()
    episode(args.material.resolve(),args.scene.resolve(),args.output.resolve(),args.cpu,args.id,args.passes)
