"""Retain CPU fitting/development checks and three independent raw trajectories."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

LAB=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','trial','packet','validation','native-binary','rollout-binary'):
        p.add_argument('--'+name,type=str if name=='trial' else Path,required=True)
    args=p.parse_args();root=args.root.absolute()/args.trial
    restore=root/'managed-full/stage-01/artifacts';inference=root/'managed-full/stage-02/artifacts'
    if not (root/'managed-full/receipt.json').exists():raise ValueError('completed managed sequence required')
    base=['--project',str(root/'project'),'--artifacts',str(restore)]
    native=[sys.executable,str(LAB/'experiments/water-conditioning-v1/native_parallel.py'),'--binary',str(args.native_binary.absolute())]
    audit=[sys.executable,str(LAB/'experiments/water-residual-learning-v1/audit.py'),*base]
    commands=[('fitted-audit',audit+['--packet',str(args.packet.absolute()),'--training-only','--output',str(root/'fitted-audit')]),
        ('fitted-native',native+['--packet',str(args.packet.absolute()),'--inference',str(restore),'--training-only','--output',str(root/'fitted-evaluation')]),
        ('validation-audit',audit+['--packet',str(args.validation.absolute()),'--output',str(root/'validation-audit')]),
        ('validation-native',native+['--packet',str(args.validation.absolute()),'--inference',str(inference),'--output',str(root/'validation-evaluation')])]
    cpus=sorted(os.sched_getaffinity(0));cpus=cpus[1:] or cpus
    for index,name in enumerate(('cube','long','compact')):
        commands.append((name+'-trajectory',['taskset','-c',str(cpus[index%len(cpus)]),sys.executable,str(LAB/'experiments/water-residual-learning-v1/rollout.py'),
            '--binary',str(args.rollout_binary.absolute()),'--scene',str(LAB/f'experiments/water-residual-learning-v1/configs/trajectory-{name}.json'),
            '--output',str(root/(name+'-trajectory')),*base]))
    (root/'cpu-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    def run(item):
        name,command=item
        with (root/(name+'.log')).open('xb') as stream:subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
        print(json.dumps({'trial':args.trial,'completed':name}),flush=True)
    for item in commands[:4]:run(item)
    with ThreadPoolExecutor(max_workers=min(3,len(cpus))) as pool:list(pool.map(run,commands[4:]))


if __name__=='__main__':main()
