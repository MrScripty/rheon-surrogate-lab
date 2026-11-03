"""Use the qualified CUDA environment for zero-update admission checks."""
import argparse,subprocess
from v6_support import ROOT,HERE,read,environment
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    subprocess.run([runtime['python'],'-B',str(HERE/'qualify.py'),'--project',a.project,'--output',a.output,'--cuda'],cwd=ROOT,env=environment(runtime,'cuda'),check=True,timeout=600)
