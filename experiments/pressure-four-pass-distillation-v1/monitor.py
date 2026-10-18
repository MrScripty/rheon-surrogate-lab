"""Bounded read-only GPU/resource samples until the managed controllers finish."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import time

from common import ROOT, write


def monitor(output, completion):
    started=time.monotonic();samples=[]
    while time.monotonic()-started<4000 and not completion.exists():
        result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],
                              capture_output=True,text=True,timeout=10)
        if result.returncode:
            write(output,dict(status='observation_failed',samples=samples,message='GPU observation unavailable; no estimate substituted'))
            return
        util,memory,power=[float(value.strip())for value in result.stdout.strip().split(',')]
        available=int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')))*1024
        samples.append(dict(utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.monotonic()-started,
                            gpu_utilization_percent=util,gpu_memory_mib=memory,gpu_power_w=power,
                            host_available_bytes=available,disk_free_bytes=shutil.disk_usage(ROOT).free))
        write(output,dict(status='observing',scope='Whole GPU samples; includes desktop and any other GPU work, not per-process attribution',samples=samples))
        time.sleep(10)
    write(output,dict(status='completed'if completion.exists()else'deadline',elapsed_s=time.monotonic()-started,
                      scope='Whole GPU samples, including unrelated GPU work; managed trainer receipts own per-run elapsed time and peak CUDA allocation',samples=samples))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--completion',type=Path,required=True)
    args=parser.parse_args()
    monitor(args.output.resolve(),args.completion.resolve())
