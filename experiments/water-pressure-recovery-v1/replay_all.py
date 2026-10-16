"""Bounded replay coordinator; each fresh process owns one CPU checkpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
import sys
import time

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB/'experiments/water-shape-repeatability-v1'))
from execution import command, read, write, utc
sys.path.insert(0, str(LAB/'experiments/live-project-v1'))
from observer import Observer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    config = read(args.config)
    root = LAB/'runs/water-pressure-recovery-v1/frontier'
    root.mkdir(parents=True, exist_ok=False)
    reports = LAB/'docs/plans/water-pressure-recovery-v1/reports/frontier'
    reports.mkdir(exist_ok=False)
    observer = Observer(LAB/'runs/live-project')
    started = time.perf_counter()
    tasks = [(m, s) for s in config['samples'] for m in config['models']]
    leased = Queue()
    for cpu in (4, 5):
        leased.put(cpu)
    def run(item):
        model, sample = item
        scene = Path(sample).parent.name.split('-')[1]
        label = model+'-'+scene+'-'+Path(sample).stem
        memory = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        assert memory >= 8*1024**3
        cpu = leased.get()
        try:
            result = command(['taskset','-c',str(cpu),args.python,Path(__file__).with_name('frontier.py'),
                              '--config',args.config,'--model',model,'--sample',LAB/sample,
                              '--output',reports/(label+'.json')],root,label)
        finally:
            leased.put(cpu)
        return result
    receipt = {'started_at_utc':utc(),'state':'running','commands':[],'optimizer_updates':0}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(run, tasks):
                receipt['commands'].append(result)
                write(root/'execution.json', receipt)
                observer.publish('Pressure recovery · CPU replay', 'Same saved inputs; original and stricter stopping targets',len(receipt['commands']),len(tasks))
        receipt['state']='completed'
    except BaseException as error:
        receipt.update(state='failed',error_type=type(error).__name__)
        raise
    finally:
        receipt.update(finished_at_utc=utc(),elapsed_seconds=time.perf_counter()-started)
        write(root/'execution.json',receipt)
        write(reports.parent/'frontier-execution.json',receipt)


if __name__ == '__main__':
    main()
