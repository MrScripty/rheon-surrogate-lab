"""Publish the bounded long C98 follow-up after its recording is sealed."""
import json
from pathlib import Path
import re
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
DIRECTORY = BASE / 'chart-diagnostics/pressure-methods-v2/c98-drift-01'


def status(state, **details):
    destination = DIRECTORY / 'publication.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(status=state, **details), indent=2)+'\n')
    temporary.replace(destination)


def main():
    deadline = (DIRECTORY / 'specification.json').stat().st_mtime + 1950
    status('waiting_for_long_recording')
    while not (DIRECTORY / 'manifest.json').exists():
        receipt = DIRECTORY / 'scene0-7200/native/receipt.json'
        if receipt.exists():
            value = json.loads(receipt.read_text())
            if value['status'] == 'failed':
                status('followup_failed', error=value.get('error'),
                       short_scene_charts_preserved=True, native_process=value['native_process'])
                return
        if time.time() >= deadline:
            status('followup_not_published', reason='Bounded recording deadline passed',
                   short_scene_charts_preserved=True)
            return
        time.sleep(10)
    status('refreshing')
    subprocess.run([sys.executable, '-B', str(BASE / 'refresh-current-cohorts.py')],
                   cwd=BASE, timeout=180, check=True)
    page = (BASE / 'cohort-progress.html').read_text()
    data = json.loads(re.search(r'<script id="chart-data" type="application/json">(.*?)</script>', page, re.S)[1])
    point = next(p for p in data['cohort_points'] if p['id'] == 'C98')
    traces = [t for t in data['drift_traces'] if t['weights_sha256'] == point['weights_sha256']]
    if len(traces) != 4:
        raise ValueError('Expected all four measured C98 recordings')
    status('published', recordings=len(traces), complete_recordings=sum(t['completed'] for t in traces),
           longest_recorded_time_s=max(t['points'][-1][1] for t in traces))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('publication_failed', error=str(error))
        raise
