"""Summarize serial current-adapter timing episodes, retaining scope and startup."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',action='append',required=True,help='LABEL=benchmark receipt.json');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();rows=[];sources={}
    for item in a.campaign:
        label,separator,value=item.partition('=');assert separator and label not in sources
        path=Path(value);report=json.loads(path.read_text());assert report['status']=='passed' and report['workers']==1
        assert report['all_worker_children_reaped'] is True
        sources[label]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            config_sha256=report['config_sha256'],source_sha256=report['source_sha256'],
            complete_elapsed_s=report['elapsed_seconds'],memory=report['memory'],
            exact_to_baseline=report.get('exact_to_baseline'),all_worker_children_reaped=True)
        for case in report['records']:
            folder=path.parent/f"case-{case['slot']:02d}-scene-{case['scene']:02d}"/'native'
            frames=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
            frames=[r for r in frames if r.get('ai_quality')]
            callbacks=[json.loads(line) for line in (folder/'callbacks.jsonl').read_text().splitlines()]
            ai=[r['ai_seconds'] for r in frames];native=[r['reference_seconds'] for r in frames]
            rows.append(dict(model=label,scene=case['scene'],steps=len(frames),
                mean_ai_step_s=statistics.mean(ai),median_ai_step_s=statistics.median(ai),
                mean_reference_step_s=statistics.mean(native),median_reference_step_s=statistics.median(native),
                ai_over_reference_mean_ratio=statistics.mean(ai)/statistics.mean(native),
                mean_network_calls=statistics.mean(r['network_passes'] for r in callbacks),
                maximum_network_calls=max(r['network_passes'] for r in callbacks),
                adapter_loading_s=case['adapter_loading_seconds'],complete_child_s=case['complete_child_elapsed_seconds'],
                cpu_affinity=case['cpu_affinity'],source=str(path),
                frames_sha256=hashlib.sha256((folder/'frames.jsonl').read_bytes()).hexdigest(),
                callbacks_sha256=hashlib.sha256((folder/'callbacks.jsonl').read_bytes()).hexdigest()))
    result=dict(format='rheon_pressure_serial_timing_v1',sources=sources,rows=rows,
        scope='One core and one episode at a time after broad work drains. ai_seconds is the current Rheon paired diagnostic AI-branch time, including its additional same-request native reference solve, request capture, learned correction, IPC and protocol logging. External sparse capture and spectral-oracle work are absent from timed episodes. Startup is separate. These branch ratios do not measure standalone AI versus native solver or model inference speed; do not subtract the independently evolving reference_seconds as an estimate of the diagnostic solve.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
