"""Summarize serial current-adapter timing episodes, retaining scope and startup."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',action='append',required=True,help='LABEL=benchmark receipt.json');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();rows=[]
    for item in a.campaign:
        label,separator,value=item.partition('=');assert separator
        path=Path(value);report=json.loads(path.read_text());assert report['status']=='passed' and report['workers']==1
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
                cpu_affinity=case['cpu_affinity'],source=str(path)))
    result=dict(format='rheon_pressure_serial_timing_v1',rows=rows,
        scope='One core and one episode at a time after broad work drains; full paired simulation steps through current adapter, iterative correction and protocol trace logging included. No sparse capture or independent oracle in timed episodes. Startup retained separately. Not a pure neural-forward benchmark.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
