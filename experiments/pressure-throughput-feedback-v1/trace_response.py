"""Read recorded per-pass contraction on a consumed development scene/step."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observe(campaign, scene, step):
    cases = list(campaign.glob(f'case-*-scene-{scene:02d}'))
    if len(cases) != 1:
        raise ValueError('one unique consumed case directory required')
    folder = cases[0]
    episode = json.loads((folder/'episode.json').read_text())
    if (episode['status'] != 'passed' or episode['metrics']['scene'] != scene
            or episode['cuda_initialized']):
        raise ValueError('completed physically qualified CPU case required')
    path = folder/'native/callbacks.jsonl'
    callbacks = [json.loads(line) for line in path.read_text().splitlines()]
    if [row['network_passes'] for row in callbacks] != episode['metrics']['calls']:
        raise ValueError('recorded callback call counts differ from qualified case')
    selected = [row for row in callbacks if row['step'] == step]
    if len(selected) != 1:
        raise ValueError('one exact recorded step required')
    row = selected[0]
    trace = row['trace']
    if row['network_passes'] != len(trace) or len(trace) < 4:
        raise ValueError('at least four recorded one-network-call proposals required')
    ratios = [current['residual_l2']/previous['residual_l2']
              for previous,current in zip(trace[2:],trace[3:])]
    return dict(episode_sha256=digest(folder/'episode.json'),callbacks_sha256=digest(path),
                source=str(folder),manifest_sha256=episode['manifest_sha256'],
                network_calls=row['network_passes'],model_correction_s=row['model_correction_seconds'],
                first_equation_pass=next((r['proposals'] for r in trace if r['equation_passed']),None),
                first_pressure_bound_pass=next((r['proposals'] for r in trace if r['pressure_error_bound_passed']),None),
                median_residual_ratio_from_proposal4=statistics.median(ratios),
                residual_ratios_from_proposal4=ratios,trace=trace)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',action='append',required=True,help='LABEL=consumed development campaign')
    p.add_argument('--scene',type=int,required=True)
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if not 0 <= a.scene < 74 or not 16 <= a.step <= 512 or a.output.exists() or not 1 <= len(a.campaign) <= 8:
        raise ValueError('bounded consumed scene/step, one to eight campaigns and fresh output required')
    rows = {}
    for item in a.campaign:
        label,separator,path = item.partition('=')
        if not separator or not label or label in rows:
            raise ValueError('unique labelled campaign required')
        rows[label] = observe(Path(path).resolve(strict=True),a.scene,a.step)
    result = dict(format='pressure_consumed_contraction_trace_v1',source_sha256=digest(Path(__file__)),
                  scene=a.scene,step=a.step,models=rows,optimizer_updates=0,simulations_advanced=0,
                  scope='Post hoc recorded development-case analysis; each model has its own autonomous inputs. No matched-RHS causal conclusion, general noise-floor claim or promotion. Ratios use consecutive recorded residual L2 norms from proposal4 onward.')
    with a.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({label:{key:row[key] for key in ('network_calls','first_equation_pass','first_pressure_bound_pass','median_residual_ratio_from_proposal4')}
                      for label,row in rows.items()}))


if __name__ == '__main__':
    main()
