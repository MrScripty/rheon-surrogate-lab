"""Inspect actual correction traces without rerunning or selecting a model."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from execution import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--studies', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for study in args.studies:
        report = read(study/'report.json')
        for row in report['records']:
            if row['model'] == 'native':
                continue
            directory = study/f'scene-{row["scene"]:02d}'/f'{row["model"]}-{row["repetition"]}'
            callbacks = [json.loads(line) for line in (directory/'callbacks.jsonl').read_text().splitlines()]
            frames = [json.loads(line) for line in (directory/'frames.jsonl').read_text().splitlines()]
            frames = [frame for frame in frames if frame.get('ai_quality')]
            if [frame['step'] for frame in frames] != [c['step'] for c in callbacks]:
                raise ValueError('complete matched native frames and callbacks required')
            ratios = [[b['relative_residual']/a['relative_residual']
                       for a,b in zip(c['trace'],c['trace'][1:]) if a['relative_residual']>0]
                      for c in callbacks]
            all_ratios = [value for values in ratios for value in values]
            tails = [value for values in ratios for value in values[-4:]]
            rows.append({'study': str(study), 'model': row['model'],
                'counts_xyz': report['configuration']['scenes'][row['scene']]['counts'],
                'scene_index': row['scene'], 'steps': len(callbacks),
                'failed_native_steps': [f['step'] for f in frames if not f['ai_quality']['qualified']],
                'mean_network_evaluations': statistics.mean(c['network_passes'] for c in callbacks),
                'steps_with_residual_growth': sum(any(r>1 for r in values) for values in ratios),
                'median_successive_ratio': statistics.median(all_ratios) if all_ratios else None,
                'maximum_successive_ratio': max(all_ratios) if all_ratios else None,
                'median_last_four_ratio': statistics.median(tails) if tails else None,
                'callback_sha256': hashlib.sha256((directory/'callbacks.jsonl').read_bytes()).hexdigest(),
                'native_frames_sha256': hashlib.sha256((directory/'frames.jsonl').read_bytes()).hexdigest()})
    write(args.output, {'schema': 'rheon-flow-correction-feedback-v1', 'cases': rows,
        'scope': 'post-test diagnostics on consumed cases; ratios include numerical floors; no global contraction proof, new fitting or checkpoint reselection'})


if __name__ == '__main__':
    main()
