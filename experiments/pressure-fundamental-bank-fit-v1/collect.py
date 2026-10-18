"""Combine completed capacity screens without replacing their immutable evidence."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def collect(directory):
    names=['combined-fundamentals'];combined=read(directory/names[0]/'results.json')
    capacity=combined['decision']['capacity_gate_passed']
    if not capacity:names+=['axis-isolation','shape-isolation']
    arms={name:read(directory/name/'results.json')for name in names}
    if any(r['status']!='completed'for r in arms.values()):
        raise ValueError('Completed capacity screens required')
    decision=dict(status='study_completed',baseline_retained='C53',model_promoted=False,
        combined_capacity_gate_passed=capacity,original_untouched_holdouts_consumed=0,
        next_action='Design a separate mixed-loss retention test.'if capacity else
            'Use the per-axis/per-shape screens to choose a bounded optimization diagnosis; retain C53.')
    result=dict(schema='rheon-fundamental-bank-study-results-v1',status='completed',arms=arms,decision=decision,
        sources={str((directory/name/'results.json').relative_to(ROOT)):sha(directory/name/'results.json')for name in names})
    write(directory/'results.json',result);write(directory/'decision.json',decision)
    print(decision,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    collect(parser.parse_args().directory.resolve())
