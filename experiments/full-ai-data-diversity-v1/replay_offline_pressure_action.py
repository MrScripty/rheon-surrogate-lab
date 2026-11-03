"""OFFLINE debug only: substitute the scoring edge action, retaining learned pressure."""
import argparse
from pathlib import Path
from support import ROOT, read, write, digest
from legacy_table_forcing import bind_for_diagnostic
from replay_pressure_history_failure_v2 import replay
from learned_pressure_history import LearnedPressureHistory
from numerics import fine_action
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    reg = read(args.protocol)
    if reg['action_role'] != 'OFFLINE_edge_action':
        raise ValueError('Explicit OFFLINE diagnostic action required')
    previous = LearnedPressureHistory.action

    def offline_action(self, pressure, counts, h, rho):
        return fine_action(pressure, torch.tensor([h], dtype=torch.float64),
                           torch.tensor([rho], dtype=torch.float64))[0, 0]

    LearnedPressureHistory.action = offline_action
    try:
        with bind_for_diagnostic():
            code = replay(args)
    finally:
        LearnedPressureHistory.action = previous
    write(args.output/'OFFLINE-action.json', dict(
        status='completed_OFFLINE_action_control', result_sha256=digest(args.output/'result.json'),
        numerical_operator_in_diagnostic=True, numerical_operator_allowed_in_candidate=False,
        optimizer_updates=0, independent_confirmation=False, candidate_frontier_point=False))
    raise SystemExit(code)
