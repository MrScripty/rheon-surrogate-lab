"""Single-variable forcing-arithmetic control on the minimized learned-model failure."""
import argparse
from pathlib import Path
from support import ROOT, read, write, digest
from legacy_exact_forcing import bind_for_diagnostic
from replay_pressure_history_failure_v2 import replay


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    reg = read(args.protocol)
    qualification = read(ROOT/reg['legacy_force_qualification'])
    if qualification['status'] != 'passed_strict_legacy_external_forcing':
        raise ValueError('Native-byte-exact external force qualification required')
    with bind_for_diagnostic():
        code = replay(args)
    result = args.output/'result.json'
    write(args.output/'force-binding.json', dict(
        status='completed_strict_legacy_force_diagnostic', result_sha256=digest(result),
        forcing_role='Only known external inputs changed to native scalar arithmetic',
        native_solver_in_candidate=False, optimizer_updates=0,
        independent_confirmation=False, candidate_frontier_point=False,
        legacy_force_qualification_sha256=digest(ROOT/reg['legacy_force_qualification']),
        protocol_sha256=digest(args.protocol)))
    raise SystemExit(code)
