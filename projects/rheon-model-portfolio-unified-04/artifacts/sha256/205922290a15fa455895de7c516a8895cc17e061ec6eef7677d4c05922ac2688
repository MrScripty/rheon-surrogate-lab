"""Capped managed clipping stability; never a completed model or accuracy fit."""
import argparse
import json
from pathlib import Path
import sys

from capture_policy import verify_sources
from gradient_screen import Probe, gradient_snapshot, PREFIX


class StabilityProbe(Probe):
    def __init__(self, *args):
        super().__init__(*args)
        self.repairs = 0
        self.cuda_checks = 0

    def event(self, kind, **fields):
        if kind == 'diagnostic_cap_reached':
            if self.cuda_checks != self.limit:
                raise ValueError('Original CUDA finite-state checks incomplete')
            fields.update(successful_cuda_checks=self.cuda_checks, clipping_repairs=self.repairs,
                          training_completed=False, final_candidate_eligible=False)
        return super().event(kind, **fields)

    def install(self):
        super().install()
        original_clip = self.trainer.clip_l2_
        def repaired(values):
            self.repairs += 1
            details = dict(update=self.current, gradient_snapshot_after_clip=gradient_snapshot(self.network)) if self.repairs == 1 else {}
            self.event('clip_repaired', repair_index=self.repairs, norms=values, **details)
        def clip(parameters, maximum):
            try:
                return original_clip(parameters, maximum, on_repair=repaired)
            except RuntimeError:
                self.event('repair_rejected', update=self.current,
                           gradient_snapshot=gradient_snapshot(self.network))
                raise
        self.patch(self.trainer, 'clip_l2_', clip)
        original_verify = self.trainer.verify_cuda_update
        def verify(*args, **kwargs):
            result = original_verify(*args, **kwargs)
            self.cuda_checks += 1
            return result
        self.patch(self.trainer, 'verify_cuda_update', verify)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gradient-limit', type=int, required=True)
    parser.add_argument('--gradient-reference', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if not 1 <= args.gradient_limit <= 1500:
        raise ValueError('One through1500 stability updates only')
    reference = json.loads(args.gradient_reference.read_text())
    if reference['format'] != 'pressure-coarse-clipping-reference-v1':
        raise ValueError('Explicit qualified clipping reference required')
    verify_sources(Path(__file__).resolve().parent, reference)
    config = Path(remaining[remaining.index('--config')+1])
    if config.read_bytes() != Path(__file__).with_name('config.json').read_bytes():
        raise ValueError('Selected original captured config required')
    import train
    import evidence
    def emit(value):
        print(PREFIX+json.dumps(value, sort_keys=True, allow_nan=False), flush=True)
    probe = StabilityProbe(train, args.gradient_limit, reference, emit)
    probe.install()
    sys.argv[:] = [sys.argv[0], *remaining]
    try:
        evidence.main()
    finally:
        probe.close()


if __name__ == '__main__':
    main()
