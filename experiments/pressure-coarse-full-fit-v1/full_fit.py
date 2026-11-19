"""Observe the unchanged full5000 fit and let the existing evidence owner finish."""
import argparse
import json
from pathlib import Path
import sys

from capture_policy import verify_sources
from gradient_screen import PREFIX, gradient_snapshot
from stability_screen import StabilityProbe


class FullFitProbe(StabilityProbe):
    def finish(self):
        if self.completed != self.limit or self.cuda_checks != self.limit:
            raise ValueError('Full fit requires every configured optimizer/CUDA check')
        return self.event('full_fit_completed', configured_updates=self.limit,
            successful_cuda_checks=self.cuda_checks, clipping_repairs=self.repairs,
            training_completed=True, pressure_accuracy_qualified=False)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--fit-reference',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    reference=json.loads(args.fit_reference.read_text())
    if reference['format']!='pressure-coarse-full-fit-reference-v1':
        raise ValueError('Explicit qualified full-fit reference required')
    project=Path(__file__).resolve().parent
    verify_sources(project,reference)
    config_path=Path(remaining[remaining.index('--config')+1])
    if config_path.read_bytes()!=(project/'config.json').read_bytes():
        raise ValueError('Selected original captured configuration required')
    config=json.loads(config_path.read_text())
    if config['training']['updates']!=5000:
        raise ValueError('Original5000-update full fit required')
    import train
    import evidence
    def emit(value):print(PREFIX+json.dumps(value,sort_keys=True,allow_nan=False),flush=True)
    probe=FullFitProbe(train,config['training']['updates'],reference,emit)
    probe.install()
    sys.argv[:]=[sys.argv[0],*remaining]
    try:
        evidence.main()
        probe.finish()
    except BaseException as error:
        fields=dict(exception_type=type(error).__name__,message=str(error),successful_cuda_checks=probe.cuda_checks)
        if probe.network is not None:fields['final_gradient_snapshot']=gradient_snapshot(probe.network)
        probe.event('full_fit_failed',**fields)
        raise
    finally:probe.close()


if __name__=='__main__':main()
