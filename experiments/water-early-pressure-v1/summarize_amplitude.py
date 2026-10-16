"""Scalar summaries keep signal amplitude distinct from relative error."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,required=True);p.add_argument('--models',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert not args.output.exists()
    results={}
    for name in args.models:
        source=args.reports/f'amplitude-{name}.json';data=json.loads(source.read_text())
        temporal=[]
        for calls in sorted({r['calls'] for r in data['temporal']['rows']}):
            for hz in sorted({r['frequency_hz'] for r in data['temporal']['rows']}):
                rows=sorted([r for r in data['temporal']['rows'] if r['calls']==calls and r['frequency_hz']==hz
                             and r['signal_power_pa2_per_hz']>0 and r['error_power_pa2_per_hz']>0],
                            key=lambda r:r['signal_power_pa2_per_hz'])
                if len(rows)<3:continue
                signal=np.log10([r['signal_power_pa2_per_hz'] for r in rows])
                error=np.log10([r['error_power_pa2_per_hz'] for r in rows])
                parts=np.array_split(np.arange(len(rows)),3)
                temporal.append({'calls':calls,'frequency_hz':hz,'observations':len(rows),
                    'log_power_error_vs_signal_slope':float(np.polyfit(signal,error,1)[0]),
                    'log_signal_vs_relative_error_correlation':float(np.corrcoef(signal,error-signal)[0,1]),
                    'amplitude_tertiles':[{'tier':tier,'observations':len(indices),
                        **{key:statistics.median(rows[int(i)][key] for i in indices) for key in
                           ('signal_power_pa2_per_hz','error_power_pa2_per_hz','relative_rms_error')}}
                        for tier,indices in zip(('weak','middle','strong'),parts)]})
        results[name]={'source':str(source),'weights_sha256':data['weights_sha256'],
            'temporal':temporal,'spatial':data['spatial']['rows'],'elapsed_seconds':data['elapsed_seconds'],
            'interpretation_limits':'Within each temporal bin, pooled scenes/windows are correlated and differ in dynamics; descriptive associations, not causal effects.32-frame windows at60Hz resolve1.875Hz bins and cannot measure slower cycles. Spatial sweeps are stationary mode interventions, not temporal dynamics. Relative errors near absent signal are ill-conditioned.'}
    args.output.write_text(json.dumps({'models':results,'optimizer_updates':0,'method':'Slope1 means proportional signal/error power; slope0 is consistent with constant absolute error. Neither slope alone identifies numerical versus learned leakage.'},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
