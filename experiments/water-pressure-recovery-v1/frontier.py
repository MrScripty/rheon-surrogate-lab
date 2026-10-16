"""Saved-input convergence with signal amplitude and spatial error spectra."""
import argparse
import json
from pathlib import Path
import resource
import sys

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB/'experiments/water-pressure-accuracy-v1'))
import diagnose


def spectral_metrics(pressure, exact):
    audit = diagnose.audit
    torch = audit.TORCH
    bases, _ = audit.BASES[pressure.shape[2:]]
    signal = audit.cosine_transform((exact-exact.mean())[0, 0], bases)
    error = audit.cosine_transform(((pressure-exact)-(pressure-exact).mean())[0, 0], bases)
    signal[0, 0, 0] = 0
    error[0, 0, 0] = 0
    amplitude = signal.abs()
    maximum = float(amplitude.max())
    low = torch.zeros_like(amplitude, dtype=torch.bool)
    low[:3, :3, :3] = True
    low[0, 0, 0] = False
    active = torch.ones_like(low)
    active[0, 0, 0] = False
    rows = []
    # The tiny bin includes numerically zero coefficients; absolute error remains
    # useful there, while a ratio alone would be misleading.
    for band, selected in [('low-modes-0-to-2', low), ('remaining-modes', active & ~low)]:
        for lower, upper in [(0., 1e-6), (1e-6, 1e-3), (1e-3, .1), (.1, 1.0000001)]:
            mask = selected & (amplitude >= lower*maximum) & (amplitude < upper*maximum)
            power = float(signal[mask].square().sum())
            noise = float(error[mask].square().sum())
            rows.append({'band': band, 'amplitude_relative_to_peak': [lower, upper],
                         'modes': int(mask.sum()), 'signal_energy_pa2': power,
                         'error_energy_pa2': noise,
                         'relative_rms': (noise/power)**.5 if power > 0 else None})
    return {**diagnose.mode_metrics(pressure, exact), 'spectral_signal_error': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    config = diagnose.read(args.config)
    result = diagnose.replay(config, config['models'][args.model], args.sample, args.output,
                             spectral_metrics=spectral_metrics)
    result.update(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  diagnostic_source_sha256=diagnose.digest(__file__))
    diagnose.write(args.output, result)
    print(json.dumps({'sample': result['sample'], 'model': args.model,
                      'elapsed_seconds': result['elapsed_seconds'],
                      'peak_rss_bytes': result['peak_rss_bytes'],
                      'selected_calls': {k: v['calls'] if v else None for k, v in result['selected'].items()}}))


if __name__ == '__main__':
    main()
