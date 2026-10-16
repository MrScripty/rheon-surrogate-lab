"""Bound a real spectral solve's memory before exercising the OOM regression."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB/'experiments/water-residual-learning-v1'))
import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    audit.TORCH = torch
    before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    status = Path('/proc/self/status').read_text().splitlines()
    virtual = int(next(x for x in status if x.startswith('VmSize:')).split()[1])*1024
    # Installed Torch libraries are already mapped; bound additional mappings
    # in this disposable process so the regression cannot exhaust host RAM.
    extra = 512*1024**2
    resource.setrlimit(resource.RLIMIT_AS, (virtual+extra, virtual+extra))
    started = time.perf_counter()
    report = {'counts_xyz': [40, 40, 40], 'additional_address_space_limit_bytes': extra,
        'before_peak_rss_bytes': before_rss, 'optimizer_updates': 0}
    try:
        n = 40
        wave = torch.cos(torch.pi*(torch.arange(n, dtype=torch.float64)+.5)/n)
        pressure = wave[None, None, None, None, :].expand(1, 1, n, n, n)
        sample = {'cell_size_m': torch.tensor([1/16], dtype=torch.float64),
                  'rho': torch.tensor([1000.], dtype=torch.float64)}
        eigenvalue = 2-2*__import__('math').cos(__import__('math').pi/n)
        residual = pressure*(eigenvalue/16000)
        correction = audit.spectral_correction(residual, sample)
        expected = pressure-pressure.flatten()[0]
        error = float(torch.linalg.vector_norm(correction-expected)/torch.linalg.vector_norm(expected))
        assert error < 1e-11
        report.update(status='passed', relative_pressure_error=error)
    except RuntimeError as error:
        if 'allocat' not in str(error).lower() and 'memory' not in str(error).lower():
            raise
        report.update(status='memory_rejected', error=str(error)[:1500])
    report.update(elapsed_seconds=time.perf_counter()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cuda_initialized=torch.cuda.is_initialized())
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(report, flush=True)
    raise SystemExit(0 if report['status']=='passed' else 1)


if __name__ == '__main__':
    main()
