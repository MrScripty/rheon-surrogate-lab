"""Analytical orthogonal modes check signal/error energies and weak-mode bins."""
import unittest
import torch
import frontier


class Spectra(unittest.TestCase):
    def test_analytical_modes_and_parseval(self):
        audit = frontier.diagnose.audit
        audit.TORCH = torch
        n = 9
        x = torch.arange(n, dtype=torch.float64)+.5
        low = torch.cos(torch.pi*x/n)
        high = torch.cos(7*torch.pi*x/n)
        exact = (low+1e-4*high).reshape(1, 1, 1, 1, n)
        pressure = exact+(.01*low+.02*high).reshape_as(exact)
        sample = {'cell_size_m': torch.tensor([.125]), 'rho': torch.tensor([1000.])}
        audit.spectral_correction(torch.zeros_like(exact), sample)
        rows = frontier.spectral_metrics(pressure, exact)['spectral_signal_error']
        self.assertAlmostEqual(sum(r['signal_energy_pa2'] for r in rows), n/2*(1+1e-8), places=11)
        self.assertAlmostEqual(sum(r['error_energy_pa2'] for r in rows), n/2*(.01**2+.02**2), places=13)
        weak = next(r for r in rows if r['band']=='remaining-modes' and r['amplitude_relative_to_peak']==[1e-6, 1e-3])
        self.assertEqual(weak['modes'], 1)
        self.assertAlmostEqual(weak['relative_rms'], 200., places=7)


if __name__ == '__main__':
    unittest.main()
