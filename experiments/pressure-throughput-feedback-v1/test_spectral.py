"""Independent cosine amplitude and error-energy check for aggregate bins."""
import unittest
import torch
import spectral


class Spectrum(unittest.TestCase):
    def test_known_low_cosine_amplitude_and_ten_percent_error(self):
        torch.set_num_threads(1);spectral.audit.TORCH=torch
        shape=(5,7,9);wave=torch.cos(torch.pi*(torch.arange(9,dtype=torch.float64)+.5)/9).reshape(1,1,1,1,9).expand(1,1,*shape)
        sample={'rho':torch.tensor([1000.]),'cell_size_m':torch.tensor([.125])}
        spectral.audit.spectral_correction(torch.zeros_like(wave),sample)
        bases,_=spectral.audit.BASES[shape]
        rows=spectral.bins(wave*1.1,wave,torch.zeros_like(wave),bases,.125)
        chosen=next(r for r in rows if r['frequency_bin']==0 and r['relative_amplitude_bin']==4 and r['training_low_block'])
        self.assertAlmostEqual(chosen['signal_energy_pa2'],.5,places=12)
        self.assertAlmostEqual(chosen['error_energy_pa2'],.005,places=12)
        self.assertAlmostEqual(chosen['relative_rms_error'],.1,places=12)
        self.assertEqual(sum(r['modes'] for r in rows),5*7*9-1)
