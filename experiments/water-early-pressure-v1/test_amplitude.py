"""Known sinusoid powers independently identify normalization and weak signals."""
import unittest
import numpy as np
from amplitude import temporal_power

class Spectrum(unittest.TestCase):
    def test_sinusoid_integrated_power_and_error_ratio(self):
        dt=1/60;length=32
        signal=2*np.sin(2*np.pi*5*np.arange(length)/length)
        _,frequencies,power=temporal_power(signal[:,None],dt)
        _,_,error=temporal_power((signal*.01)[:,None],dt)
        self.assertAlmostEqual(float(power.sum()*(frequencies[1]-frequencies[0])),2.,places=12)
        self.assertAlmostEqual(float(error[0,5]/power[0,5]),1e-4,places=12)
        _,_,weak=temporal_power((signal*1e-6)[:,None],dt)
        self.assertAlmostEqual(float(weak[0,5]/power[0,5]),1e-12,places=22)

if __name__=='__main__':unittest.main()
