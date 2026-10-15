"""Validate spectral control against known sealed-grid cosine eigenfunctions."""
import unittest
import torch
import audit


class SpectralControl(unittest.TestCase):
    def test_recovers_known_rectangular_neumann_mode_and_truncates(self):
        torch.set_num_threads(1);audit.TORCH=torch
        shape=(5,7,9);modes=(1,2,3)
        axes=[torch.cos(torch.pi*k*(torch.arange(n,dtype=torch.float64)+.5)/n) for n,k in zip(shape,modes)]
        pressure=torch.einsum('z,y,x->zyx',*axes)[None,None]
        pressure=pressure-pressure[:,:,:1,:1,:1]
        eigenvalue=sum(2-2*torch.cos(torch.tensor(torch.pi*k/n,dtype=torch.float64)) for n,k in zip(shape,modes))
        # L ignores the gauge offset. Use the original cosine product as RHS.
        q=eigenvalue*torch.einsum('z,y,x->zyx',*axes)[None,None]
        sample={'rho':torch.tensor([1000.],dtype=torch.float64),'cell_size_m':torch.tensor([.125],dtype=torch.float64)}
        residual=q/8000.
        torch.testing.assert_close(audit.spectral_correction(residual,sample),pressure,rtol=2e-14,atol=2e-14)
        self.assertLess(float(audit.spectral_correction(residual,sample,2).abs().max()),1e-14)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
