"""Same correction network, using exact learned CUDA gathers for inference."""
from temporal_residual_generator_v2 import TemporalResidualGenerator as Parent
from inference_restored_linear_v1 import InferenceRestoredLinear


class TemporalResidualGenerator(Parent):
    def __init__(self, *args):
        super().__init__(*args)
        self.linear = InferenceRestoredLinear(self.model)
        self.asset_identity.update(inference_correction_gathers='ordered learned CUDA CSR',
            training_correction_gathers='original differentiable learned CSR',
            learned_coefficients_and_network_unchanged=True)
