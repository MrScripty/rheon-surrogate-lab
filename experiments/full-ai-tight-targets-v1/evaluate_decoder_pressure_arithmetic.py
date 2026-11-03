"""Three preregistered learned pressure controls after uniform head rounding."""
import argparse
from pathlib import Path
from profile_frozen_components import read, write
from decoder_modal_execution import load_model as restored_model
from sampler_precision_policies import TensorDCTPressure
from learned_division_pressure import LearnedDivisionPressure
from fftw_learned_division import FFTWLearnedDivision
import evaluate_sampler_quantization as quantization
import evaluate_recurrent_sampler as evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reg = read(args.protocol)
    policy = reg['pressure_policy']
    if policy not in ('tensor_multiplier', 'tensor_division', 'fftw_division') or reg['quantization_step'] != 2.**-40:
        raise ValueError('Only the three predeclared learned pressure representations allowed')
    def construct(pair, library_path):
        model = restored_model(pair, library_path)
        try:
            old = model.pressure_core
            old.close()
            if policy == 'tensor_multiplier':
                replacement = TensorDCTPressure(old)
            elif policy == 'tensor_division':
                replacement = LearnedDivisionPressure(old, 'negative_log_division')
            else:
                replacement = FFTWLearnedDivision(old, library_path)
            model.pressure_core = replacement
            model.asset_identity['diagnostic_pressure_arithmetic'] = policy
            if hasattr(replacement, 'encoding'):
                model.asset_identity['diagnostic_pressure_encoding'] = replacement.encoding
            model.frozen_versions = {name: p._version for name, p in model.named_parameters()}
            return model
        except BaseException:
            model.close()
            raise
    quantization.restored_model = construct
    evaluation.load_model = lambda pair, library_path: quantization.construct(pair, library_path, reg['quantization_step'])
    args.mode = 'rollout'
    evaluation.run(args)
    result = read(args.output / 'results.json')
    result.update(status='completed_consumed_scene_learned_pressure_arithmetic_diagnostic',
                  pressure_policy=policy, quantization_step=reg['quantization_step'],
                  raw_managed_artifacts_preserved=True, numerical_pressure_oracle_used=False,
                  analytic_eigenvalues_or_inverse_scales_used=False, optimizer_updates=0,
                  candidate_frontier_point=False, independent_target_pass=False,
                  no_new_learning_or_independent_confirmation=True, latency_claim=False)
    write(args.output / 'results.json', result)
