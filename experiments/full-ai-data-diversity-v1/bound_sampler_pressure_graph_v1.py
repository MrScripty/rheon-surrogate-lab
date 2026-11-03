"""GPU sampler compiled from exact managed-restored parameter bytes."""
import hashlib
from support import ROOT, read, digest
from nested_F64_pressure_graph_v1 import NestedF64PressureGraph
from complete_native_entry_v1 import CompleteNativeEntry


class BoundSamplerPressureGraph(NestedF64PressureGraph):
    def __init__(self, pair, shared_pair, build, control_build, sampler_build='runs/full-ai-data-diversity-v1/cuda-fusion-build-20'):
        super().__init__(pair, shared_pair, build, control_build, sampler_build)
        identity = self.nested_kernels.identity
        path = ROOT/identity['sampler_parameter_manifest']
        if digest(path) != identity['sampler_parameter_manifest_sha256']:
            raise ValueError('Compiled sampler parameter manifest changed')
        manifest = read(path); expected = manifest['tensors']; core = self.model
        rows, channels, offsets, weights = core.sparse['point']
        if rows != expected['point']['rows'] or channels != expected['point']['channels'] or offsets != expected['point']['offsets']:
            raise ValueError('Current learned CSR metadata differs from compiled checkpoint')
        for value, record in [(weights, expected['point']['coefficients']), (core.mid_gate, expected['mid_gate']), (core.mix, expected['mix'])]:
            array = value.detach().cpu().numpy()
            if array.dtype.str != record['dtype'] or list(array.shape) != record['shape'] or hashlib.sha256(array.tobytes()).hexdigest() != record['sha256']:
                raise ValueError('Exact deployed learned parameter binding failed; regenerate for a changed checkpoint')
        codegen = read(ROOT/identity['sampler_codegen_receipt'])
        if codegen['manifest_sha256'] != digest(path) or codegen['generated_source_sha256'] != identity['source_sha256']:
            raise ValueError('Compiled source is not the exact recorded learned-parameter serialization')
        self.asset_identity.update(sampler_parameters_static_bound=True, sampler_parameter_manifest=str(path.relative_to(ROOT)),
                                   sampler_parameter_manifest_sha256=digest(path), learned_parameters_changed=False,
                                   ordered_arithmetic_preserved_in_generated_source=True, native_numerical_oracle_in_candidate=False,
                                   whole_model_accuracy_requires_qualification=True)


class BoundSamplerPressureNativeEntry(CompleteNativeEntry):
    generator_class = BoundSamplerPressureGraph

    def __init__(self, *args):
        super().__init__(*args)
        self.asset_identity.update(learned_coefficients_unchanged=True, sampler_parameters_static_bound=True,
                                   original_N1_arithmetic_unchanged=True, new_qualification_required=True)
