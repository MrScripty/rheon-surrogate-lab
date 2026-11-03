"""Callable full learned generator with scoped known-input construction."""
from support import ROOT
from learned_divisor_pressure import LearnedDivisorPressure
from learned_pressure_history import LearnedPressureHistory
from legacy_table_forcing import bind_for_diagnostic
from mixed_modal_sampler import load_model as load_base
from recurrent_sampler_contiguous_v3 import library
import restored_recurrent_sampler as restored


class DivisorGenerator:
    def __init__(self, pair, library_path):
        restored.library = library
        base = load_base(pair, library_path)
        try:
            source = base.pressure_core
            source.close()
            base.pressure_core = LearnedDivisorPressure(source)
            base.asset_identity['pressure_execution'] = 'Tensor basis, division by reciprocals of unchanged managed learned gains'
            self.history = LearnedPressureHistory(base, mode='reuse')
            self.asset_identity = dict(self.history.asset_identity,
                known_external_inputs='Qualified scalar coordinate-table construction; no native physics call',
                worker_contract='One owning caller per model; CPU kernels may use their registered thread team',
                numerical_pressure_solver=False, optimizer_updates=0)
        except BaseException:
            base.close()
            raise

    def __getattr__(self, name):
        return getattr(self.history, name)

    def predict(self, fields, scene, step, horizons):
        # Input selection is part of the complete measured application call.
        with bind_for_diagnostic():
            return self.history.predict(fields, scene, step, horizons)

    def close(self):
        self.history.close()


def load_model(pair, library_path):
    return DivisorGenerator(pair, library_path)
