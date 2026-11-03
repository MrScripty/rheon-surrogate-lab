"""Scoped complete-model diagnostic using identical reused known-input scalars."""
from contextlib import contextmanager
from support import ROOT
import legacy_exact_forcing
from scheduled_force_tables import force


@contextmanager
def bind_for_diagnostic():
    import learned_pressure_history
    previous_force = legacy_exact_forcing.force
    previous_binding = learned_pressure_history.forcing
    legacy_exact_forcing.force = force
    learned_pressure_history.forcing = legacy_exact_forcing.forcing
    try:
        yield
    finally:
        legacy_exact_forcing.force = previous_force
        learned_pressure_history.forcing = previous_binding
