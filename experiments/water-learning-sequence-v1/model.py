"""Reuse the qualified CUDA/CPU model without architecture or loss changes."""
from policy import HERE, LAB, load_module
source=HERE/'portable_model.py'
if not source.exists():source=LAB/'experiments/cuda-pressure-v1/model.py'
owner=load_module('water_portable_model',source)
ARCHITECTURE=owner.ARCHITECTURE
CHANNELS=owner.CHANNELS
PressureCorrectionNet=owner.PressureCorrectionNet
configure_cpu=owner.configure_cpu
configure=owner.configure
parameter_count=owner.parameter_count
build_inputs=owner.build_inputs
predict_pressure=owner.predict_pressure
full_operator=owner.full_operator
pressure_loss=owner.pressure_loss
frozen=owner.frozen
