"""Reuse the unchanged qualified native implementation and historical source binding."""
from policy import LAB, load_module

owner=load_module('cuda_pressure_existing_native',LAB/'experiments/variable-pressure-v2/build_native.py')
owner.HERE=LAB/'experiments/variable-pressure-v2'
admit=owner.admit
call=owner.call
