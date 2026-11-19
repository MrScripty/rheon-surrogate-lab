"""Load immutable controls without shadowing the captured runtime's support module."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT/'runs/pressure-methods-v4'
DOCS = ROOT/'docs/plans/pressure-methods-v4'
sys.path[:0] = [str(ROOT/'experiments'/name) for name in (
    'pressure-four-pass-distillation-v1', 'pressure-pass-sweep-v1',
    'pressure-methods-v1', 'pressure-methods-v2', 'pressure-methods-v3')]
from common import read, write, sha

def load():
    import inference
    material = read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    entry = read(ROOT/'runs/pressure-methods-v3/entries/v3-native.json')
    state = inference.load(dict(material, models=[entry]), entry['id'])
    import scalable
    wrapper, parity = scalable.qualify(state)
    return state, wrapper, parity

def native(state, population='development'):
    import importlib.util
    path=ROOT/'experiments/pressure-methods-v3/evaluate.py'
    spec=importlib.util.spec_from_file_location('v4_native_population',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return list(module.native_cases(state, population))

def summary(rows):
    import statistics
    return dict(cases=len(rows), qualified=sum(r['qualified'] for r in rows),
        median_ms=1000*statistics.median(r['seconds'] for r in rows),
        worst_pressure_relative_l2=max(r['pressure_relative_l2'] for r in rows),
        worst_relative_residual=max(r['relative_residual'] for r in rows),
        median_network_calls=statistics.median(r['network_calls'] for r in rows))

def pressure_error(pressure, sample):
    import torch
    difference = pressure-sample['reference']; difference = difference-difference.mean()
    reference = sample['reference']-sample['reference'].mean()
    return float(torch.linalg.vector_norm(difference))/max(float(torch.linalg.vector_norm(reference)),1e-300)
