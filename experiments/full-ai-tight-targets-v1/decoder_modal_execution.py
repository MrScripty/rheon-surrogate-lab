"""Frozen decoder-compatible managed checkpoint for equivalent CPU executions."""
from refined_pressure_sampler import load_model as load_refined


def load_model(pair, library_path):
    return load_refined(pair, library_path,
                        'runs/full-ai-tight-targets-v1/decoder-modal-pair-09')
