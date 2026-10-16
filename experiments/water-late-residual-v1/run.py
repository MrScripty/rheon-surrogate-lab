"""Execute matched direction fits through the existing managed/native owner."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]
spec = spec_from_file_location('coverage_sequence_owner', LAB/'experiments/water-coverage-feedback-v1/run.py')
owner = module_from_spec(spec)
spec.loader.exec_module(owner)


if __name__ == '__main__':
    owner.main()
