"""Qualify the new bank and low-mode objective in addition to restore contracts."""
import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, source_inputs, write
from qualify import qualify


def check(project, output):
    qualify(project, output)
    import torch
    import model
    import data
    import analytical
    import fine_tuning
    inputs = source_inputs(project)
    _, parents = data.load(tuple(inputs[k].read_bytes() for k in ['dataset', 'dataset_extra']), read(inputs['splits']))
    actual = [data.sample(p, slot) for p in parents for slot in range(len(p['frames']))]
    config = read(project/'config.json')
    amplitude = analytical.training_correction_rms(actual)
    specification = config['supplemental_feedback']
    bank, order, evidence = analytical.bank(actual[0], {k:v for k,v in specification.items() if k != 'source'},
                                           config['training']['updates'], amplitude_pa=amplitude)
    exposures = evidence['grid_exposures']
    assert len(exposures) == 8 and max(exposures)-min(exposures) <= 1
    defect = 0.
    for sample in bank:
        reference = sample['reference']
        rhs = sample['rhs']
        value = float(torch.linalg.vector_norm(model.full_operator(reference, sample['cell_size_m'], sample['rho'])-rhs))/float(torch.linalg.vector_norm(rhs))
        defect = max(defect, value)
        assert value < 1e-10
    # An independent pure Neumann mode fixes the exact low-block energy ratio.
    shape = (9, 11, 13)
    target, _ = analytical.eigenmode(shape, (1, 0, 0))
    sample = {'reference': target, 'base': torch.zeros_like(target)}
    prediction = (target*1.01).requires_grad_()
    observed = fine_tuning.final_feedback(prediction, sample)
    weight = fine_tuning.SPEC['low_mode_feedback']['weight']
    expected = weight*math.log1p(.01**2/1e-12)
    assert math.isclose(float(observed.detach()), expected, rel_tol=1e-10, abs_tol=1e-12)
    observed.sum().backward()
    assert bool(torch.isfinite(prediction.grad).all())
    assert weight == 0 or bool((prediction.grad != 0).any())
    shifted = fine_tuning.final_feedback(prediction.detach()+7., sample)
    assert torch.allclose(observed.detach(), shifted, rtol=1e-10, atol=1e-12)
    assert float(fine_tuning.final_feedback(target, sample)) == 0.
    high, _ = analytical.eigenmode(shape, (4, 4, 4))
    assert float(fine_tuning.final_feedback(target+high, sample)) < 1e-14
    result = read(output)
    result.update(low_mode_objective=dict(independent_pure_mode_expected=expected, observed=float(observed.detach()),
                  constant_gauge_invariant=True, high_mode_excluded=True, finite_gradient=True),
                  hard_bank=dict(samples=len(bank), grid_exposures=exposures, bytes=evidence['resident_tensor_bytes'],
                  description_sha256=evidence['bank_description_sha256'], order_sha256=evidence['sampling_order_sha256'],
                  independent_operator_worst_defect=defect), supplemental_qualification_source_sha256=sha(__file__))
    write(output, result)
    print('Qualified independent bank and low-mode objective', project.name, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.project.resolve(), args.output.resolve())
