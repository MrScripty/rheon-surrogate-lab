"""Reach the actual C69 guard with a value-preserving derivative mutation."""
import argparse
from pathlib import Path
import measure
from common import sha, write


def check(config, scratch, output):
    original = measure.component_owner.components
    control = []
    def mutated(state, sample):
        terms, joint = original(state, sample)
        expected = state.torch.autograd.grad(joint, list(state.network.parameters()), retain_graph=True)
        valid = (terms['canonical']+terms['final']+terms['early']).mean()
        actual = state.torch.autograd.grad(valid, list(state.network.parameters()), retain_graph=True)
        assert all(state.torch.equal(a,b) for a,b in zip(expected,actual,strict=True))
        control.append(True)
        return terms, (terms['canonical']+terms['final'].detach()+terms['early']).mean()
    measure.component_owner.components = mutated
    message = None
    try:
        measure.measure(config, 'C69', 23, scratch/'must-not-publish.json', scratch/'vectors')
    except ValueError as error:
        message = str(error)
        assert message == 'Reconstructed joint gradient differs from captured full gradient'
    finally:
        measure.component_owner.components = original
    assert message and control == [True] and not (scratch/'must-not-publish.json').exists()
    write(output, dict(status='passed', model='C69', valid_control_joint_gradient_exact=True,
        mutation='Detach final-feedback derivative only, preserving all scalar values.',
        expected_diagnostic=message, source_sha256=sha(__file__), measure_source_sha256=sha(measure.__file__)))
    print('Actual C69 guard detects value-preserving derivative mutation', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'scratch', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    check(args.config.resolve(), args.scratch.resolve(), args.output.resolve())
