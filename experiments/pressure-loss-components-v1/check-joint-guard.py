"""Detect a value-preserving derivative mutation on an actual captured case."""
import argparse
from pathlib import Path
import measure
from common import sha,write


def check(config,output,scratch):
    original=measure.components;control=[]
    def mutated(state,sample):
        terms,joint=original(state,sample)
        parameters=list(state.network.parameters())
        valid=(terms['canonical']+terms['final']+terms['early']).mean()
        expected=state.torch.autograd.grad(joint,parameters,retain_graph=True)
        actual=state.torch.autograd.grad(valid,parameters,retain_graph=True)
        assert all(state.torch.equal(a,b)for a,b in zip(actual,expected,strict=True))
        control.append(True)
        # Preserve all term values; remove only the final feedback derivative.
        return terms,(terms['canonical']+terms['final'].detach()+terms['early']).mean()
    measure.components=mutated
    message=None
    try:measure.measure(config,'C66',22,scratch/'should-not-publish.json',scratch/'vectors')
    except ValueError as error:
        message=str(error)
        assert message=='Reconstructed joint gradient differs from captured full gradient'
    finally:measure.components=original
    assert control==[True]
    assert message and not (scratch/'should-not-publish.json').exists()
    write(output,dict(status='passed',mutation='Detach only final feedback in the returned joint graph; values remain unchanged.',
        expected_diagnostic=message,valid_control_gradient_exact=True,actual_captured_model='C66',source_sha256=sha(__file__),measure_source_sha256=sha(measure.__file__)))
    print('Joint guard detects value-preserving derivative mutation',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['config','output','scratch']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();check(a.config.resolve(),a.output.resolve(),a.scratch.resolve())
