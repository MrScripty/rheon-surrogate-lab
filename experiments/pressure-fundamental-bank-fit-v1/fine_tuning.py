"""Pure-fundamental fitting over the existing four-pass pressure objective."""
import copy
import original_fine_tuning as original
import fit_bank

SPEC=None


def __getattr__(name):
    return getattr(original,name)


def select(spec):
    fit_bank.validate(spec['fitting_bank'])
    if spec['teacher_weight']!=0:
        raise ValueError('No teacher loss in the fitting-capacity test')
    original.select({k:v for k,v in spec.items()if k!='fitting_bank'})
    global SPEC
    SPEC=copy.deepcopy(spec)


def loss(model, network, sample, normalization, criterion=None):
    return original.loss(model,network,sample,normalization,criterion)
