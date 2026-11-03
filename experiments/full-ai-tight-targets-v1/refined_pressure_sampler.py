"""Same trained complete generator with separately managed refined pressure weights."""
from profile_frozen_components import ROOT,read,digest
from restored_recurrent_sampler import load_model as load_transport
from restored_modal_core import RestoredModalCore


def load_model(pair,library_path,pressure_pair):
    path=ROOT/pressure_pair;receipt=read(path/'receipt.json')
    if receipt['status']!='completed'or receipt['optimizer_updates']!=80 or not all(receipt['artifact_parity'].values()):
        raise ValueError('Completed registered80-call managed pressure precision fit/restore required')
    for key in ('training_run_id','restore_run_id'):
        terminal=read(path/('terminal-'+receipt[key]+'.json'))
        if terminal['state']!='completed'or terminal['custody_phase']!='drained':raise ValueError('Both precision owners must drain')
    model=load_transport(pair,library_path)
    try:
        refined=RestoredModalCore(path,library_path)
        model.pressure_core.close();model.pressure_core=refined
        model.eval().requires_grad_(False)
        model.frozen_versions={name:value._version for name,value in model.named_parameters()}
        model.asset_identity['pressure_precision_refinement']=dict(pair=pressure_pair,receipt_sha256=digest(path/'receipt.json'),
            weights_sha256=digest(path/'restore/weights.npz'),training_optimizer_calls=80,
            exact_managed_cpu_restore_verified=True,analytic_inverse_constructed=False)
        return model
    except BaseException:model.close();raise
