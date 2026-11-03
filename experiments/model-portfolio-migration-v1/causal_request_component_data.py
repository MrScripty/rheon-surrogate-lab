"""Original CPU decoder/normalization/known-controls; never imports a model."""
import importlib
from pathlib import Path
import sys

from import_common_generator_measurements import ROOT, require

HELPERS = ('pressure_request_data','temporal_codec','temporal_data','temporal_features_v2')


def load_original(entry):
    source = next(row for row in entry['sources'] if row['original_name'] == 'pressure_request_data.py')
    directory = (ROOT/source['path']).parent
    for name in HELPERS:
        prior = sys.modules.get(name)
        if prior is not None:
            require(Path(prior.__file__).resolve() == directory/(name+'.py'), 'Captured data helper module ownership differs')
    sys.path.insert(0,str(directory))
    try:
        module = importlib.import_module('pressure_request_data')
        manifest,norm,parents = module.load(ROOT/entry['inputs']['manifest']['path'],ROOT/entry['inputs']['normalization']['path'],
            [ROOT/entry['inputs'][f'part_{i:02d}']['path'] for i in range(9)],'cpu')
        caches = [module.features(parent,norm['scales'],1,'cpu') for parent in parents]
    finally:
        sys.path.remove(str(directory))
    require(len(parents) == len(caches) == 16 and all(parent['velocity'].shape[0] == 129 for parent in parents),
            'Original complete16parent/129state population required')
    return module,manifest,norm,parents,caches


def owned_case(module,parents,caches,index):
    parent_index,step = divmod(index,128)
    require(0 <= parent_index < 16 and 0 <= step < 128, 'Original one-step transition index required')
    parent = parents[parent_index];scene = parent['scene']
    inputs = module.inputs(parent,caches[parent_index],step,1,'cpu')
    host = {name:value.detach().numpy().copy() for name,value in zip(('velocity','known_forces','scalars'),inputs,strict=True)}
    target = parent['q'][step+1:step+2].unsqueeze(0).numpy()
    identity = dict(parent_index=parent_index,identifier=parent['identifier'],current_state=step,target_state=step+1,
        counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],nominal_dt_s=1/60.,
        scalar_h_dt_rho_mu_F32=host['scalars'][0].tolist())
    return identity,host,target
