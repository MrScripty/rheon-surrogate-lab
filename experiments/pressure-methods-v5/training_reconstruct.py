"""Deterministic same-equation spectral labels, including declared on-policy states."""
import hashlib
import numpy as np
import torch
import reference_spectral

SOURCES={'native-generated-spectral-reference-cold-warm','native-generated-diverse-spectral','native-generated-onpolicy-spectral-label','operator-generated-spatial-band'}
def complete(arrays,manifest):
    if not isinstance(manifest.get('derived_fields'),dict):return arrays
    if len(manifest['derived_fields'])>64:raise ValueError('Bounded derived native inventory required')
    with torch.no_grad():
        for index,parent in enumerate(manifest['parents']):
            names={f'p{index:04d}_rhs',f'p{index:04d}_reference'};selected=names.intersection(manifest['derived_fields'])
            if not selected:continue
            if selected!=names or parent['source']not in SOURCES or parent['split']!='train':raise ValueError('Declared same-equation training labels required')
            if any(name in arrays for name in names):raise ValueError('Derived field conflicts with encoded field')
            h=torch.tensor([parent['cell_size_m']],dtype=torch.float64);rho=torch.tensor([parent['density_kg_m3']],dtype=torch.float64)
            rhs=[];reference=[]
            for frame in range(len(parent['frames'])):
                base=torch.from_numpy(arrays[f'p{index:04d}_base'][frame:frame+1]).unsqueeze(1)
                residual=torch.from_numpy(arrays[f'p{index:04d}_native_input_residual'][frame:frame+1]).unsqueeze(1)
                value=residual+reference_spectral.fine_action(base,h,rho)
                pressure=base+reference_spectral.correction(residual,h,rho,None);pressure=pressure-pressure[:,:,:1,:1,:1].clone()
                rhs.append(value[0,0].numpy().copy());reference.append(pressure[0,0].numpy().copy())
            for field,values in [('rhs',rhs),('reference',reference)]:
                name=f'p{index:04d}_{field}';array=np.stack(values);spec=manifest['arrays'][name]
                if manifest['derived_fields'][name]!='exact-native-spectral-v1'or list(array.shape)!=spec['shape']or array.dtype.str!=spec['dtype']or hashlib.sha256(array.tobytes()).hexdigest()!=spec['sha256']:
                    raise ValueError('Recreated labels differ from captured full-field bytes')
                arrays[name]=array
    return arrays
