"""Decode the exact source bank, then materialize one frozen numerical cycle."""
from historical_data import *
import historical_data
import json
from pathlib import Path
import methods_view
CONDITIONING={}
def load(raw,manifest):
    parents=methods_view.decode(raw,manifest)
    spec=json.loads(Path(__file__).with_name('fine-tuning.json').read_text())
    cycles=spec['coarse_training_cycles'];calls=0
    if cycles:
        import torch
        import training_multilevel as mg
        import model
        torch.set_num_threads(1)
        with torch.no_grad():
            for parent in parents:
                h=torch.tensor([parent['cell_size_m']],dtype=torch.float64);rho=torch.tensor([parent['density_kg_m3']],dtype=torch.float64)
                bases=[];residuals=[]
                for i in range(len(parent['frames'])):
                    base=torch.from_numpy(parent['arrays']['base'][i:i+1]).unsqueeze(1)
                    initial=torch.from_numpy(parent['arrays']['native_input_residual'][i:i+1]).unsqueeze(1)
                    delta=mg.correction(initial,h,rho)
                    bases.append((base+delta)[0,0].numpy());residuals.append((initial-model.full_operator(delta,h,rho))[0,0].numpy());calls+=1
                parent['arrays']=dict(parent['arrays'],base=np.stack(bases),native_input_residual=np.stack(residuals))
    CONDITIONING.clear();CONDITIONING.update(coarse_training_cycles=cycles,materialized_cycles=calls,full_source_parents=len(parents),
        label_policy='Original same-equation reference/rhs unchanged; only supplied base and its true residual change.')
    packet=dict(schema='rheon-multilevel-smoother-training-view-v1',max_cells=2097152,units='SI',model_dt_s=1/60,
        trajectory_ids=[p['entry']['trajectory_id']for p in parents],coarse_training_cycles=cycles)
    return packet,parents

