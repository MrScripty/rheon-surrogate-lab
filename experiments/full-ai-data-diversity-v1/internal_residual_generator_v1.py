"""Managed-restored residual on the frozen full-state CUDA group generator."""
import torch
from support import ROOT,read,digest
from internal_group_residual_v1 import InternalGroupGenerator
from internal_residual_network_v2 import InternalGroupResidual
from cuda_native_transfer_v1 import upload_state,download_outputs
from managed_fit_artifacts import load_weights


class InternalResidualGenerator:
    def __init__(self,residual_pair,pair,shared_pair,build,control_build):
        directory=ROOT/residual_pair;receipt=read(directory/'receipt.json');restored=read(directory/'restore/receipt.json')
        if (receipt['status']!='completed' or receipt['role']!='internal_group_nonlinear_residual_component'
                or receipt['optimizer_updates']!=1566 or receipt['parameters']!=10876
                or not all(receipt['artifact_parity'].values()) or not restored['typed_cpu_weight_restore_verified']
                or restored['cpu_model_forward_executed'] or restored['execution']['cuda_initialized']):
            raise ValueError('Exact managed residual fit/CPUrestore without CPUforward required')
        path=directory/'restore/weights.npz'
        if digest(path)!=restored['outputs']['weights.npz']['sha256']:raise ValueError('Restored residual weights changed')
        self.network=InternalGroupResidual(normalization=read(directory/'restore/normalization.json'))
        load_weights(self.network,path,restored['final_inventory']);self.network.eval().requires_grad_(False)
        self.backbone=InternalGroupGenerator(pair,shared_pair,build,control_build)
        self.asset_identity=dict(self.backbone.asset_identity,residual_pair=residual_pair,residual_weights_sha256=digest(path),
            residual_parameters=10876,physical_output_dt_s=1./60.,N8group_widths=[2,2,3,1],
            original_N1_branch_preserved=True,CPU_forward_unported=True,optimizer_updates_during_inference=0)

    def state_dict(self):
        return {**{'backbone.'+n:v for n,v in self.backbone.state_dict().items()},
                **{'residual.'+n:v for n,v in self.network.state_dict().items()}}

    @torch.no_grad()
    def predict(self,fields,scene,start,horizons,reference_substeps):
        return self.backbone.predict(fields,scene,start,horizons,self.network,reference_substeps)

    @torch.no_grad()
    def predict_native(self,fields,scene,start,horizons,reference_substeps):
        resident=upload_state(fields)
        return download_outputs(self.predict(resident,scene,start,horizons,reference_substeps))

    def close(self):self.backbone.close()
