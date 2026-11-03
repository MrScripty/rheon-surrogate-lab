"""Registered full-objective managed residual checkpoint; CUDA inference only."""
import torch
from support import ROOT,read,digest
from internal_group_residual_v1 import InternalGroupGenerator
from internal_residual_network_v2 import InternalGroupResidual
from cuda_native_transfer_v1 import upload_state,download_outputs
from managed_fit_artifacts import load_weights

class InternalResidualGenerator:
    def __init__(self,residual_pair,pair,shared_pair,build,control_build):
        directory=ROOT/residual_pair;receipt=read(directory/'receipt.json');trained=read(directory/'train/receipt.json');restored=read(directory/'restore/receipt.json')
        if (receipt['status']!='completed' or receipt['role']!='internal_group_nonlinear_residual_component' or receipt['parameters']!=10876
                or not all(receipt['artifact_parity'].values()) or not restored['typed_cpu_weight_restore_verified']
                or restored['cpu_model_forward_executed'] or restored['execution']['cuda_initialized']):
            raise ValueError('Exact managed residual fit/typedCPUrestore required')
        if (trained['schema']!='rheon-internal-residual-managed-v3' or trained['optimizer']!='full_batch_LBFGS'
                or not 1<=trained['optimizer_updates']<=8 or not 1<=trained['closure_evaluations']<=18
                or trained['optimizer_updates']!=receipt['optimizer_updates']
                or restored['originating_training_run']['run_id']!=receipt['training_run_id']
                or digest(directory/'train/receipt.json')!=receipt['training_receipt_sha256']
                or digest(directory/'restore/receipt.json')!=receipt['restore_receipt_sha256']):
            raise ValueError('Registered finite full-objective training provenance required')
        path=directory/'restore/weights.npz'
        if digest(path)!=restored['outputs']['weights.npz']['sha256'] or digest(path)!=trained['outputs']['weights.npz']['sha256']:raise ValueError('Typed restored weights changed')
        self.network=InternalGroupResidual(normalization=read(directory/'restore/normalization.json'))
        load_weights(self.network,path,restored['final_inventory']);self.network.eval().requires_grad_(False)
        self.backbone=InternalGroupGenerator(pair,shared_pair,build,control_build)
        self.asset_identity=dict(self.backbone.asset_identity,residual_pair=residual_pair,residual_weights_sha256=digest(path),residual_parameters=10876,
            physical_output_dt_s=1./60.,N8group_widths=[2,2,3,1],original_N1_branch_preserved=True,CPU_forward_unported=True,
            training_optimizer='full_batch_LBFGS',training_iterations=trained['optimizer_updates'],training_closure_evaluations=trained['closure_evaluations'],optimizer_updates_during_inference=0)
    def state_dict(self):return {**{'backbone.'+n:v for n,v in self.backbone.state_dict().items()},**{'residual.'+n:v for n,v in self.network.state_dict().items()}}
    @torch.no_grad()
    def predict(self,fields,scene,start,horizons,reference_substeps):return self.backbone.predict(fields,scene,start,horizons,self.network,reference_substeps)
    @torch.no_grad()
    def predict_native(self,fields,scene,start,horizons,reference_substeps):return download_outputs(self.predict(upload_state(fields),scene,start,horizons,reference_substeps))
    def close(self):self.backbone.close()
