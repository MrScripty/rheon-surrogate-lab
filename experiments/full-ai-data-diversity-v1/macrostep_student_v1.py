"""Causal GPU student using a managed fitted temporal readout and frozen stages."""
import hashlib
import torch
from support import ROOT,read,digest
from macrostep_backbone_v1 import FrozenMacroBackbone
from macrostep_readout_v1 import MacroReadout
from train_macrostep_readout_v1 import load,inventory
from cuda_native_transfer_v1 import FIELDS,upload_state,download_outputs


class MacrostepStudent:
    def __init__(self,stages,readout_pair,pair,shared_pair,build,control_build):
        if stages not in (3,4) or not torch.cuda.is_available():raise ValueError('Registered CUDA student required')
        directory=ROOT/readout_pair;receipt=read(directory/'receipt.json');restored=read(directory/'restore/receipt.json')
        if (receipt['status']!='completed' or receipt['role']!='macrostep_temporal_readout_component'
                or receipt['optimizer_updates']!=160 or receipt['parameters']!=3*stages
                or not all(receipt['artifact_parity'].values())):
            raise ValueError('Exact managed160-call student fit/restore pair required')
        if (restored['mode']!='restore' or restored['optimizer_updates']!=0 or restored['student_stages']!=stages
                or not restored['typed_cpu_weight_restore_verified'] or restored['cpu_model_forward_executed']
                or restored['execution']['cuda_initialized']):
            raise ValueError('Typed CPU weight restore without CPU model inference required')
        for key in ('training_run_id','restore_run_id'):
            terminal=read(directory/('terminal-'+receipt[key]+'.json'))
            if terminal['state']!='completed' or terminal['custody_phase']!='drained':raise ValueError('Both managed runs must complete and drain')
        path=directory/'restore/weights.npz'
        if digest(path)!=restored['outputs']['weights.npz']['sha256']:raise ValueError('Typed readout artifact changed')
        self.stages=stages;self.head=MacroReadout(stages)
        load(self.head,path,restored['final_inventory']);raw=inventory(self.head)
        with torch.no_grad():
            q=2.**-40
            self.head.residual_weights.copy_((self.head.residual_weights/q).round()*q)
        self.head.eval().requires_grad_(False)
        self.backbone=FrozenMacroBackbone(pair,shared_pair,build,control_build)
        deployed=self.head.residual_weights.detach().cpu().numpy()
        self.asset_identity=dict(self.backbone.asset_identity,readout_pair=readout_pair,student_stages=stages,
            readout_weights_sha256=digest(path),raw_readout_inventory=raw,
            deployment_quantization=2.**-40,deployed_readout_sha256=hashlib.sha256(deployed.tobytes()).hexdigest(),
            output_dt_s=1/60,actual_stage_rule='TeacherN1uses unchanged one-stage branch; teacherN8uses the declared3/4student stages per physical output.',
            timing_denominator='Physical K outputs only; never divide by internal student stages.',
            CPU_forward_unported=True,new_optimizer_updates=0)

    def state_dict(self):
        return {**{'backbone.'+k:v for k,v in self.backbone.state_dict().items()},
                **{'readout.'+k:v for k,v in self.head.state_dict().items()}}

    @torch.no_grad()
    def predict(self,fields,scene,start,horizons,teacher_substeps):
        if horizons not in (1,2,3,4) or teacher_substeps not in (1,8) or set(fields)!=set(FIELDS):
            raise ValueError('Every physical output and explicit target integration scheme required')
        current=fields;outputs=[]
        for lead in range(horizons):
            stages=1 if teacher_substeps==1 else self.stages
            features=self.backbone.feature_states(current,scene,start+lead,stages,1)[0]
            current=features[-1] if teacher_substeps==1 else self.head.read(features)
            outputs.append(current)
        return outputs

    @torch.no_grad()
    def predict_native(self,fields,scene,start,horizons,teacher_substeps):
        if any(v.device.type!='cpu' for v in fields.values()):raise ValueError('Complete current native CPU fields required')
        resident=upload_state(fields)
        result=download_outputs(self.predict(resident,scene,start,horizons,teacher_substeps))
        torch.cuda.synchronize()
        return result

    def close(self):
        self.backbone.close()
