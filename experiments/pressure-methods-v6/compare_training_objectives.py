"""Compare actual three-step losses and every model gradient across executions."""
import copy,importlib.util,os,sys,time
from v6_support import ROOT,HERE,BASE,read,write,sha
def run():
    output=BASE/'adjoint-objective-parity-01.json'
    if output.exists():raise ValueError('Fresh composed-objective comparison required')
    old=BASE/'v6-unrolled-project-01';new=BASE/'v6-unrolled-project-02';os.sched_setaffinity(0,{22});sys.path.insert(0,str(old))
    import torch,model,data,fine_tuning
    model.configure('cpu');torch.set_num_threads(1);model.select(read(old/'config.json'))
    network=model.PressureCorrectionNet();slots={r['id']:ROOT/r['path']for r in read(old/'project.json')['data']}
    fine_tuning.initialize(network,slots['initial_weights'],data.helpers());normalization=read(old/'fixed-normalization.json')
    sys.path.insert(0,str(new));spec=importlib.util.spec_from_file_location('new_adjoint_objective',new/'fine_tuning.py');obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);obj.select(read(new/'fine-tuning.json'))
    torch.manual_seed(2026101918);rows=[]
    for shape in [(17,13,9),(64,64,64),(128,128,128)]:
        reference=torch.randn((1,1,*shape),dtype=torch.float64);reference-=reference[:,:,:1,:1,:1].clone();h=torch.tensor([.0625],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
        rhs=model.full_operator(reference,h,rho);sample=dict(base=torch.zeros_like(rhs),reference=reference,rhs=rhs,native_input_residual=rhs,cell_size_m=h,rho=rho,dt=torch.tensor([1/60],dtype=torch.float64))
        values=[]
        for objective in [fine_tuning,obj]:
            network.zero_grad(set_to_none=True);tick=time.perf_counter();loss=objective.loss(model,network,sample,normalization).mean();forward=time.perf_counter()-tick
            tick=time.perf_counter();loss.backward();backward=time.perf_counter()-tick
            gradients=torch.cat([p.grad.flatten()for p in network.parameters()]).clone();values.append((float(loss.detach()),gradients,forward,backward))
        loss_relative=abs(values[0][0]-values[1][0])/max(abs(values[0][0]),1e-100);gradient_relative=float(torch.linalg.vector_norm(values[0][1]-values[1][1]))/max(float(torch.linalg.vector_norm(values[0][1])),1e-100)
        if loss_relative>2e-6 or gradient_relative>1e-5:raise ValueError('Composed GCR objective/gradient differs: '+str((shape,loss_relative,gradient_relative)))
        rows.append(dict(counts_zyx=shape,loss_relative_difference=loss_relative,all_parameter_gradient_relative_difference=gradient_relative,
          original_forward_seconds=values[0][2],original_backward_seconds=values[0][3],adjoint_forward_seconds=values[1][2],adjoint_backward_seconds=values[1][3]))
    write(output,dict(status='passed',rows=rows,all_parameters=648,source_sha256=sha(__file__),old_objective_sha256=sha(old/'fine_tuning.py'),new_objective_sha256=sha(new/'fine_tuning.py'),optimizer_updates=0,
        interpretation='Complete3-GCR-step objective and648 parameter gradients, including full128³; one-shot diagnostic times, not final isolated performance claims.'))
    print('Complete three-step losses and all648 gradients agree, including full128³',flush=True)
if __name__=='__main__':run()
