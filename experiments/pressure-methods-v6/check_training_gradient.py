"""Independent finite-difference check through all three hierarchy/GCR steps."""
import argparse,sys
from pathlib import Path
from v6_support import ROOT,HERE,BASE,read,write,sha
def run(args):
    if args.output.exists():raise ValueError('Fresh gradient check required')
    project=args.project.resolve();sys.path.insert(0,str(project))
    import torch,model,data,fine_tuning
    model.configure('cpu');torch.set_num_threads(1);model.select(read(project/'config.json'))
    network=model.PressureCorrectionNet();slots={r['id']:ROOT/r['path']for r in read(project/'project.json')['data']}
    fine_tuning.initialize(network,slots['initial_weights'],data.helpers());normalization=read(project/'fixed-normalization.json')
    torch.manual_seed(2026101912);reference=torch.randn((1,1,7,5,3),dtype=torch.float64);reference-=reference[:,:,:1,:1,:1].clone()
    h=torch.tensor([.0625],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64);rhs=model.full_operator(reference,h,rho)
    sample=dict(base=torch.zeros_like(reference),reference=reference,rhs=rhs,native_input_residual=rhs,cell_size_m=h,rho=rho,dt=torch.tensor([1/60],dtype=torch.float64))
    loss=model.training_loss(network,sample,normalization).mean();loss.backward()
    name,p=max(network.named_parameters(),key=lambda row:float(row[1].grad.abs().max()));index=int(p.grad.abs().flatten().argmax());analytic=float(p.grad.flatten()[index]);original=float(p.detach().flatten()[index]);rows=[]
    for epsilon in [.001,.0005]:
        values=[]
        for sign in [-1,1]:
            with torch.no_grad():p.flatten()[index]=original+sign*epsilon;values.append(float(model.training_loss(network,sample,normalization).mean()))
        finite=(values[1]-values[0])/(2*epsilon);relative=abs(finite-analytic)/max(abs(finite),abs(analytic),1e-10)
        if relative>.025:raise ValueError('Hierarchy/GCR gradient differs from finite difference: '+str((analytic,finite,relative)))
        rows.append(dict(epsilon=epsilon,analytic=analytic,finite_difference=finite,relative_difference=relative))
    with torch.no_grad():p.flatten()[index]=original
    write(args.output,dict(status='passed',solver_training_iterations=fine_tuning.SPEC['solver_training_iterations'],parameter=name,index=index,rows=rows,
        source_sha256=sha(__file__),objective_sha256=sha(project/'fine_tuning.py'),project_sha256=sha(project/'project.json'),optimizer_updates=0))
    print('Three-step hierarchy/GCR gradient agrees with independent finite differences',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=BASE/'v6-unrolled-project-01');p.add_argument('--output',type=Path,default=BASE/'unrolled-gradient-01.json');run(p.parse_args())
