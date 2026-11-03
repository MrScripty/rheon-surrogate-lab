"""Finite per-arm CUDA admission, managed fit, fresh CPU restore and entry."""
import argparse
from v5_support import BASE,write
def plan(arm):
    b='runs/pressure-methods-v5/';e='experiments/pressure-methods-v5/'
    command=lambda id,args,timeout=120:dict(id=arm+'-'+id,runtime='control',timeout=timeout,arguments=args)
    commands=[command('cuda',[e+'cuda_qualify.py','--project',b+arm+'-project-02','--output',b+arm+'-cuda-qualification-02.json'],600),
        command('owner',[e+'owner_ready.py']),
        command('fit-restore',['experiments/pressure-methods-v1/manage.py','--config',b+arm+'-experiment-01.json','--assembly',b+arm+'-project-02/assembly.json',
            '--qualification',b+arm+'-qualification-02.json','--tuldok','runs/dependency-migration-v1/Tuldok','--session-file','runs/tuldok-performance-v1/private/owner-session.txt',
            '--output',b+arm+'-study-01','--endpoint','http://127.0.0.1:8091','--catalog-id','a7da4a8de53541a2a0e2aad08005bb71','--project-alias','current-lab'],4300),
        command('entry',['experiments/pressure-methods-v4/restore_entry.py','--identifier',arm,'--project',b+arm+'-project-02','--study',b+arm+'-study-01','--output',b+'entries/'+arm+'.json'])]
    write(BASE/(arm+'-training-plan-01.json'),dict(commands=commands,automatic_retry=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['v5-raw','v5-multilevel','v5-spectrum']);plan(p.parse_args().arm)
