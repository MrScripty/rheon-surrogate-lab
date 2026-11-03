"""Ensure the authorized owner session has enough lifetime before a new fit."""
from pathlib import Path
import subprocess
import sys
import time
from v4_support import ROOT,BASE,read,write

def run():
    sys.path[:0]=[str(ROOT/'runs/dependency-migration-v1/Tuldok'),str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1')]
    import agent_api
    from qualify_typed_pressure import OwnerHTTP
    def owner():return OwnerHTTP(agent_api,'http://127.0.0.1:8091','a7da4a8de53541a2a0e2aad08005bb71',ROOT/'runs/tuldok-performance-v1/private/owner-session.txt')
    try:client=owner()
    except agent_api.AgentAPIError as error:
        if getattr(error,'status',None)!=401:raise
        client=None
    renewed=False
    if client is None or client.identity['expires_at']-time.time()<1800:
        subprocess.run([str(ROOT/'runs/dependency-migration-v1/app-env/bin/python3.12'),'-B',str(ROOT/'experiments/pressure-methods-v1/renew_session.py')],cwd=ROOT,check=True,timeout=90)
        client=owner();renewed=True
    if client.identity['expires_at']-time.time()<1800:raise ValueError('Insufficient owner session lifetime for a bounded fit')
    write(BASE/('owner-ready-'+str(time.time_ns())+'.json'),dict(status='authenticated',kind=client.identity['kind'],catalog_id=client.catalog_id,
        expires_at=client.identity['expires_at'],renewed_via_documented_existing_owner_cli=renewed,credential_values_recorded=False))
    print('Authorized owner ready for a bounded managed fit',flush=True)

if __name__=='__main__':run()
