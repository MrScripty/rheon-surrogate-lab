"""Renew the owned UI's local login without restarting or logging credentials."""
from http.cookies import SimpleCookie
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import uuid

from import_common_generator_measurements import ROOT,CONTROLLER,read,binding,require,save


def renew(receipt):
    require(not receipt.exists(),'Preserve the original native owner renewal receipt')
    state = read(CONTROLLER/'controller.json')
    source,private = Path(state['source']),CONTROLLER/'private'
    require(state['status'] == 'ready' and source.is_relative_to(ROOT/'runs/model-portfolio-migration-v1/sources')
            and not private.is_symlink() and (private/'owner-session.txt').is_file(),
            'The existing owned native controller and private session are required')
    command = [s.decode() for s in Path(f'/proc/{state["app_pid"]}/cmdline').read_bytes().split(b'\0') if s]
    require(str(source/'app.py') in command and state['control'] in command
            and Path(f'/proc/{state["launcher_pid"]}').exists(),'Exact owned live application required')
    environment = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TULDOK_DATABASE_URL=read(private/'database.json')['dsn'])
    login = private/('owner-login-renew-'+uuid.uuid4().hex+'.txt')
    result = subprocess.run([sys.executable,'-B',str(source/'access.py'),'login','--data',state['control'],'--out',str(login)],
        env=environment,cwd=source,capture_output=True,timeout=30)
    require(result.returncode == 0,'Documented local login failed; private CLI output withheld')
    try:
        request = urllib.request.Request(state['origin']+'/api/access/session',
            data=json.dumps(dict(token=login.read_text().strip())).encode(),
            headers={'Content-Type':'application/json','Origin':state['origin']})
        with urllib.request.urlopen(request,timeout=30) as response:
            cookies = SimpleCookie()
            for value in response.headers.get_all('Set-Cookie',[]):
                cookies.load(value)
            identity = json.loads(response.read())
        require(identity.get('kind') == 'owner' and identity.get('catalog_id') == state['catalog_id']
                and 'tuldok_session' in cookies,'Renewed owner identity must match this catalog')
        token = cookies['tuldok_session'].value
        request = urllib.request.Request(state['origin']+'/api/access/session',headers={'Authorization':'Bearer '+token})
        with urllib.request.urlopen(request,timeout=30) as response:
            verified = json.loads(response.read())
        require(verified.get('kind') == 'owner' and verified.get('catalog_id') == state['catalog_id']
                and verified.get('actor_id') == identity['actor_id'],'Fresh session verification failed')
        temporary = private/('.owner-session-'+uuid.uuid4().hex+'.pending')
        descriptor = os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w') as stream:
            stream.write(token+'\n');stream.flush();os.fsync(stream.fileno())
        temporary.replace(private/'owner-session.txt')
        parent = os.open(private,os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        login.unlink(missing_ok=True)
    save(receipt,dict(status='same_owned_native_catalog_session_renewed',created_unix=time.time(),catalog_id=state['catalog_id'],
        actor_id=verified['actor_id'],expires_at=verified.get('expires_at'),producer=binding(Path(__file__)),
        authorization='Existing user authorization to refresh the saved owner session: you do that.',
        documented_local_login=True,existing_private_database_configuration_only=True,
        credentials_logged=False,server_restarted=False,project_mutations=0))
    print('Native owner session renewed and verified; UI remains open; no credential values logged.',flush=True)


if __name__ == '__main__':
    try:
        require(len(sys.argv) == 2,'Specify one fresh receipt path')
        renew(Path(sys.argv[1]).resolve())
    except Exception as error:
        print('Native owner renewal failed:',type(error).__name__,file=sys.stderr)
        raise SystemExit(1)
