"""User-delegated normal local login renewal; never prints credential values."""
import hashlib
from http.cookies import SimpleCookie
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import uuid


def main():
    root=Path(__file__).resolve().parents[2]
    service=root/'runs/dependency-migration-v1/Tuldok-deployed/current-07'
    catalog=Path('/media/jeremy/OrangeCream/Linux Software/repos/owned/creative-media/Tuldok/data-postgres')
    expected='a7da4a8de53541a2a0e2aad08005bb71'
    private=root/'runs/tuldok-performance-v1/private'
    session_file=private/'owner-session.txt'
    receipt=root/'docs/plans/full-ai-data-diversity-v1/owner-session-renewal-01.json'
    if receipt.exists() or not session_file.is_file() or session_file.is_symlink():
        raise ValueError('One fresh renewal of the existing private session required')
    # The normal CLI needs the same local PostgreSQL connection configuration
    # as the already identified service. Read only that named setting; never
    # print it or inspect existing owner/agent credentials.
    process=Path('/proc/869892')
    argv=[v.decode() for v in (process/'cmdline').read_bytes().split(b'\0') if v]
    if str(service/'app.py') not in argv or str(catalog) not in argv:
        raise RuntimeError('Configured service identity changed before normal login renewal')
    prefix=b'TULDOK_DATABASE_URL='
    configured=None
    with (process/'environ').open('rb') as stream:
        record=bytearray()
        while True:
            byte=stream.read(1)
            if not byte or byte==b'\0':
                if record.startswith(prefix):
                    configured=bytes(record[len(prefix):]).decode();break
                record.clear()
                if not byte:break
            else:record.extend(byte)
    if configured is None:raise RuntimeError('Running service local database setting is unavailable')
    os.environ['TULDOK_DATABASE_URL']=configured
    sys.path.insert(0,str(service))
    from metadata import Metadata
    db=Metadata(catalog,create=False)
    try:
        if db.dataset_id!=expected:raise ValueError('Live local catalog differs from the authorized owner catalog')
    finally:db.close()
    code_file=private/('owner-login-renewal-'+uuid.uuid4().hex+'.txt')
    result=subprocess.run([sys.executable,str(service/'access.py'),'login','--data',str(catalog),'--out',str(code_file)],capture_output=True,timeout=30)
    if result.returncode:
        # The CLI may include connection details in its stderr; do not relay it.
        raise RuntimeError('Documented owner-login CLI failed with exit code '+str(result.returncode))
    endpoint='http://127.0.0.1:8091'
    request=urllib.request.Request(endpoint+'/api/access/session',data=json.dumps(dict(token=code_file.read_text().strip())).encode(),
        headers={'Content-Type':'application/json','Origin':endpoint},method='POST')
    with urllib.request.urlopen(request,timeout=30) as response:
        cookies=SimpleCookie()
        for header in response.headers.get_all('Set-Cookie',[]):cookies.load(header)
        identity=json.loads(response.read())
    if 'tuldok_session' not in cookies:raise RuntimeError('Normal login exchange did not return a session cookie')
    token=cookies['tuldok_session'].value
    def get(path):
        req=urllib.request.Request(endpoint+path,headers={'Authorization':'Bearer '+token})
        with urllib.request.urlopen(req,timeout=30) as response:return json.loads(response.read())
    verified=get('/api/access/session');context=get('/api/workspace/context')
    if (identity.get('kind')!='owner' or verified.get('kind')!='owner' or verified.get('catalog_id')!=expected
            or context.get('dataset_id')!=expected):
        raise RuntimeError('Renewed session did not verify as owner of the exact authorized catalog')
    temporary=private/('.owner-session-renewed-'+uuid.uuid4().hex+'.tmp')
    descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w') as handle:
        os.fchmod(handle.fileno(),0o600);handle.write(token+'\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,session_file)
    parent=os.open(private,os.O_RDONLY)
    try:os.fsync(parent)
    finally:os.close(parent)
    code_file.unlink()
    receipt.write_text(json.dumps(dict(status='renewed_same_catalog_owner_session_via_documented_local_login',created_unix=time.time(),
        authorization='User asked the assistant to refresh the saved owner session: you do that.',
        method='Documented access.py login, then ordinary HTTP one-time-code exchange and authenticated session/workspace identity checks.',
        catalog_id=expected,actor_id=verified.get('actor_id'),credential_id=verified.get('credential_id'),expires_at=verified.get('expires_at'),
        session_file=str(session_file.relative_to(root)),session_file_mode=oct(session_file.stat().st_mode&0o777),
        alternate_existing_credentials_read=False,raw_credentials_logged=False,server_restarted=False,training_runs_submitted=0,
        login_source_sha256=hashlib.sha256((service/'access.py').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print('Renewed and verified the same-catalog owner session; private file updated, no credential values logged.')


if __name__=='__main__':
    try:main()
    except Exception as error:
        # Do not expose HTTP bodies, headers, database DSNs or credential values.
        print('Owner renewal did not complete:',type(error).__name__,getattr(error,'name','') if isinstance(error,ModuleNotFoundError) else '',file=sys.stderr)
        raise SystemExit(1)
