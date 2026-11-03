"""Independently verify the completed local publication and original requirement mapping."""
import argparse,ast,hashlib,json,os,time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit
ROOT=Path(__file__).resolve().parents[2]
DOCS=ROOT/'docs/plans/pressure-methods-v6'
BASE=ROOT/'runs/pressure-methods-v6'
HERE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def verify(output):
    if output.exists():raise ValueError('Fresh independent publication verification required')
    publication=read(DOCS/'publication.json');audit=read(DOCS/'final-audit.json');completion=read(DOCS/'completion-audit.json');report=read(DOCS/'results.json');campaign=read(DOCS/'campaign.json')
    require(publication['status']=='complete'and audit['status']=='passed'and completion['status']==report['status']=='complete','All final artifacts must be complete')
    require(completion['final_audit_sha256']==sha(DOCS/'final-audit.json')and audit['source_sha256']==sha(HERE/'audit.py'),'Completion/audit source chain differs')
    verified=0
    for mapping in [publication['artifacts'],publication['sources'],audit['evidence']]:
        for name,expected in mapping.items():
            require(sha(ROOT/name)==expected,'Changed final evidence: '+name);verified+=1
    actual={str(p.relative_to(ROOT))for p in DOCS.rglob('*')if p.is_file()and p.name!='publication.json'}
    require(actual==set(publication['artifacts']),'Publication omits or adds a documentation artifact')
    source_paths={str(p.relative_to(ROOT))for p in HERE.iterdir()if p.suffix in ['.py','.cpp','.sh']}
    require(source_paths==set(publication['sources']),'Publication omits a current experiment source')
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    require(len(campaign['tracks'])==len(completion['requirements'])==5,'Exactly five original tracks required')
    require(sha(DOCS/'requested-scope.txt')==campaign['requested_scope_sha256'],'Original requested scope changed')
    subrequirements=0
    for original,checked in zip(campaign['tracks'],completion['requirements']):
        require(original['id']==checked['id']and original['name']==checked['name']and checked['status']=='proven','Completion changed a track')
        require(original['required']==[r['requirement']for r in checked['requirements']],'Completion narrowed the original subrequirements')
        for item in checked['requirements']:
            require(item['status']=='proven'and item['evidence'],'A required subitem remains unproven')
            require(all(audit['evidence'].get(p)==h for p,h in item['evidence'].items()),'A subrequirement cites evidence outside the actual audit');subrequirements+=1
    for key,value in [('successful_fits',4),('successful_managed_runs',8),('successful_optimizer_updates',9600)]:require(audit[key]==report[key]==value,'Fit/update accounting differs')
    require(len(report['fits'])==4 and len(report['blind'])==8 and all(r['cases']==14 for r in report['blind']),'Incomplete fitted/blind comparison')
    require(len(report['native_timing'])==18 and all(r['repeats']==3 for r in report['native_timing']),'Incomplete repeated native timing')
    require(len(report['long_drift'])==8 and all(r['diagnostic_only']and not r['application_acceptance_claimed']for r in report['long_drift']),'Incomplete or overstated drift comparisons')
    figures=read(DOCS/'figure-evidence.json');require(len(figures['names'])==5 and len(figures['artifacts'])==15,'Missing standalone figures')
    for name in figures['names']:
        for suffix in ['png','svg','pdf']:require((DOCS/'figures'/(name+'.'+suffix)).is_file(),'Missing figure export')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.count=0
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key not in ['href','src']or not value:continue
                u=urlsplit(value)
                if u.scheme or u.netloc or not u.path:continue
                require((DOCS/unquote(u.path)).resolve().exists(),'Broken rendered link: '+value);self.count+=1
    links=Links();links.feed((DOCS/'results.html').read_text())
    for version in [3,4,5]:
        old=read(ROOT/f'docs/plans/pressure-methods-v{version}/publication.json')
        for mapping in [old.get('artifacts',{}),old.get('sources',{})]:
            for name,expected in mapping.items():require(sha(ROOT/name)==expected,'Earlier publication changed: '+name)
    workers=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit()or int(p.name)==os.getpid():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0');exe=Path(args[0].decode()).name
            if(exe.startswith('python')and any(b'pressure-methods-v6'in a for a in args))or exe=='rheon-large-single-track':workers.append(int(p.name))
        except(OSError,UnicodeError,IndexError):pass
    require(not workers,'Experiment workers remain live')
    result=dict(status='verified',checked_unix=time.time(),original_tracks=5,original_subrequirements=subrequirements,hash_checks=verified,local_report_links=links.count,
        figure_exports=15,publication_sha256=sha(DOCS/'publication.json'),verifier_source_sha256=sha(__file__),live_owned_workers=[],promotion=False)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'publication-verification-01.json');verify(p.parse_args().output)
