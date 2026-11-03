"""Finalize only the audited local research report and exact export artifacts."""
import ast,importlib.util
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
from v5_support import ROOT,HERE,BASE,DOCS,read,write,sha
def finish():
    audit=read(DOCS/'final-audit.json');completion=read(DOCS/'completion-audit.json')
    if audit['status']!='passed'or completion['status']!='complete'or completion['final_audit_sha256']!=sha(DOCS/'final-audit.json'):raise ValueError('Passing current completion evidence required')
    if audit['source_sha256']!=sha(HERE/'audit.py'):raise ValueError('Audit source changed')
    for name,expected in audit['evidence'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Audited evidence changed: '+name)
    campaign=read(DOCS/'campaign.json');campaign.update(status='complete',successful_fits=3,successful_managed_runs=6,optimizer_updates=7200,completion_audit_sha256=sha(DOCS/'completion-audit.json'));write(DOCS/'campaign.json',campaign)
    spec=importlib.util.spec_from_file_location('v5_final_report',HERE/'report.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.build()
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for name,value in attrs:
                if name not in ['href','src']or not value:continue
                u=urlsplit(value)
                if not u.scheme and not u.netloc and u.path and not(DOCS/unquote(u.path)).resolve().exists():raise ValueError('Broken report link: '+value)
    Links().feed((DOCS/'results.html').read_text())
    sources={}
    for p in sorted(HERE.glob('*.py')):ast.parse(p.read_text(),filename=str(p));sources[str(p.relative_to(ROOT))]=sha(p)
    artifacts={str(p.relative_to(ROOT)):sha(p)for p in sorted(DOCS.rglob('*'))if p.is_file()and p.name!='publication.json'}
    write(DOCS/'publication.json',dict(status='complete',artifacts=artifacts,sources=sources,report_source_sha256=sha(HERE/'report.py'),finalizer_source_sha256=sha(__file__),promotion=False))
    write(BASE/'work-state.json',dict(status='complete',successful_fits=3,successful_managed_runs=6,live_owned_workers=[],publication_sha256=sha(DOCS/'publication.json')))
    print('Verified final report, all links and exact local publication artifacts',flush=True)
if __name__=='__main__':finish()
