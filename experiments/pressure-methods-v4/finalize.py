"""Publish the local report only after the six-track evidence audit passes."""
import ast
import importlib.util
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
from v4_support import ROOT,BASE,DOCS,read,write,sha

def finish():
    completion=read(DOCS/'completion-audit.json');audit=read(DOCS/'final-audit.json')
    if completion['status']!='complete' or audit['status']!='passed':raise ValueError('Passing final evidence audit required')
    if completion['final_audit_sha256']!=sha(DOCS/'final-audit.json'):raise ValueError('Completion audit identity differs')
    if audit['source_sha256']!=sha(ROOT/'experiments/pressure-methods-v4/audit.py'):raise ValueError('Audit source changed')
    for name,expected in audit['evidence'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Audited evidence changed: '+name)
    campaign=read(DOCS/'campaign.json');campaign.update(status='complete',completion_audit_sha256=sha(DOCS/'completion-audit.json'),
        successful_fits=8,successful_managed_runs=16,optimizer_updates=19200)
    write(DOCS/'campaign.json',campaign)
    report_path=ROOT/'experiments/pressure-methods-v4/report.py'
    spec=importlib.util.spec_from_file_location('v4_final_report',report_path)
    report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report);report.build()
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for name,value in attrs:
                if name not in ['href','src']or not value:continue
                parsed=urlsplit(value)
                if parsed.scheme or parsed.netloc or not parsed.path:continue
                if not(DOCS/unquote(parsed.path)).resolve().exists():raise ValueError('Broken report link: '+value)
    Links().feed((DOCS/'results.html').read_text())
    sources={}
    for path in sorted((ROOT/'experiments/pressure-methods-v4').glob('*.py')):
        ast.parse(path.read_text(),filename=str(path));sources[str(path.relative_to(ROOT))]=sha(path)
    artifacts={}
    for path in sorted(DOCS.rglob('*')):
        if path.is_file()and path.name!='publication.json':artifacts[str(path.relative_to(ROOT))]=sha(path)
    write(DOCS/'publication.json',dict(status='complete',artifacts=artifacts,sources=sources,
        report_source_sha256=sha(ROOT/'experiments/pressure-methods-v4/report.py'),finalizer_source_sha256=sha(__file__),promotion=False))
    write(BASE/'work-state.json',dict(status='complete',live_owned_workers=[],successful_fits=8,successful_managed_runs=16,
        evidence_audit='docs/plans/pressure-methods-v4/final-audit.json',publication_sha256=sha(DOCS/'publication.json')))
    print('Local report, links, sources and all publication artifacts verified',flush=True)

if __name__=='__main__':finish()
