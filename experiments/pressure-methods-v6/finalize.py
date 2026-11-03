"""Finalize exact local artifacts without changing any audited scientific evidence."""
import ast
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
from v6_support import ROOT,HERE,BASE,DOCS,read,write,sha
def run():
    audit=read(DOCS/'final-audit.json');completion=read(DOCS/'completion-audit.json')
    if audit['status']!='passed'or completion['status']!='complete'or completion['final_audit_sha256']!=sha(DOCS/'final-audit.json'):raise ValueError('Current passing final audit required')
    if audit['source_sha256']!=sha(HERE/'audit.py'):raise ValueError('Final audit source changed')
    for name,expected in audit['evidence'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Audited evidence changed: '+name)
    import final_report
    final_report.build()
    class Links(HTMLParser):
        def __init__(self,pending=False):super().__init__();self.pending=pending
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key not in ['href','src']or not value:continue
                u=urlsplit(value)
                if u.scheme or u.netloc or not u.path:continue
                target=(DOCS/unquote(u.path)).resolve()
                if self.pending and target==DOCS/'publication.json':continue
                if not target.exists():raise ValueError('Broken final report link: '+value)
    Links(pending=True).feed((DOCS/'results.html').read_text());sources={}
    for p in sorted(HERE.iterdir()):
        if p.suffix not in ['.py','.cpp','.sh']:continue
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        sources[str(p.relative_to(ROOT))]=sha(p)
    write(DOCS/'campaign-completion.json',dict(status='complete',original_plan_sha256=sha(DOCS/'campaign.json'),successful_fits=4,successful_managed_runs=8,
      successful_optimizer_updates=9600,discarded_partial_attempt_sha256=sha(BASE/'failed-unrolled-attempt-01/summary.json'),completion_audit_sha256=sha(DOCS/'completion-audit.json'),promotion=False))
    artifacts={str(p.relative_to(ROOT)):sha(p)for p in sorted(DOCS.rglob('*'))if p.is_file()and p.name!='publication.json'}
    write(DOCS/'publication.json',dict(status='complete',artifacts=artifacts,sources=sources,report_source_sha256=sha(HERE/'final_report.py'),finalizer_source_sha256=sha(__file__),promotion=False))
    Links().feed((DOCS/'results.html').read_text())
    # Recheck the audit closure after rendering. Original campaign and fitting
    # registration documents remain immutable; completion has its own record.
    for name,expected in audit['evidence'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Rendering changed audited evidence: '+name)
    write(BASE/'work-state.json',dict(status='complete',successful_fits=4,successful_managed_runs=8,live_owned_workers=[],publication_sha256=sha(DOCS/'publication.json')))
    print('Verified final report, all local links, audited evidence and exact local publication',flush=True)
if __name__=='__main__':run()
