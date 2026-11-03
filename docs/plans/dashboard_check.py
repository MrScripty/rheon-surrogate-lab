"""Run dashboard interactions and reload checks in a local Chromium executable."""
import argparse
import html
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent


def check_browser(browser):
    with tempfile.TemporaryDirectory(prefix='rheon-dashboard-') as tmp:
        folder = Path(tmp)
        page = (ROOT / 'cohort-progress.html').read_text()
        capture = '''<script>window.__dashboardErrors=[];window.addEventListener('error',e=>window.__dashboardErrors.push(e.message+' @'+e.lineno));try{if(sessionStorage.getItem('rheon-qa-phase')!=='reload')localStorage.removeItem('rheon-evaluation-filters-v1')}catch{}</script>'''
        page = page.replace('<head>', '<head>' + capture, 1)
        page = page.replace('</body>', '<script>' + (ROOT / 'dashboard/browser-check.js').read_text() + '</script></body>')
        target = folder / 'check.html'
        target.write_text(page)
        reports = {}
        for name, size in [('desktop', '1440,1600'), ('mobile', '390,1000')]:
            run = subprocess.run([str(browser), '--no-sandbox', '--disable-gpu', '--no-proxy-server',
                '--user-data-dir=' + str(folder / name), '--window-size=' + size,
                '--virtual-time-budget=30000', '--dump-dom', target.as_uri()],
                capture_output=True, text=True, timeout=90, check=True)
            match = re.search(r'<pre id="qa-result">(.*?)</pre>', run.stdout, re.S)
            if not match:
                raise RuntimeError(f'No {name} browser result. {run.stderr[-1000:]}')
            reports[name] = json.loads(html.unescape(match[1]))
        return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', type=Path, required=True)
    args = parser.parse_args()
    results = check_browser(args.browser)
    print(json.dumps(results, indent=2))
    raise SystemExit(1 if any(r['failures'] for r in results.values()) else 0)
