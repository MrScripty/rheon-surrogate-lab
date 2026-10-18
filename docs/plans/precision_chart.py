"""Retain a standalone precision tab when the cohort dashboard is refreshed."""
import json
from pathlib import Path
import re


def enrich_precision(data, base):
    source = Path(base) / 'pressure-precision-drift-v1/chart-data.json'
    if not source.is_file():
        raise FileNotFoundError(f'Cannot refresh precision evidence: {source}')
    precision = json.loads(source.read_text())
    if precision.get('schema') != 'rheon-storage-precision-chart-v1' or precision.get('status') != 'completed':
        raise ValueError('Precision evidence must be a completed storage-precision study')
    data['precision'] = precision


def _region(text, name, body, anchor, javascript=False):
    start = f'// BEGIN {name}' if javascript else f'<!-- BEGIN {name} -->'
    end = f'// END {name}' if javascript else f'<!-- END {name} -->'
    block = start + '\n' + body.rstrip() + '\n' + end + '\n'
    if start in text or end in text:
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError(f'Ambiguous dashboard region: {name}')
        return re.sub(re.escape(start) + r'.*?' + re.escape(end) + r'\n?', lambda _: block, text, count=1, flags=re.S)
    if text.count(anchor) != 1:
        raise ValueError(f'Dashboard seam missing or ambiguous: {anchor}')
    return text.replace(anchor, block + anchor, 1)


def precision_markup(text, base):
    templates = Path(base) / 'pressure-precision-drift-v1'
    tab = '<button type="button" role="tab" id="tab-precision" aria-controls="panel-precision" aria-selected="false" tabindex="-1" data-analysis="precision"><span>Precision drift</span><small id="tab-state-precision">Float32 / float64 controls</small></button>'
    text = _region(text, 'PRECISION TAB', tab, '<button type="button" role="tab" id="tab-drift"')
    text = _region(text, 'PRECISION PANEL', (templates / 'section.html').read_text(), '<div id="panel-drift"')
    text = _region(text, 'PRECISION CHART', (templates / 'chart.js').read_text(), 'const passVisibleCounts=', javascript=True)
    original = 'setupPassSweep();\nrender();'
    replacement = 'setupPassSweep();setupPrecision();\nrender();'
    if replacement not in text:
        if text.count(original) != 1:
            raise ValueError('Dashboard initialization seam missing or ambiguous')
        text = text.replace(original, replacement, 1)
    original = 'const id=panel.id.slice(6),scope=panel.dataset.scope;'
    extra = "\n  if(id==='precision'){$('tab-state-precision').textContent=$('precision-scene').selectedOptions[0]?.textContent||'Precision controls';return}"
    if original + extra not in text:
        if text.count(original) != 1:
            raise ValueError('Dashboard summary seam missing or ambiguous')
        text = text.replace(original, original + extra, 1)
    return text


def main():
    base = Path(__file__).resolve().parent
    page = base / 'cohort-progress.html'
    text = page.read_text()
    match = re.search(r'(<script id="chart-data" type="application/json">)(.*?)(</script>)', text, re.S)
    if not match:
        raise ValueError('Dashboard data missing')
    data = json.loads(match[2])
    enrich_precision(data, base)
    text = text[:match.start(2)] + json.dumps(data, separators=(',', ':'), allow_nan=False) + text[match.end(2):]
    page.write_text(precision_markup(text, base))


if __name__ == '__main__':
    main()
