"""Keep the new matched study discoverable without merging historical populations."""
from pathlib import Path
import json
import re


def reserved_four_pass_ids(base):
    result = Path(base) / 'pressure-four-pass-distillation-v1/results.json'
    if not result.is_file():
        return set()
    data = json.loads(result.read_text())
    if data['schema'] != 'rheon-four-pass-results-v1' or data['status'] != 'completed':
        raise ValueError('Completed study required for its model identifiers')
    return {row['id'] for row in data['training']}


def four_pass_link(text, base):
    report = Path(base) / 'pressure-four-pass-distillation-v1/index.html'
    if not report.is_file():
        return text
    start, end = '<!-- BEGIN FOUR PASS STUDY -->', '<!-- END FOUR PASS STUDY -->'
    block = start + '\n<p class="note"><a href="pressure-four-pass-distillation-v1/index.html">New: matched four-pass CUDA training study</a> — C53/C55 baselines, C57–C59 fine-tunes, CPU accuracy per pass, fresh-simulation drift, timing and retained methods.</p>\n' + end + '\n'
    if start in text or end in text:
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError('Ambiguous study link region')
        return re.sub(re.escape(start) + r'.*?' + re.escape(end) + r'\n?', lambda _: block, text, count=1, flags=re.S)
    anchor = '<noscript>'
    if text.count(anchor) != 1:
        raise ValueError('Dashboard study link seam missing or ambiguous')
    return text.replace(anchor, block + anchor, 1)
