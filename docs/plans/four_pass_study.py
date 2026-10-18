"""Keep the new matched study discoverable without merging historical populations."""
from pathlib import Path
import json
import re


def reserved_four_pass_ids(base):
    experiments = Path(base).resolve().parents[1] / 'experiments'
    reserved = set()
    for study in ['pressure-low-mode-tail-v1', 'pressure-single-mode-fit-v1', 'pressure-fundamental-bank-fit-v1', 'pressure-shape-decay-v1', 'pressure-balanced-shapes-v1']:
        config = experiments / study / 'config.json'
        if config.is_file():
            reserved.update(row['model_id'] for row in json.loads(config.read_text())['arms'])
    result = Path(base) / 'pressure-four-pass-distillation-v1/results.json'
    if not result.is_file():
        return reserved
    data = json.loads(result.read_text())
    if data['schema'] != 'rheon-four-pass-results-v1' or data['status'] != 'completed':
        raise ValueError('Completed study required for its model identifiers')
    return {row['id'] for row in data['training']} | reserved


def four_pass_link(text, base):
    report = Path(base) / 'pressure-four-pass-distillation-v1/index.html'
    if not report.is_file():
        return text
    start, end = '<!-- BEGIN FOUR PASS STUDY -->', '<!-- END FOUR PASS STUDY -->'
    block = start + '\n<p class="note"><a href="pressure-four-pass-distillation-v1/index.html">New: matched four-pass CUDA training study</a> — C53/C55 baselines, C57–C59 fine-tunes, CPU accuracy per pass, fresh-simulation drift, timing and retained methods.</p>\n' + end + '\n'
    if (Path(base) / 'pressure-low-mode-tail-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-low-mode-tail-v1/index.html">Small-grid low-mode training screen: C60/C61</a> — matched CUDA training, CPU accuracy/time, mode/amplitude diagnostics and retained results.</p>\n' + end)
    if (Path(base) / 'pressure-single-mode-fit-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-single-mode-fit-v1/index.html">Fundamental fit and frequency curriculum: C62/C63</a> — learning curve, matched spectral exposure, restored CPU accuracy/time and retained results.</p>\n' + end)
    if (Path(base) / 'pressure-fundamental-bank-fit-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-fundamental-bank-fit-v1/index.html">Shared fundamental fitting capacity</a> — per-grid/axis capacity, learning curves, CPU regression accuracy and retained time/memory.</p>\n' + end)
    if (Path(base) / 'pressure-shape-decay-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-shape-decay-v1/index.html">Later-decay fitting comparison: C67 and conditional C68</a> — matched learning curves, shape-gradient relationships, restored CPU accuracy and time/memory.</p>\n' + end)
    if (Path(base) / 'pressure-loss-components-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-loss-components-v1/index.html">Loss-component diagnosis: C53/C66/C67</a> — final/early shape-gradient alignment, precision limitations and retained methods/timings.</p>\n' + end)
    if (Path(base) / 'pressure-balanced-shapes-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-balanced-shapes-v1/index.html">Balanced shape aggregation: C69</a> — matched example work, distinct Adam-update counts, CPU fitting/regression accuracy and retained time/memory.</p>\n' + end)
    if (Path(base) / 'pressure-aggregate-gradients-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-aggregate-gradients-v1/index.html">Shared gradient diagnosis: C53/C66/C69</a> — case influence, final/early directions, per-case norm caps and retained CPU timings/limits.</p>\n' + end)
    if (Path(base) / 'pressure-adam-replay-v1/index.html').is_file():
        block = block.replace(end, '<p class="note"><a href="pressure-adam-replay-v1/index.html">C69 Adam observation replay</a> — exact checkpoint replay, observed optimizer steps, pressure changes, timings and precision limits.</p>\n' + end)
    if start in text or end in text:
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError('Ambiguous study link region')
        return re.sub(re.escape(start) + r'.*?' + re.escape(end) + r'\n?', lambda _: block, text, count=1, flags=re.S)
    anchor = '<noscript>'
    if text.count(anchor) != 1:
        raise ValueError('Dashboard study link seam missing or ambiguous')
    return text.replace(anchor, block + anchor, 1)
