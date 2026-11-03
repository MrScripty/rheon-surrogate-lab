"""Render the standalone dashboard from maintainable sources and retained data.

Refresh importers own experimental records. This renderer owns presentation and
derives an explicit measurement catalog; it does not run inference or training.
"""
from datetime import date
from pathlib import Path
import json
import re

from evaluation_measurements import measurement_catalog

DATA = re.compile(r'(<script id="chart-data" type="application/json">)(.*?)(</script>)', re.S)
GROUPS = [
    ('Compare models', [('overview', 'Pressure accuracy & speed'), ('partial', 'Hybrid solver accuracy & speed'),
                        ('multistep', 'Multi-step prediction'), ('expanded', 'Expanded CPU & GPU models'), ('pilots', 'Temporal pilots & cadence'), ('progress', 'Training progress')]),
    ('Simulation performance', [('policies', 'Solver policies'), ('outcomes', 'Native outcomes'),
                                ('controlled', 'Controlled comparisons')]),
    ('Diagnose errors', [('drift', 'Drift & error growth'), ('passsweep', 'Correction budgets'),
                         ('domains', 'Hardest resolutions'), ('factors', 'Domain size & exposure'),
                         ('spatial', 'Spatial frequency'), ('spectrum', 'Temporal spectrogram'),
                         ('precision', 'Precision & numerical drift')]),
    ('Evidence', [('methodology', 'Benchmarks & definitions'), ('history', 'Complete result history')]),
    ('Study reports', [('fullai', 'Full AI execution study'), ('methods', 'Pressure methods I'),
                       ('methodsv2', 'Pressure methods II'), ('methodsv3', 'Pressure methods III'),
                       ('methodsv4', 'Pressure methods IV'), ('methodsv5', 'Pressure methods V'),
                       ('methodsv6', 'Pressure methods VI')]),
]


def navigation():
    nav = ['<label class="control analysis-jump-label">Analysis<select id="analysis-jump">']
    for title, tabs in GROUPS:
        nav.append(f'<optgroup label="{title}">')
        nav.extend(f'<option value="{key}">{label}</option>' for key, label in tabs)
        nav.append('</optgroup>')
    nav.append('</select></label><nav id="analysis-tabs" role="tablist" aria-label="Evaluation analyses" aria-orientation="vertical">')
    for title, tabs in GROUPS:
        archive = title == 'Study reports'
        nav.append('<details class="nav-studies"><summary>Study reports</summary>' if archive
                   else f'<div class="nav-group"><div class="nav-heading">{title}</div>')
        for key, label in tabs:
            button = (f'<button type="button" role="tab" id="tab-{key}" aria-controls="panel-{key}" '
                      f'aria-selected="false" tabindex="-1" data-analysis="{key}"><span>{label}</span>'
                      f'<small id="tab-state-{key}">Independent controls</small></button>')
            if key == 'precision':
                button = '<!-- BEGIN PRECISION TAB -->\n' + button + '\n<!-- END PRECISION TAB -->'
            nav.append(button)
        nav.append('</details>' if archive else '</div>')
    nav.append('</nav>')
    return '\n'.join(nav)


def render_dashboard(text, base):
    base = Path(base)
    match = DATA.search(text)
    if not match:
        raise ValueError('Dashboard data missing')
    data = json.loads(match[2])
    data['evaluation'] = measurement_catalog(data, base)
    source = base / 'dashboard'
    read = lambda name: (source / name).read_text()
    app = read('app.js')
    for marker, name in [('FILTERS','filters'),('HISTORY','history'),('PRESSURE','pressure'),
                         ('POLICIES','policies'),('FREQUENCY','frequency'),('DOMAINS','domains'),
                         ('DRIFT','drift'),('BUDGETS','budgets'),('STUDIES','studies')]:
        app = app.replace('@@' + marker + '_JS@@', read(name + '.js'))
    for marker, content in [
        ('COMPARISON_JS', read('comparison.js')),
        ('NAVIGATION_JS', read('navigation.js')),
        ('PRECISION_JS', (base / 'pressure-precision-drift-v1/chart.js').read_text()),
        ('FULL_AI_JS', (base / 'full_ai_study_ui.js').read_text()),
    ]:
        app = app.replace('@@' + marker + '@@', content)
    app = read('measurements.js') + '\n' + app
    # Declarations precede initialization even for modules with top-level state.
    seam = 'setupCohortSelectors();setupEvaluation();'
    if app.count(seam) != 1:
        raise ValueError('Evaluation initialization seam missing or ambiguous')
    app = app.replace(seam, read('multistep.js') + '\n' + read('expanded.js') + '\n' + read('pilots.js') + '\n' + seam)
    result = read('shell.html')
    for marker, content in [('STYLES', read('styles.css')), ('NAVIGATION', navigation()),
                            ('NEW_PANELS', read('panels.html')+'\n'+read('expanded.html')+'\n'+read('pilots.html')),
                            ('PRESSURE_CONTROLS', read('pressure-controls.html')),
                            ('APPLICATION', app)]:
        result = result.replace('@@' + marker + '@@', content)
    # Live study links and precision sections keep their established refresh seam.
    region = r'<!-- BEGIN FOUR PASS STUDY -->.*?<!-- END FOUR PASS STUDY -->'
    links = re.search(region, text, re.S)
    if links:
        result = re.sub(region, lambda _: links[0], result, count=1, flags=re.S)
    from precision_chart import precision_markup
    result = precision_markup(result, base)
    snapshot = date.fromisoformat(data['snapshot_date']).strftime('%d %b %Y')
    result = re.sub(r'Evidence snapshot · [^<]+', 'Evidence snapshot · ' + snapshot, result)
    if re.search(r'@@(?!CHART_DATA@@)[A-Z_]+@@', result):
        raise ValueError('Unresolved dashboard source token')
    encoded = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    return result.replace('@@CHART_DATA@@', encoded)


if __name__ == '__main__':
    base = Path(__file__).resolve().parent
    page = base / 'cohort-progress.html'
    page.write_text(render_dashboard(page.read_text(), base))
    print('Built standalone cohort-progress.html from dashboard/ sources; retained records unchanged.')
