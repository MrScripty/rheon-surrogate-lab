"""Verify retained evidence, refresh preservation, UI receipts and measured bounds."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from build import read, sha


def chart(path):
    return json.loads(re.search(r'<script id="chart-data" type="application/json">(.*?)</script>', path.read_text(), re.S)[1])


def verify(build, run, output, config, chart_key):
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    base = root / 'docs/plans'
    sys.path.insert(0, str(base))
    from precision_chart import enrich_precision, precision_markup

    page = base / 'cohort-progress.html'
    before = chart(build / 'selection-chart.html')
    after = chart(page)
    changed = [key for key in before if before[key] != after.get(key)]
    assert set(changed) <= {'history_refreshed_utc'}, changed
    assert set(after) - set(before) == {chart_key}
    data = read(output / 'chart-data.json')
    assert after[chart_key] == data
    text = page.read_text()
    assert precision_markup(text, base) == text, 'Refresh must be idempotent'
    enriched = dict(before)
    enrich_precision(enriched, base)
    assert enriched[chart_key] == data
    try:
        precision_markup(text.replace('// END PRECISION CHART', '// END MISSING REGION'), base)
    except ValueError:
        pass
    else:
        raise AssertionError('Damaged dashboard region must fail explicitly')
    for source, expected in data['sources'].items():
        assert sha(base / source) == expected, source
    material = read(build / 'material.json')
    for source, expected in material['generated_sources'].items():
        assert sha(build / 'source' / source) == expected, source
    for source, expected in material['original_sources'].items():
        assert sha(root / material['config']['source'] / source) == expected, source
    assert sha(config) == material['config_sha256']
    build_receipt = read(build / 'build-receipt.json')
    assert sha(root / build_receipt['binary']) == build_receipt['sha256']
    assert all(c['returncode'] == 0 for c in build_receipt['commands'])
    admission = []
    scene = read(build / 'matched16-scene.json')
    with tempfile.TemporaryDirectory(prefix='precision-admission-') as temporary:
        for name, changes in [('zero_steps', {'steps': 0}), ('excess_horizon', {'steps': 3601}),
                              ('invalid_axis', {'counts': [4, 16, 16]}), ('excess_cells', {'counts': [128, 128, 128]})]:
            path = Path(temporary) / (name + '.json')
            destination = Path(temporary) / (name + '.jsonl')
            path.write_text(json.dumps({**scene, **changes}))
            command = [str(root / build_receipt['binary']), str(path), str(destination)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            assert result.returncode != 0 and not destination.exists(), name
            admission.append(dict(case=name, returncode=result.returncode, rejected_before_output=True, message=result.stderr.strip()))
    tests_command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).parent), '-p', 'test_build.py', '-v']
    tests = subprocess.run(tests_command, capture_output=True, text=True, timeout=30)
    assert tests.returncode == 0, tests.stdout + tests.stderr
    frames = 0
    checks = 0
    max_bound_ratio = 0.
    for trace in data['traces']:
        rows = [json.loads(line) for line in (run / trace['id'] / 'frames.jsonl').read_text().splitlines()]
        assert rows[-1]['kind'] == 'complete'
        assert [f['step'] for f in rows[:-1]] == list(range(1, trace['scene']['steps'] + 1))
        for frame in rows[:-1]:
            assert frame['canonical_f32_exact'] and frame['f64_repeat_exact']
            assert frame['f32_quality']['qualified'] and frame['f64_quality']['qualified']
            for name in ['advection', 'tracer_advection', 'force', 'viscosity', 'projection']:
                cast = frame['casts'][name]
                assert cast['cast_mismatches'] == 0
                assert cast['maximum'] <= cast['maximum_bound'] * (1 + 1e-12)
                checks += cast['values']
                if cast['maximum_bound']:
                    max_bound_ratio = max(max_bound_ratio, cast['maximum'] / cast['maximum_bound'])
            if trace['id'] == 'zero':
                assert all(frame[name]['rms'] == 0 for name in ['pressure', 'velocity', 'tracer'])
            frames += 1
    assert checks == data['verified_cast_values']
    browser = read(run / 'browser/browser-receipt.json')
    historical_browser = read(run / 'browser-historical/browser-receipt.json')
    for receipt in [browser, historical_browser]:
        assert receipt['status'] == 'passed' and not receipt['console_errors']
        assert receipt['html_sha256'] == sha(page), 'Browser must test the final dashboard'
    lifecycle = read(output / 'lifecycle-verification.json')
    assert lifecycle['status'] == 'passed' and lifecycle['all_observed_children_absent']
    assert lifecycle['native_binary_sha256'] == build_receipt['sha256']
    sources = {str(p.relative_to(root)): sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.is_file()}
    templates = base / 'pressure-precision-drift-v1'
    sources.update({str(p.relative_to(root)): sha(p) for p in [base / 'precision_chart.py', templates / 'chart.js', templates / 'section.html', page]})
    (output / 'source-closure.json').write_text(json.dumps(material, indent=2) + '\n')
    receipt = dict(status='passed', completed_utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_s=time.perf_counter()-started, physical_steps=frames, verified_cast_values=checks,
                   maximum_observed_cast_to_bound_ratio=max_bound_ratio,
                   historical_data=dict(unchanged_scientific_fields=True, allowed_metadata_changes=changed,
                                        baseline_sha256=sha(build / 'selection-chart.html')),
                   refresh=dict(command='python3 docs/plans/refresh-current-cohorts.py', returncode=0,
                                markup_idempotent=True, damaged_region_rejected=True),
                   tests=dict(command=tests_command, returncode=tests.returncode, output=tests.stdout+tests.stderr),
                   invalid_horizon_and_geometry_admission=admission,
                   browser=browser, historical_browser=historical_browser, lifecycle=lifecycle,
                   sources=sources, source_closure_sha256=sha(output / 'source-closure.json'),
                   standards_snapshot='snapshot:v1:ffcfc3f8-b52a-434e-b57f-d259a0355711',
                   prefix_exact_steps={s['id']:s.get('prefix_exact_steps')for s in data['summaries']},
                   command=[sys.executable, __file__, '--build', str(build), '--run', str(run), '--output', str(output),
                            '--config',str(config),'--chart-key',chart_key])
    (output / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status='passed', physical_steps=frames, verified_cast_values=checks, elapsed_s=receipt['elapsed_s'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('experiments/pressure-precision-drift-v1/config.json'))
    parser.add_argument('--chart-key', default='precision')
    args = parser.parse_args()
    verify(args.build.resolve(), args.run.resolve(), args.output.resolve(), args.config.resolve(), args.chart_key)
