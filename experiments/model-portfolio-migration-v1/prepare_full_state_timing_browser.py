"""Prepare read-only native chart checks from the exact admitted timing packets.

Expected counts and maxima come from retained scalar bytes. This prepares UI
actions only; it does not publish data, execute models, or accept rendered UI.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/model-portfolio-migration-v1'


def main():
    job = BASE / 'native-frozen-full-state-timing-import-01'
    state = json.loads((job / 'status.json').read_bytes())
    if state['status'] != 'two_frozen_full_state_GPU_timing_imports_native_metadata_and_maxima_passed':
        raise ValueError('Original native import must finish before UI checks')
    destination = job / 'browser-actions-01.json'
    if destination.exists():
        raise ValueError('Preserve the original UI action sequence')
    actions = [dict(kind='assert_text', text=state['last_acknowledged_checkpoint']),
               dict(kind='click', selector='#project-measurements-sources'),
               dict(kind='wait_text', selector='#project-measurements-source-page', text='1–19 of 19 sources'),
               dict(kind='wait_enabled', selector='#project-measurements-source')]
    expected = []

    def add(kind, **values):
        actions.append(dict(kind=kind, **values))

    for family in ('cell', 'temporal'):
        admitted = next(s for s in state['native_sources'] if s['family'] == family)
        raw = (ROOT / admitted['source']['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != admitted['id']:
            raise ValueError('Admitted original packet bytes changed')
        packet = json.loads(raw)
        add('select_values', selector='#project-measurements-source',
            values=[admitted['origin_catalog_id'] + '.' + admitted['id']])
        add('click', selector='#project-measurements-load')
        n = len(packet['subjects'])
        add('wait_text', selector='#project-measurements-subject-page', text=f'{n} selected across all pages')
        add('wait_enabled', selector='#project-measurements-compare')
        add('click', selector='#project-measurements-form > details > summary')
        previous = {}
        if family == 'cell':
            queries = [
                ('cell-K4-cpm100-anchor-time', 'per_output_median_ms',
                 dict(cells_per_metre=100, K=4, start_index=192,
                      trajectory_id='diversity-development-g025-cpm100')),
                ('cell-K4-cpm100-anchor-error', 'pressure_relative_rms-block-worst',
                 dict(cells_per_metre=100, K=4, start_index=192,
                      trajectory_id='diversity-development-g025-cpm100')),
                ('cell-K4-cpm100-grid64-time', 'per_output_median_ms',
                 dict(cells_per_metre=100, K=4, grid_x=64, grid_y=64, grid_z=64))]
        else:
            queries = [(f'temporal-K4-cpm50-{rate}-time', 'per_output_median_ms',
                        dict(cells_per_metre=50, K=4, output_steps_per_s=rate))
                       for rate in (120, 60, 30)]
            queries.append(('temporal-K4-cpm50-30-error', 'pressure_relative_rms-block-worst',
                            dict(cells_per_metre=50, K=4, output_steps_per_s=30)))
        for name, metric, filters in queries:
            for index, dimension in enumerate(packet['dimensions']):
                key = dimension['id']
                if key in filters:
                    add('fill', selector=f'#project-measurements-filter-mode-{index}', value='value')
                    add('fill', selector=f'#project-measurements-filter-value-{index}', value=str(filters[key]))
                elif key in previous:
                    add('fill', selector=f'#project-measurements-filter-mode-{index}', value='any')
            previous = filters
            cases = {c['id'] for c in packet['cases']
                     if all(c['dimensions'].get(k) == v for k, v in filters.items())}
            rows = [r for r in packet['observations'] if r['case_id'] in cases and r['metric_id'] == metric]
            if not cases or len(rows) != len(cases) * n or any(r['state'] != 'measured' for r in rows):
                raise ValueError('Explicit complete measured UI population required')
            maxima = {s['id']: max(r['value'] for r in rows if r['subject_id'] == s['id'])
                      for s in packet['subjects']}
            expected.append(dict(name=name, packet_sha256=admitted['id'], metric_id=metric,
                                 filters=filters, cases=len(cases), observations=len(rows), maxima=maxima))
            add('fill', selector='#project-measurements-metric', value=metric)
            add('click', selector='#project-measurements-compare')
            add('wait_text', selector='#project-measurements-counts',
                text=f'{n} selected subjects of {n}; {len(rows)} matching observations.')
            add('wait_text', selector='#project-measurements-status', text='Exact retained comparison.')
            # UI uses JavaScript's shortest float representation. Numeric maxima
            # are independently checked from API responses; chart labels here
            # assert subject identity and units without Python/JS repr coupling.
            for subject in packet['subjects']:
                add('wait_text', selector='#project-measurements-maxima', text=subject['label'])
            add('scroll', selector='#project-measurements-heading')
            add('screenshot', name=name + '.png')
        subject = packet['subjects'][-1]
        add('click', selector='#project-measurements-maxima button', text='inspect maximum for ' + subject['label'])
        add('wait_text', selector='#project-measurements-inspection', text=subject['label'])
        add('click', selector='#project-measurements-inspection > details:last-child summary')
        add('wait_text', selector='#project-measurements-inspection', text=admitted['id'])
        add('scroll', selector='#project-measurements-inspection')
        add('screenshot', name=family + '-GPU-original-source-inspection.png')
    destination.write_text(json.dumps(actions, indent=2) + '\n')
    (job / 'browser-expectations-01.json').write_text(json.dumps(expected, indent=2) + '\n')
    print(json.dumps(dict(actions=len(actions), comparisons=len(expected), path=str(destination.relative_to(ROOT)))))


if __name__ == '__main__':
    main()
