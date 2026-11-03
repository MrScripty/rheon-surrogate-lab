"""Prepare real UI checks of the two already imported pressure timing tables.

Only writes bounded browser actions beside the original acknowledged importer.
No browser, project mutation, model execution or benchmark timing occurs here.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
JOB = RUN/'native-pressure-portfolio-GPU-timing-import-01'


def prepare():
    state = json.loads((JOB/'status.json').read_bytes())
    if state['status'] != 'two_pressure_portfolio_GPU_timing_imports_native_metadata_and_maxima_passed':
        raise ValueError('Original native import must be acknowledged before UI checks')
    output = JOB/'browser-actions.json'
    if output.exists():
        raise ValueError('Preserve the original explicit browser action list')
    sources = state['native_sources']
    locators = [s['origin_catalog_id']+'.'+s['id'] for s in sources]
    actions = []

    def action(kind, **values):
        actions.append(dict(kind=kind, **values))

    def choose(values, count):
        action('wait_enabled',selector='#project-measurements-source')
        action('select_values',selector='#project-measurements-source',values=values)
        action('click',selector='#project-measurements-load')
        action('wait_text',selector='#project-measurements-subject-page',text=f'{count} selected across all pages')
        action('wait_enabled',selector='#project-measurements-compare')

    def compare(metric, selected, total, observations, screenshot):
        action('fill',selector='#project-measurements-metric',value=metric)
        action('click',selector='#project-measurements-compare')
        action('wait_text',selector='#project-measurements-counts',
               text=f'{selected} selected subjects of {total}; {observations} matching observations.')
        action('wait_text',selector='#project-measurements-status',text='Exact retained comparison.')
        action('scroll',selector='#project-measurements-heading')
        action('screenshot',name=screenshot)

    action('assert_text',text=state['last_acknowledged_checkpoint'])
    action('click',selector='#project-measurements-sources')
    action('wait_text',selector='#project-measurements-source-page',text='1–21 of 21 sources')
    choose(locators,127)
    compare('equation_relative_l2',127,127,127*128,'all127-equation.png')
    compare('pressure_call_ms',127,127,127*128,'all127-GPU-call.png')
    action('click',selector='#project-measurements-form > details > summary')
    action('fill',selector='#project-measurements-filter-mode-3',value='value')
    action('fill',selector='#project-measurements-filter-value-3',value='8')
    compare('pressure_call_ms',127,127,127*64,'density8-GPU-call.png')
    for i in range(3):
        action('fill',selector=f'#project-measurements-filter-mode-{i}',value='value')
        action('fill',selector=f'#project-measurements-filter-value-{i}',value='16')
    packet = json.loads((RUN/'native-pressure-portfolio-GPU-timing-01/part-01.json').read_bytes())
    matched = sum(c['dimensions']['cells_per_metre'] == 8 and
                  all(c['dimensions'][axis] == 16 for axis in ('grid_x','grid_y','grid_z'))
                  for c in packet['cases'])
    if not matched:
        raise ValueError('Expected real density8 / 16-cubed cases')
    compare('pressure_mean_free_relative_l2',127,127,127*matched,'density8-grid16-pressure.png')
    for i in range(4):
        action('fill',selector=f'#project-measurements-filter-mode-{i}',value='any')
    action('fill',selector='#project-measurements-filter-mode-24',value='value')
    action('fill',selector='#project-measurements-filter-value-24',value='false')
    compare('pressure_call_p95_ms',127,127,117*128,'known-host-overlap-excluded-p95.png')
    # C106/C107 both occur on the first 40-subject page of the second packet.
    second = json.loads((RUN/'native-pressure-portfolio-GPU-timing-01/part-02.json').read_bytes())
    if not {'C106','C107'} <= {s['id'] for s in second['subjects'][:40]}:
        raise ValueError('Explicit comparison subjects moved off the selected page')
    choose([locators[1]],63)
    action('click',selector='#project-measurements-subjects-clear')
    action('select_values',selector='#project-measurements-subjects',values=['C106','C107'])
    action('wait_text',selector='#project-measurements-subject-page',text='2 selected across all pages')
    compare('equation_relative_l2',2,63,256,'C106-C107-equation.png')
    compare('pressure_call_ms',2,63,256,'C106-C107-GPU-call.png')
    action('click',selector='#project-measurements-maxima button',text='inspect maximum for C107 · GPU pressure call')
    action('wait_text',selector='#project-measurements-status',text='Exact retained observation and original source evidence.')
    action('wait_text',selector='#project-measurements-inspection',text='C107 · GPU pressure call')
    action('scroll',selector='#project-measurements-inspection')
    action('screenshot',name='C107-original-timing-evidence.png')
    output.write_text(json.dumps(actions,indent=2,ensure_ascii=False)+'\n')
    print(len(actions),'explicit native UI actions prepared; matched density/grid cases',matched)


if __name__ == '__main__':
    prepare()
