"""Prepare real native GPU charts after the four-table ingestion custody passes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
JOB = RUN/'native-common-generator-GPU-import-01'
PACKETS = RUN/'common-generator-GPU-measurements-01'


def read(path):
    return json.loads(path.read_bytes())


def run():
    destination = JOB/'browser-actions.json'
    if destination.exists():
        raise ValueError('Preserve original generator chart actions')
    audit = read(ROOT/'docs/plans/model-portfolio-migration-v1/reports/native-common-generator-GPU-import-custody-01.json')
    state = read(JOB/'status.json')
    if (audit['status'] != 'four_common_generator_GPU_tables_native_custody_exact'
            or state['status'] != 'four_common_generator_GPU_imports_native_metadata_and_maxima_passed'
            or audit['checkpoint'] != state['last_acknowledged_checkpoint']):
        raise ValueError('Complete native import and custody are required before chart checks')
    prep = read(PACKETS/'preparation.json')
    actions = [dict(kind='assert_text',text=audit['checkpoint']),
        dict(kind='click',selector='#project-measurements-sources'),
        dict(kind='wait_text',selector='#project-measurements-source-page',text='1–25 of 25 sources'),
        dict(kind='wait_enabled',selector='#project-measurements-source')]
    expected = []

    def add(kind,**values):
        actions.append(dict(kind=kind,**values))

    for part,admitted in zip(prep['packets'],state['imports'],strict=True):
        raw = (PACKETS/part['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != admitted['id'] or admitted['blob']['sha256'] != part['sha256']:
            raise ValueError('Admitted generator table differs from its qualified source')
        packet = json.loads(raw)
        k = part['K']
        add('select_values',selector='#project-measurements-source',values=[admitted['origin_catalog_id']+'.'+admitted['id']])
        add('click',selector='#project-measurements-load')
        count = len(packet['subjects'])
        add('wait_text',selector='#project-measurements-subject-page',text=f'{count} selected across all pages')
        add('wait_enabled',selector='#project-measurements-compare')
        add('click',selector='#project-measurements-form > details > summary')
        queries = [(f'K{k}-all-{metric}',metric,{}) for metric in
            ('pressure_relative_l2','velocity_relative_l2','tracer_relative_l2','equation_relative_l2',
             'block_median_ms','per_output_median_ms')]
        for density,size in ((8,16),(16,32)):
            for suffix,metric in (('pressure','pressure_relative_l2'),('time','per_output_median_ms')):
                queries.append((f'K{k}-cpm{density}-grid{size}-{suffix}',metric,
                    dict(cells_per_metre=density,grid_x=size,grid_y=size,grid_z=size)))
        previous = {}
        for name,metric,filters in queries:
            for index,dimension in enumerate(packet['dimensions']):
                key = dimension['id']
                if key in filters:
                    add('fill',selector=f'#project-measurements-filter-mode-{index}',value='value')
                    add('fill',selector=f'#project-measurements-filter-value-{index}',value=str(filters[key]))
                elif key in previous:
                    add('fill',selector=f'#project-measurements-filter-mode-{index}',value='any')
            previous = filters
            selected = {c['id'] for c in packet['cases'] if all(c['dimensions'].get(n) == v for n,v in filters.items())}
            rows = [o for o in packet['observations'] if o['case_id'] in selected and o['metric_id'] == metric]
            if not selected or len(rows) != count*len(selected) or any(o['state'] != 'measured' for o in rows):
                raise ValueError('Exact measured chart population required')
            maxima = {s['id']:max(o['value'] for o in rows if o['subject_id'] == s['id']) for s in packet['subjects']}
            expected.append(dict(name=name,K=k,packet_sha256=part['sha256'],metric_id=metric,filters=filters,
                cases=len(selected),observations=len(rows),maxima=maxima,
                device='CUDA',chart_statistic='Maximum of per-case block statistic over matching cases; not median across cases.'))
            add('fill',selector='#project-measurements-metric',value=metric)
            add('click',selector='#project-measurements-compare')
            add('wait_text',selector='#project-measurements-counts',
                text=f'{count} selected subjects of {count}; {len(rows)} matching observations.')
            add('wait_text',selector='#project-measurements-status',text='Exact retained comparison.')
            for subject in packet['subjects']:
                add('wait_text',selector='#project-measurements-maxima',text=subject['label'])
                if 'GPU' not in subject['label']:
                    raise ValueError('Every displayed subject must explicitly label GPU timing')
            add('scroll',selector='#project-measurements-heading')
            add('screenshot',name=name+'.png')
        subject = packet['subjects'][-1]
        add('click',selector='#project-measurements-maxima button',text='inspect maximum for '+subject['label'])
        add('wait_text',selector='#project-measurements-inspection',text=subject['label'])
        add('click',selector='#project-measurements-inspection > details:last-child summary')
        add('wait_text',selector='#project-measurements-inspection',text=admitted['id'])
        add('scroll',selector='#project-measurements-inspection')
        add('screenshot',name=f'K{k}-GPU-source-inspection.png')
    destination.write_text(json.dumps(actions,indent=2)+'\n')
    (JOB/'browser-expectations.json').write_text(json.dumps(expected,indent=2)+'\n')
    print(len(actions),'real chart actions prepared for40comparisons; no browser or model execution.')


if __name__ == '__main__':
    run()
