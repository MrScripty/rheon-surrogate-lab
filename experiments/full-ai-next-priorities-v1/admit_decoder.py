"""Admit one bounded full-state expert fit from completed prior evidence."""
from study_support import HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'temporal-decoder-admission-01.json'
    if output.exists():raise ValueError('Fresh admission required; no overwrite')
    prior=read(DOCS/'temporal-pressure-expert-results.json')
    if prior['status']!='completed'or prior['accuracy_qualified_small_grid_recipe']:
        raise ValueError('Completed failed representation comparison required')
    port=read(DOCS/'temporal-pressure-core-port-results.json')
    if port['status']!='passed_component_qualification':raise ValueError('Qualified learned pressure port required')
    qualifications={}
    for device in ('cpu','cuda'):
        path=BASE/f'temporal-decoder-trainer-{device}-qualification-01.json';qualified=read(path)
        if qualified['status']!='passed'or qualified['optimizer_updates']!=0 or len(qualified['rows'])!=96:
            raise ValueError('Complete actual constructor/gradient qualification required')
        if not qualified['actual_trainer_constructor_exercised']or not qualified['actual_objective_constructor_exercised']:
            raise ValueError('Actual managed constructors required')
        for name,digest in qualified['source_sha256'].items():
            if sha(HERE/name)!=digest:raise ValueError('Qualified expert source changed')
        if qualified['registration_sha256']!=sha(DOCS/'temporal-decoder-registration-01.json'):
            raise ValueError('Fit registration changed')
        qualifications[device]=sha(path)
    write(output,dict(status='admitted',model='b2-endpoint-decoder',updates=10000,
        registration_sha256=sha(DOCS/'temporal-decoder-registration-01.json'),
        prior_comparison_sha256=sha(DOCS/'temporal-pressure-expert-results.json'),
        pressure_port_result_sha256=sha(DOCS/'temporal-pressure-core-port-results.json'),qualification_sha256=qualifications,
        admission_source_sha256=sha(__file__),reserved_scene_capture_admitted=False,optimizer_updates_executed=0,
        reason='Completed expert fit fails accuracy and long-rollout stability; predeclared controls identify pressure/lead-time/tracer decoder weaknesses',
        comparison='Combined decoder representation follow-up with common fluid/request initialization and unchanged inherited training; not a single-factor attribution',
        inherited_training='C107-derived frozen pressure parameters and prior11000-update causal request fit; all provenance retained',
        original_goal_complete=False))
    print('Admitted one bounded K2 endpoint pressure-expert fit after CPU/CUDA two-block qualification',flush=True)


if __name__=='__main__':run()
