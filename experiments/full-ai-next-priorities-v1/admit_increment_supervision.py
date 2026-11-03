"""Evaluate the predeclared next-loss experiment gate from completed evidence."""
from study_support import BASE,DOCS,read,write,sha


if __name__=='__main__':
    output=DOCS/'temporal-increment-supervision-admission-01.json'
    if output.exists():raise ValueError('Fresh evidence admission required')
    batch=read(BASE/'temporal-step-scaled-batch-01/execution.json')
    if batch['status']!='completed'or not batch['owned_child_reaped']:
        raise ValueError('Conditioning comparison must complete and reap first')
    result_path=DOCS/'temporal-step-scaled-results.json';result=read(result_path)
    diagnostic=BASE/'temporal-step-scaled-fitting-diagnosis-b2-endpoint-step-scaled-01.json'
    rows=read(diagnostic)['summaries']
    velocity=next(row['learned_median']for row in rows if row['split']=='train'and row['phase']=='cold'and row['field']=='velocity')
    triggered=not result['accuracy_qualified_small_grid_recipes']and velocity>.5
    write(output,dict(status='admitted'if triggered else'not_triggered',
        registration_sha256=sha(DOCS/'temporal-increment-supervision-registration-01.json'),
        prior_comparison_sha256=sha(result_path),prior_batch_sha256=sha(BASE/'temporal-step-scaled-batch-01/execution.json'),
        training_diagnostic_sha256=sha(diagnostic),endpoint_cold_training_velocity_median=velocity,
        prior_accuracy_qualified_recipes=result['accuracy_qualified_small_grid_recipes'],
        condition='No accuracy-qualified conditioning recipe and endpoint cold training velocity error>0.5',
        source_sha256=sha(__file__),optimizer_updates=0,new_training_runs_submitted=0))
    print('Increment-supervision follow-up','admitted'if triggered else'not triggered',flush=True)
