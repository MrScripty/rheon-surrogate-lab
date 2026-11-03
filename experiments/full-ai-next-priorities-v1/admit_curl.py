"""Bind the new representation trial to its completed matched-loss control."""
from study_support import BASE,DOCS,read,write,sha


if __name__=='__main__':
    output=DOCS/'temporal-curl-admission-01.json'
    if output.exists():raise ValueError('Fresh representation admission required')
    batch=read(BASE/'temporal-increment-supervision-batch-01/execution.json')
    if batch['status']!='completed'or not batch['owned_child_reaped']:raise ValueError('Matched supervision comparison must finish first')
    path=DOCS/'temporal-increment-supervision-results.json';result=read(path)
    fitting=next(row for row in result['fitting']if row['mode']=='increments')['summaries']
    cold=next(row['learned_median']for row in fitting if row['split']=='train'and row['phase']=='cold'and row['field']=='velocity')
    triggered=not result['accuracy_qualified_small_grid_recipes']and cold>.05
    write(output,dict(status='admitted'if triggered else'not_triggered',
        registration_sha256=sha(DOCS/'temporal-curl-registration-01.json'),prior_comparison_sha256=sha(path),
        prior_batch_sha256=sha(BASE/'temporal-increment-supervision-batch-01/execution.json'),
        direct_increment_cold_training_velocity_median=cold,prior_accuracy_qualified_recipes=result['accuracy_qualified_small_grid_recipes'],
        condition='No accuracy-qualified matched-loss recipe and remaining cold training velocity median>0.05',
        decision_timing='Defined from completed baseline evidence before candidate fitting',
        source_sha256=sha(__file__),optimizer_updates=0,new_training_runs_submitted=0))
    print('Curl representation trial','admitted'if triggered else'not triggered',flush=True)
