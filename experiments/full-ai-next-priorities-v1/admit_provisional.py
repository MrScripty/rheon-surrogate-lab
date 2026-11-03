"""Admit only the registered matched provisional-supervision continuations."""
from study_support import HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'temporal-provisional-admission-01.json'
    if output.exists():raise ValueError('Fresh continuation admission required')
    regpath=DOCS/'temporal-provisional-registration-01.json';reg=read(regpath)
    prior=read(DOCS/'temporal-coupled-results.json');diagnosis=read(DOCS/'temporal-coupled-transport-results.json')
    if prior['status']!='completed'or prior['accuracy_qualified_small_grid_recipe']or diagnosis['status']!='completed':
        raise ValueError('Completed failed coupled fit and transport diagnosis required')
    if sha(DOCS/'temporal-coupled-transport-results.json')!=reg['prior_diagnostic_result_sha256']:raise ValueError('Registered diagnostic evidence changed')
    evaluator_path=BASE/'temporal-coupled-evaluator-qualification-01/receipt.json';evaluator=read(evaluator_path)
    if evaluator['status']!='passed'or evaluator['evaluator_sha256']!=sha(HERE/'evaluate_coupled.py'):
        raise ValueError('Qualified unchanged full-state evaluator required')
    if evaluator['protocol_sha256']!=sha(DOCS/'temporal-coupled-evaluation-protocol-01.json'):raise ValueError('Original evaluation criteria changed')
    bindings={};inventories=[]
    for device in ('cpu','cuda'):
        path=BASE/f'temporal-provisional-trainer-{device}-qualification-01.json';q=read(path)
        if q['status']!='passed'or q['optimizer_updates']!=0 or len(q['rows'])!=32 or len(q['training_label_rows'])!=128:
            raise ValueError('Actual trainer, objective and labels must qualify')
        if not q['actual_trainer_constructor_exercised']or not q['actual_objective_constructor_exercised']:
            raise ValueError('Actual constructors required')
        if q['registration_sha256']!=sha(regpath)or q['config_sha256']!=reg['config_sha256']:raise ValueError('Qualified registration/config changed')
        for name,digest in q['source_sha256'].items():
            if sha(HERE/name)!=digest:raise ValueError('Qualified source changed')
        # The inference family and all its transitive model layers are unchanged.
        old=read(BASE/f'temporal-coupled-trainer-{device}-qualification-01.json')
        for name,digest in old['source_sha256'].items():
            if sha(HERE/name)!=digest:raise ValueError('Previously qualified inference dependency changed')
        bindings[device]=sha(path);inventories.append(q['matched_continuation_initial_inventory'])
    if inventories[0]!=inventories[1]:raise ValueError('CPU/CUDA continuation initialization differs')
    parent=BASE/'b2-coupled-pair-01/restore/weights.npz'
    if sha(parent)!=reg['continuation_weights_sha256']:raise ValueError('Registered parent changed')
    write(output,dict(status='admitted',models=reg['models'],updates_per_arm=4000,
        registration_sha256=sha(regpath),prior_comparison_sha256=sha(DOCS/'temporal-coupled-results.json'),
        qualification_sha256=bindings,evaluator_qualification_sha256=sha(evaluator_path),
        continuation_weights_sha256=sha(parent),config_sha256=reg['config_sha256'],
        actual_inherited_training='Same completed10000-update coupled checkpoint and unchanged frozen pressure/flux assets',
        source_sha256=sha(__file__),reserved_capture_admitted=False,optimizer_updates_executed=0,original_goal_complete=False))
    print('Admitted two matched4000-update continuations; only auxiliary objective weight differs',flush=True)


if __name__=='__main__':run()
