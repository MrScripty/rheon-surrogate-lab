"""Require actual trained-weight accuracy and the declared execution evidence."""
from study_support import ROOT,HERE,read,sha


def require(method,bank_sha256,scene_identifiers):
    spec=method['model'];path=ROOT/method['accuracy_summary']['path']
    if sha(path)!=method['accuracy_summary']['sha256']:raise ValueError('Declared full-state accuracy summary changed')
    value=read(path)
    if value['status']!='completed'or value['bank_sha256']!=bank_sha256 or value['metrics_sha256']!=sha(HERE/'temporal_metrics.py'):
        raise ValueError('Complete original every-field/own240 evaluation required')
    if len(value['scenes'])!=8 or {r['identifier']for r in value['scenes']}!=set(scene_identifiers):
        raise ValueError('Actual complete eight-scene population required')
    if value['model']['weights_sha256']!=spec['weights_sha256']or value['model']['horizons']!=spec['horizons']:
        raise ValueError('Accuracy belongs to a different trained checkpoint')
    complete=all(s['complete']for r in value['scenes']for s in r['summaries'])
    passed=complete and all(s['all_accuracy_criteria_pass']for r in value['scenes']for s in r['summaries'])
    if method['role']=='candidate'and not passed:raise ValueError('An application finalist must first pass all original accuracy criteria')
    if spec['policy']=='fast-float64':
        source=ROOT/method['execution_qualification']['path']
        if sha(source)!=method['execution_qualification']['sha256']:raise ValueError('Exact execution qualification changed')
        proof=read(source)
        if proof['status']!='passed'or not proof['all_physical_bytes_exact']or proof['outputs']!=(121*spec['horizons']+240)*8:
            raise ValueError('Every output must have exact float64 execution evidence')
        if proof['model']['weights_sha256']!=spec['weights_sha256']:raise ValueError('Execution proof belongs to different weights')
        for name in ('bundle_model_fast_preparation.py','fast_preparation_model_support.py'):
            if proof['source_sha256'][name]!=sha(HERE/name):raise ValueError('Qualified execution math changed')
    elif spec['policy']=='fast-float32-map':
        identity=value['model']
        if identity.get('application_policy')!=spec['policy']or identity.get('application_loader_sha256')!=sha(HERE/'application_model_support.py'):
            raise ValueError('Generic float32 deployment needs its own actual evaluation; old composition results cannot be inherited')
        if identity.get('pressure_execution_source_sha256')!=sha(HERE/'float32_learned_pressure_map.py'):
            raise ValueError('Evaluated float32 pressure arithmetic changed')
        if not identity.get('original_neural_tensor_inventory_unchanged'):raise ValueError('Declared neural tensor inventory changed')
    else:raise ValueError('Unknown unqualified deployment policy')
    return dict(all_eight_scene_accuracy_pass=passed,all_intervals_complete=complete,accuracy_summary_sha256=sha(path),
        actual_trained_horizons=spec['horizons'],policy=spec['policy'])
