"""Author two real retained model definitions through native project owners.

These are new editable source definitions referencing retained checkpoints.
They do not recreate old training authority or claim a new fit. Original Run
history remains a separate extraction/migration obligation.
"""
import copy
import hashlib
import json
from pathlib import Path

from native_project_owner import run as owner_call


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/model-portfolio-migration-v1'
CONTROLLER = BASE / 'native-controller-01'
OUTPUT = BASE / 'native-authoring-01'
PROJECT = ROOT / 'projects/rheon-model-portfolio'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def perform(number, action, body):
    path = OUTPUT / (number + '-' + action + '.body.json')
    prefix = OUTPUT / (number + '-' + action)
    if path.exists():
        if json.loads(path.read_text()) != body:
            raise ValueError('Resumed native authoring body differs from retained request')
    else:path.write_text(json.dumps(body, indent=2) + '\n')
    response = prefix.with_suffix('.response.json')
    if not response.exists():
        if prefix.with_suffix('.request.json').exists():
            raise ValueError('Uncertain owner request requires explicit reconciliation')
        owner_call(CONTROLLER, 'project_files.' + action, path, prefix)
    value = json.loads(response.read_text())
    if value['status'] != 200:raise ValueError('Retained native owner error requires review')
    result = value['result']
    return result['value'] if set(result) == {'binding','checkpoint','project_id','value'} else result


def run():
    project_id = json.loads((PROJECT / '.tuldok/project.json').read_text())['project_id']
    material = json.loads((BASE / 'discovery/pressure-material-01.json').read_text())
    entries = {m['id']: m for m in material['models']}
    old_models = []
    for path in (BASE / 'discovery/catalog-01/calls').glob('*-models.response.json'):
        old_models.extend(json.loads(path.read_text()).get('models', []))
    original_project = json.loads((ROOT / entries['C107']['project'] / 'project.json').read_text())
    originals = [m for m in old_models if isinstance(m.get('source_selector'), dict)
                 and m['source_selector']['project_id'] == original_project['id']]
    if len(originals) != 1:raise ValueError('One exact retained C107 source definition required')
    c107 = copy.deepcopy(originals[0])
    task = dict(c107['task'], catalog_id=project_id)
    results = []
    for slot, identifier in enumerate(('C107', 'C128')):
        entry = entries[identifier]
        source_path = ROOT / entry['project'] / 'project.json'
        source = json.loads(source_path.read_text())
        train = next(r for r in source['recipes'] if r['id'] == 'train')
        if sha(ROOT / train['path']) != train['sha256']:
            raise ValueError('Retained source recipe changed')
        perform(str(10 + slot * 4), 'source_register', dict(root='lab', manifest_path=str(source_path.relative_to(ROOT))))
        if identifier == 'C107':
            body = {k: copy.deepcopy(c107[k]) for k in ('task', 'kind', 'specification',
                    'source_selector', 'preprocessing_policy', 'output_policy')}
            body['task'] = task
        else:
            # C128 has no old bound ModelDefinition. Author its actual restored
            # architecture and checkpoint as a captured-source definition under
            # the same real pressure field Task. The shared old equation and
            # all128 input/output contracts have been independently verified.
            restore = ROOT / entry['restore']
            receipt = json.loads((restore / 'receipt.json').read_text())
            body = dict(task=task, kind='captured_source',
                specification=dict(format='rheon_retained_pressure_checkpoint_v1',
                    architecture=receipt['architecture'], checkpoint_sha256=entry['weights_sha256'],
                    source_restore_receipt_sha256=sha(restore / 'receipt.json'),
                    scope='Retained fitted checkpoint; no new training or execution authority.'),
                source_selector=dict(project_id=source['id'], recipe_id='train', expected_sha256=train['sha256']),
                preprocessing_policy=dict(kind='captured_rheon_physical_residual',
                    normalization_sha256=sha(restore / 'normalization.json'),
                    source='Captured model.py predict_pressure; dimensions, cell_size_m, rho and dt remain physical inputs.'),
                output_policy=dict(kind='full_pressure', dtype='float64', axes=['z','y','x'],
                    centering='cell', gauge='cell_zero', units='Pa', layout='x_fast',
                    correction=receipt['architecture']['correction']))
        body['name'] = identifier + ' · ' + entry['name'] + ' · retained source'
        created = perform(str(11 + slot * 4), 'model_create', body)
        if 'id' not in created or created['revision'] != 1:
            raise ValueError('Unexpected native model-create result; inspect retained response')
        published = perform(str(12 + slot * 4), 'model_publish', dict(model_id=created['id'], revision=1))
        results.append(dict(identifier=identifier, model_definition=created, published=published,
                            original_checkpoint_sha256=entry['weights_sha256'],
                            source_project_id=source['id'], source_manifest_sha256=sha(source_path)))
    (OUTPUT / 'representative-models.json').write_text(json.dumps(dict(
        project_id=project_id, models=results, new_parameter_fits=0,
        original_training_history_migrated=False, full_migration_complete=False), indent=2) + '\n')
    print('Published two native captured-source model definitions; no training or inference executed', flush=True)


if __name__ == '__main__':run()
