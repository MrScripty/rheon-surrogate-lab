"""Read sealed replica deployment and reserved evaluation records, without inference."""
import hashlib
import os
from pathlib import Path

from full_ai_execution_chart import STUDY, temporal_accuracy_rows


def append_replica_evaluations(models, executions, rows, base, lab, load):
    base, lab = Path(base), Path(lab)

    def witnessed(name, reserved=False):
        path = base / STUDY / name
        if not path.exists():
            return None
        value = load(path)
        if (value['status'] != 'completed' or value.get('model_promotion')
                or value.get('training_on_reserved')
                or bool(value.get('reserved_scenes_consumed')) != reserved):
            raise ValueError('Unqualified replica evaluation: ' + name)
        for source, expected in value['source_evidence'].items():
            target = lab / source
            if target.suffix == '.json':
                load(target, expected)
            elif hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError('Replica evidence changed: ' + source)
        return value

    # A completed report binds the replica fits, including all failed seeds.
    witnessed('fixed-viscosity-replication-results.json')
    catalogs = []
    for name, catalog, reserved in [
        ('fixed-viscosity-replication-cost-results.json', 'fixed-viscosity-replication-deployments-01/qualified-models.json', False),
        ('temporal-reserved-cost-results.json', 'temporal-finalists-01/reserved-qualified-models.json', True),
    ]:
        report = witnessed(name, reserved)
        if report is None:
            continue
        source = f'runs/{STUDY}/{catalog}'
        catalog_value = load(lab / source, report['source_evidence'][source])
        catalogs.append((catalog_value, reserved, STUDY + '/' + name))
    if any(reserved for _, reserved, _ in catalogs):
        witnessed('temporal-reserved-accuracy-results.json', True)

    def accuracy(reference, parent, population):
        value = load(lab / reference['path'], reference['sha256'])
        metadata = value['model']
        if (value['status'] != 'completed' or value['optimizer_updates']
                or value['cuda_initialized'] or value['candidate_numerical_pressure_solve']
                or value.get('model_promotion') or metadata['weights_sha256'] != parent['weights_sha256']
                or metadata['horizons'] != parent['horizons']
                or len(value['scenes']) != 8
                or any(s['split'] != population for s in value['scenes'])
                or (value['bank_sha256'] == parent['bank_sha256']) != (population == 'development')):
            raise ValueError('Replica accuracy task/population differs: ' + reference['path'])
        return value

    for catalog, reserved, report_source in catalogs:
        for method in catalog['methods']:
            metadata = method['model']
            parent = next(m for m in models if m.get('record_kind', 'fit') == 'fit'
                          and m['pair'] == Path(metadata['directory']).name)
            directory = lab / metadata['directory']
            load(directory / 'receipt.json', metadata['receipt_sha256'])
            for filename, expected in [('weights.npz', metadata['weights_sha256']),
                                       ('normalization.json', metadata['normalization_sha256'])]:
                if hashlib.sha256((directory / 'restore' / filename).read_bytes()).hexdigest() != expected:
                    raise ValueError('Deployment checkpoint changed: ' + str(directory))
            if (parent['weights_sha256'] != metadata['weights_sha256']
                    or parent['horizons'] != metadata['horizons']):
                raise ValueError('Deployment does not match admitted trained fit')
            load(lab / method['specification_path'], method['specification_sha256'])
            item = next((m for m in executions if m['identifier'] == method['identifier']), None)
            if item is None:
                # K=3 is characterization-only and absent from the development
                # finalist catalog. Its byte-qualified development capture is
                # the already admitted trained fit's capture.
                reference = (method.get('original_accuracy_summary') or dict(
                    path=os.path.relpath(base / parent['evaluation_source'], lab),
                    sha256=method['pre_freeze_development_qualification']['accuracy_summary_sha256'])) if reserved else method['accuracy_summary']
                value = accuracy(reference, parent, 'development')
                qualification = method.get('execution_qualification')
                if qualification:
                    proof = load(lab / qualification['path'], qualification['sha256'])
                    if (proof['status'] != 'passed' or not proof['all_physical_bytes_exact']
                            or proof['optimizer_updates'] or proof['model']['weights_sha256'] != parent['weights_sha256']):
                        raise ValueError('Accuracy inheritance lacks exact execution proof')
                elif value['model'].get('identifier') != method['identifier']:
                    raise ValueError('Changed precision needs its own execution accuracy')
                label = parent['label'] + (' · float32 map' if metadata['policy'] == 'fast-float32-map' else ' · float64 policy')
                item = dict(parent, identifier=method['identifier'], label=label, record_kind='execution',
                    parent_label=parent['label'], parent_weights_sha256=parent['weights_sha256'], optimizer_updates=0,
                    evaluation_source=os.path.relpath(lab / reference['path'], base),
                    accuracy_inheritance='byte-qualified' if qualification else 'actual execution evaluation',
                    qualification_source=os.path.relpath(lab / qualification['path'], base) if qualification else None,
                    execution_report_source=report_source,
                    training_scope='Execution policy reuses the parent fit and frozen learned assets; zero new optimizer updates.')
                executions.append(item)
                rows.extend(temporal_accuracy_rows(label, item['horizons'], value, item['evaluation_source']))
            if reserved:
                reference = method['accuracy_summary']
                value = accuracy(reference, parent, 'reserved')
                qualification = method['actual_reserved_qualification']
                if (qualification['accuracy_summary_sha256'] != reference['sha256']
                        or qualification['actual_trained_horizons'] != item['horizons']):
                    raise ValueError('Reserved qualification belongs to a different capture')
                source = os.path.relpath(lab / reference['path'], base)
                item['reserved_evaluation'] = dict(source=source, bank_sha256=value['bank_sha256'])
                rows.extend(dict(row, population='reserved', bank_sha256=value['bank_sha256'])
                            for row in temporal_accuracy_rows(item['label'], item['horizons'], value, source))
