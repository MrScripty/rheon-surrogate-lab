"""Read completed managed pressure studies without changing their checkout."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent.parent
SOURCE_LAB = LAB / 'runs/dependency-migration-v1/lab'
SOURCE_PLANS = SOURCE_LAB / 'docs/plans'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path, base=ROOT):
    return os.path.relpath(Path(path).resolve(), base)


def read(path, expected=None):
    path = Path(path)
    if expected is not None and digest(path) != expected:
        raise ValueError(f'Recorded evidence changed: {path}')
    return json.loads(path.read_text())


def discover():
    """Only published fits with complete development reports are eligible."""
    studies = []
    for path in sorted(SOURCE_PLANS.glob('pressure-*/reports/matched-training.json')):
        directory = path.parent
        regression_path = directory / 'regression-summary.json'
        if not regression_path.exists():
            continue
        matched = read(path)
        regression = read(regression_path)
        if regression.get('failures') or regression.get('candidate_promoted'):
            raise ValueError(f'Unexpected current-study outcome: {regression_path}')
        study = directory.parent.name
        candidates = matched.get('candidates') or matched.get('arms')
        if candidates is None and 'lowband' in matched:
            candidates = {'lowband': {**matched['lowband'], 'study': str(LAB / 'runs' / study / 'lowband-study-01')}}
        if not candidates:
            continue
        models = []
        for key, arm in candidates.items():
            study_root = Path(arm['study'])
            receipt_path = study_root / 'receipt.json'
            receipt = read(receipt_path, arm.get('receipt_sha256'))
            intent = read(study_root / 'intent.json')
            assembly_path = Path(intent['assembly'])
            read(assembly_path, receipt['assembly_sha256'])
            project = assembly_path.parent
            artifacts = study_root / 'restore'
            if (receipt.get('status') not in ('passed', 'completed') or receipt['optimizer_updates'] != 5000
                    or not all(receipt['artifact_parity'].values())
                    or digest(artifacts / 'weights.npz') != arm['weights_sha256']
                    or read(artifacts / 'receipt.json') != receipt['restore_receipt']):
                raise ValueError(f'Incomplete managed fit/restore: {study_root}')
            name = 'current-' + key if study == 'pressure-current-policy-v1' else key
            if key.replace('.', '', 1).isdigit():
                name = study_root.name.rsplit('-study-', 1)[0]
            models.append(dict(name=name, study_id=study,
                family=study.removeprefix('pressure-').removesuffix('-v1').replace('-', ' ').capitalize(),
                project=relative(project, LAB), artifacts=relative(artifacts, LAB),
                weights_sha256=arm['weights_sha256'], source_lab=relative(SOURCE_LAB, LAB),
                started_at=datetime.fromtimestamp(intent['started_unix'], timezone.utc).isoformat(),
                validation_inputs='runs/water-early-pressure-v1/control/project/inputs',
                training_source=relative(receipt_path), training_sha256=digest(receipt_path),
                study_source=relative(path), study_sha256=digest(path),
                config_source=relative(project / 'config.json')))
        timing_path = directory / 'serial-timing.json'
        decision_path = directory / 'qualification-decision.json'
        decision = read(decision_path)
        if decision.get('candidate_promoted') or (decision.get('unused_holdouts_consumed') or decision.get('unused_eight_consumed')):
            raise ValueError(f'Unexpected promotion/holdout disposition: {decision_path}')
        studies.append(dict(id=study, title=study.removeprefix('pressure-').removesuffix('-v1').replace('-', ' ').capitalize(),
            models=models, regression=regression, regression_source=relative(regression_path),
            timing=read(timing_path), timing_source=relative(timing_path),
            decision_source=relative(decision_path), matched_source=relative(path)))
    return sorted(studies, key=lambda s: min(m['started_at'] for m in s['models']))


def discover_completed_models():
    """Include completed fits even when their later rollout campaign failed."""
    models = [m for study in discover() for m in study['models']]
    for path in sorted(SOURCE_PLANS.glob('pressure-*/reports/completed-fit-custody.json')):
        custody = read(path)
        measurement = custody['measurement']
        study_root = Path(measurement['study'])
        receipt_path = study_root / 'receipt.json'
        receipt = read(receipt_path, measurement['receipt_sha256'])
        intent = read(study_root / 'intent.json')
        assembly = Path(intent['assembly'])
        read(assembly, receipt['assembly_sha256'])
        artifacts = study_root / 'restore'
        if (custody['status'] != 'passed' or custody['completed_updates'] != 5000
                or receipt['status'] not in ('passed', 'completed')
                or not all(receipt['artifact_parity'].values())
                or digest(artifacts / 'weights.npz') != measurement['weights_sha256']
                or read(artifacts / 'receipt.json') != receipt['restore_receipt']
                or any(r['state'] != 'completed' or not r['echild'] or not r['spawn_closed']
                       for r in custody['runs'].values())):
            raise ValueError(f'Incomplete published fit: {path}')
        study = path.parent.parent.name
        models.append(dict(name='full-coarse', study_id=study, family='Spatial coarse correction',
            project=relative(assembly.parent, LAB), artifacts=relative(artifacts, LAB),
            weights_sha256=measurement['weights_sha256'], source_lab=relative(SOURCE_LAB, LAB),
            started_at=datetime.fromtimestamp(intent['started_unix'], timezone.utc).isoformat(),
            validation_inputs='runs/water-early-pressure-v1/control/project/inputs',
            training_source=relative(receipt_path), training_sha256=digest(receipt_path),
            study_source=relative(path), study_sha256=digest(path),
            config_source=relative(assembly.parent / 'config.json'),
            qualification_source=relative(path.parent / 'pressure-qualification.json')))
    return sorted(models, key=lambda m: m['started_at'])


def write_admissions():
    for entry in discover_completed_models():
        target = ROOT / 'chart-diagnostics' / entry['study_id'] / entry['name'] / 'admission.json'
        record = dict(models=[entry])
        if target.exists():
            if read(target) != record:
                raise ValueError(f'Existing chart admission changed: {target}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record, indent=2) + '\n')
        print(entry['name'], relative(target))


if __name__ == '__main__':
    write_admissions()
