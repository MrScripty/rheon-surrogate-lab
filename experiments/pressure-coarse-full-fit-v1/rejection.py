"""Classify current Rheon diagnostic attempts without relaxing scientific gates."""
import hashlib
import json
from pathlib import Path


def read(path):
    if path.stat().st_size>16*1024**2:raise ValueError('Bounded rejection evidence required')
    return json.loads(path.read_text())


def classify(error,directory):
    infrastructure=dict(failure_class='infrastructure',native_children_drained=False)
    if type(error)is not ValueError or str(error)not in('sparse capture schedule mismatch','incomplete or rejected native child'):
        return infrastructure
    directory=Path(directory)
    paths=[directory/'native/receipt.json',directory/'native-control/receipt.json',directory/'native/rejected-attempts.jsonl']
    try:
        learned,control=[read(p)for p in paths[:2]]
        for value in(learned,control):
            child=value['native_process']
            if not(value['status']=='completed'and value['error']is None and child['reaped']is True
                and child['returncode']==0 and child['terminated']is False
                and value['native_fallback']is False and value['native_pressure_refinement']is False
                and value['result']['native_fallback']is False and value['result']['native_pressure_refinement']is False):return infrastructure
        if not(control['result']['reference_control']is True and control['result']['hard_stop']is False
            and learned['result']['reference_control']is False and learned['result']['hard_stop']is True
            and learned['rejected_attempts']==1):return infrastructure
        if paths[2].stat().st_size>16*1024**2:return infrastructure
        attempts=[json.loads(line)for line in paths[2].read_text().splitlines()]
        if len(attempts)!=1:return infrastructure
        row=attempts[0];attempt=row['attempt'];outcome=attempt['outcome']
        if not(row['kind']=='diagnostic_attempt'and row['stop']=='Some(Rejected)'
            and attempt['dispatched']is True and attempt['history_transition']=='NotAdvanced'
            and attempt['native_fallback']is False and attempt['native_pressure_refinement']is False
            and outcome['kind']=='candidate'and outcome['qualified']is False):return infrastructure
        return dict(failure_class='scientific_rejection',native_children_drained=True,
            classification_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            rejection_evidence={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths})
    except(OSError,ValueError,KeyError,IndexError,TypeError):return infrastructure
