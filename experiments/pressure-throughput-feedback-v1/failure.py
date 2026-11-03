"""Classify only an explicit drained native numerical rejection as scientific."""
import json
from pathlib import Path


def classify(error, directory):
    result={'failure_class':'infrastructure','native_children_drained':False}
    if type(error) is not ValueError or str(error) not in (
            'sparse capture schedule mismatch','incomplete or rejected native child'):
        return result
    try:
        directory=Path(directory)
        learned=json.loads((directory/'native/receipt.json').read_text())
        control=json.loads((directory/'native-control/receipt.json').read_text())
        for receipt in (learned,control):
            child=receipt['native_process']
            if receipt['status']!='completed' or receipt['error'] is not None or not child['reaped'] or child['returncode']!=0 or child['terminated']:
                return result
        if control['result']['hard_stop'] or not control['result']['reference_control']:
            return result
        if not learned['result']['hard_stop'] or learned['rejected_attempts']<1:
            return result
        attempts=[json.loads(s) for s in (directory/'native/rejected-attempts.jsonl').read_text().splitlines()]
        outcome=attempts[-1]['outcome']
        if outcome['kind']=='candidate' and outcome['qualified'] is False:
            return {'failure_class':'scientific_rejection','native_children_drained':True}
    except (OSError,ValueError,KeyError,IndexError,TypeError):
        pass
    return result
