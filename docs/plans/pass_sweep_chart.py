"""Attach the separately retained inference-policy campaign to the offline page."""
import json
from pathlib import Path


def enrich_pass_sweep(data,base):
    artifact=Path(base)/'pressure-pass-sweep-v1/chart-data.json'
    if artifact.exists():
        result=json.loads(artifact.read_bytes())
        if result['schema']!='rheon-inference-pass-sweep-chart-v1':
            raise ValueError('Unknown inference-pass study schema')
        data['pass_sweep']=result
