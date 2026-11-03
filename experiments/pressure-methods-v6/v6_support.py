"""Isolated follow-up study; all completed studies remain immutable."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'runs/pressure-methods-v6'
DOCS=ROOT/'docs/plans/pressure-methods-v6'
sys.path[:0]=[str(ROOT/'experiments'/n) for n in ['pressure-four-pass-distillation-v1','pressure-pass-sweep-v1','pressure-methods-v1','pressure-methods-v2','pressure-methods-v3','pressure-methods-v4','pressure-methods-v5']]
sys.path.insert(0,str(HERE))
from common import read,write,sha,environment

def fit_record(arm):
    record=read(DOCS/"fit-records.json")["arms"][arm]
    if record["identifier"]!=arm:raise ValueError("Managed fit identifier differs")
    return record
