"""Freeze bounded training/development collection after completed native pilots."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DOCS=ROOT/'docs/plans/full-ai-data-diversity-v1'
BASE=ROOT/'runs/full-ai-data-diversity-v1'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__=='__main__':
    reservation=json.loads((DOCS/'scene-reservation-03.json').read_text())
    pilot=json.loads((DOCS/'pilot-and-coverage-results-01.json').read_text())
    if pilot['total_pilot_physical_steps']!=232 or pilot['total_pilot_stored_bytes']+22*1024**3>24*1024**3:
        raise ValueError('Measured capture must fit total declared storage budget')
    sources={str(p.relative_to(ROOT)):digest(p)for p in HERE.iterdir()if p.is_file()and p.suffix in('.py','.rs','.cpp')}
    build=json.loads((BASE/'native-forcing-build-02/build-receipt.json').read_text())
    sources.update(build['generated_sources'])
    for p in [DOCS/'scene-reservation-03.json',DOCS/'plan-03.json',DOCS/'pilot-and-coverage-results-01.json',
        BASE/'native-forcing-build-02/build-receipt.json',BASE/'native-forcing-build-02/rheon-temporal-probe',
        BASE/'forcing-capture-qualification-02/qualification.json',BASE/'known-force-build-01/receipt.json',
        BASE/'known-force-build-01/libscheduled_force.so']:
        sources[str(p.relative_to(ROOT))]=digest(p)
    for name,h in sources.items():
        if digest(ROOT/name)!=h:raise ValueError('Frozen capture source changed: '+name)
    for split,limit,wall in [('train',20*1024**3,10800),('development',2*1024**3,3600)]:
        rows=[r for r in reservation['rows']if r['split']==split]
        reg=dict(status='registered_after_pilot_before_bulk_capture',created_unix=time.time(),
            split=split,cpu=23,qualification=str((BASE/'forcing-capture-qualification-02/qualification.json').relative_to(ROOT)),
            rows=rows,sample_source_steps=[0,1,2,4,8,16,32,64,96,128,192,239],
            maximum_wall_seconds=wall,maximum_capture_bytes=limit,minimum_free_disk_bytes=12*1024**3,
            source_sha256=sources,reference_only=True,reserved_references_consumed=False,
            optimizer_updates=0,clock_not_model_inference=True)
        p=DOCS/f'{split}-capture-protocol-01.json'
        with p.open('x')as f:json.dump(reg,f,indent=2);f.write('\n')
        print('Registered',split,len(rows),'whole scenes',flush=True)
