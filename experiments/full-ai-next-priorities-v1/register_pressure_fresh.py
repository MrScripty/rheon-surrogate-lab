"""Reserve independent Track A scenes without consuming Track B's blind bank."""
import hashlib
import random
import time
from study_support import DOCS,read,write,sha

if __name__=='__main__':
    path=DOCS/'pressure-scene-reservation.json'
    if path.exists():raise ValueError('Track A scenes already reserved')
    rng=random.Random(2026101957);rows=[]
    for group,counts in enumerate(([16,16,16],[9,11,11],[15,6,8],[32,32,32])):
        for cpm in (8,16):
            for replica in (0,1):
                identifier=f'pressure-fresh-g{group}-cpm{cpm}-r{replica}'
                seed=(1<<50)+int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8],'little')% (1<<50)
                rows.append(dict(identifier=identifier,split='reserved_track_a',seed=seed,counts=list(counts),cells_per_metre=cpm,
                    steps=64,warmup_steps=0,center=[rng.uniform(.25,.65)for _ in range(3)],acceleration=rng.uniform(.08,.16),
                    radius=rng.uniform(.15,.25),swirl=rng.uniform(-.6,.6),fluctuation=rng.uniform(.1,.3),frequency=rng.uniform(1.,2.5),sgs=False))
    temporal=read(DOCS/'temporal-scene-reservation.json')
    if len({r['seed']for r in rows})!=len(rows)or {r['seed']for r in rows}&{r['seed']for r in temporal['rows']}:
        raise ValueError('Fresh pressure scenes overlap the temporal bank')
    write(path,dict(status='reserved_before_capture',registered_unix=time.time(),rows=rows,
        primary_registration_sha256=sha(DOCS/'registration.json'),source_sha256=sha(__file__),
        counts=dict(scenes=16,small_scenes=12,extension32_scenes=4,prepared_pressure_cases=1024),
        requirement='Capture only after three head seeds and the chosen prediction implementation are frozen',
        temporal_reserved_scenes_consumed=False,independent_from_temporal_bank=True,
        evaluation='All64consecutive pressure requests per scene; timings at steps1,16,32,64; native64-step attempts retain rejection',
        target_equation_error=7e-4,primary_timing_regime='Counts at most16 per axis;32cubed reported separately'))
    print('Reserved16independent pressure scenes; temporal blind bank remains untouched',flush=True)
