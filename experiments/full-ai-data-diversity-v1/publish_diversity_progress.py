"""Audit diversity splits and report each density/grid pilot separately."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[2]
DOCS=ROOT/'docs/plans/full-ai-data-diversity-v1'
BASE=ROOT/'runs/full-ai-data-diversity-v1'


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    output=DOCS/'pilot-and-coverage-results-01.json'
    if output.exists():raise ValueError('Fresh diversity publication required')
    reservation=read(DOCS/'scene-reservation-02.json')
    bank=read(BASE/'pilot-capture-01/bank.json')
    if bank['status']!='completed' or len(bank['trajectories'])!=29:
        raise ValueError('All29 actual pilot scenes must complete')
    groups={}
    for row in reservation['rows']:groups.setdefault(row['group'],[]).append(row)
    split_seeds={s:set()for s in ('train','development','reserved')}
    for identifier,rows in groups.items():
        if len(rows)!=6 or len({r['split']for r in rows})!=1:
            raise ValueError('Density variants must remain in one complete split')
        if {r['scene']['cells_per_metre']for r in rows}!={8,10,16,25,50,100}:
            raise ValueError('Every forcing/geometry group requires all six densities')
        first=rows[0]['scene']
        if any((r['scene']['seed'],r['scene']['counts'],r['scene']['forcing_program'],r['scene']['forcing_schedule'])
               !=(first['seed'],first['counts'],first['forcing_program'],first['forcing_schedule'])for r in rows):
            raise ValueError('Paired density group changed geometry or controls')
        split_seeds[rows[0]['split']].add(first['seed'])
    if any(split_seeds[a]&split_seeds[b]for a,b in [('train','development'),('train','reserved'),('development','reserved')]):
        raise ValueError('Forcing seeds leak across scene splits')
    coverage=[]
    for density in (8,10,16,25,50,100):
        for profile in ('continuous','continuous_varying','regular_pulse','irregular_pulse'):
            for motion in ('stationary','oscillating','reflected_line'):
                rows=[r for r in reservation['rows']if r['split']=='train'and
                      (r['scene']['cells_per_metre'],r['profile'],r['motion'])==(density,profile,motion)]
                if len(rows)!=2 or not any(max(r['scene']['counts'])<=32 for r in rows)or not any(max(r['scene']['counts'])>32 for r in rows):
                    raise ValueError('Each density/profile/motion needs a small and larger-axis training scene')
                coverage.append(dict(cells_per_metre=density,profile=profile,motion=motion,
                    scenes=len(rows),counts_xyz=[r['scene']['counts']for r in rows]))
    pilot_rows=[]
    for row in bank['trajectories']:
        path=ROOT/row['receipt']
        if digest(path)!=row['receipt_sha256']:raise ValueError('Pilot receipt changed')
        data=read(path)
        if (data['actual_steps']!=8 or len(data['frames'])!=8 or not data['native_process']['reaped']or
                data['native_process']['returncode']!=0 or not all(f['frame']['advanced']for f in data['frames'])):
            raise ValueError('All pilot physical advances and native custody must qualify')
        if [f['frame']['step']for f in data['frames']]!=list(range(1,9)):
            raise ValueError('Actual native integration step sequence changed')
        scene=data['scene'];counts=scene['counts'];density=scene['cells_per_metre']
        values=[1000*f['frame']['frame_seconds']for f in data['frames']]
        pilot_rows.append(dict(identifier=row['identifier'],cells_per_metre=density,
            counts_xyz=counts,voxel_count=math.prod(counts),physical_extent_m=data['physical_extent_m'],
            aspect_ratio=max(counts)/min(counts),profile=data['profile'],motion=data['motion'],
            actual_steps=8,stored_bytes=data['stored_bytes'],capture_elapsed_s=data['elapsed_s'],
            native_reference_frame_ms=dict(median=statistics.median(values),p95=sorted(values)[math.ceil(.95*len(values))-1],maximum=max(values)),
            complete_native_advances=True,model_target_pass=False))
    cache,checks={},[]
    for folder in (DOCS,ROOT/'docs/plans/full-ai-tight-targets-v1'):
        for path in sorted(folder.glob('*.json')):
            d=read(path)
            for field in ('source_sha256','source_evidence','execution_assets_sha256'):
                values=d.get(field)
                if not isinstance(values,dict):continue
                count=0
                for p,h in values.items():
                    if not isinstance(h,str)or len(h)!=64:continue
                    if p not in cache:cache[p]=digest(ROOT/p)
                    if cache[p]!=h:raise ValueError('Registered source changed: '+str(path)+' '+p)
                    count+=1
                checks.append(dict(registration=str(path.relative_to(ROOT)),field=field,count=count))
    summary=dict(status='completed_diverse_native_pilots_expanded_models_not_yet_trained',created_unix=time.time(),
        scene_counts_by_split=reservation['scene_counts_by_split'],
        independent_groups_by_split=reservation['independent_groups_by_split'],
        densities=[8,10,16,25,50,100],training_factorial_coverage=coverage,
        pilot_rows=pilot_rows,total_pilot_physical_steps=232,
        total_pilot_stored_bytes=sum(r['stored_bytes']for r in pilot_rows),
        reserved_reference_scenes_consumed=0,new_optimizer_updates=0,
        group_key=['cells_per_metre','counts_xyz'],mixed_density_benchmark_average_published=False,
        native_timing_scope='Native frame including external full-spectral reference callback; excludes setup and state storage. These are reference-capture costs, not AI timings',
        model_target_pass=False,earlier_original_study_sources_unchanged=True,
        source_registrations_verified=checks,distinct_artifacts_verified=len(cache),
        source_sha256={str(p.relative_to(ROOT)):digest(p)for p in [DOCS/'scene-reservation-02.json',
            DOCS/'plan-02.json',BASE/'pilot-capture-01/bank.json',BASE/'forcing-capture-qualification-02/qualification.json']},
        publisher_sha256=digest(Path(__file__)))
    output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('Published per-density/per-grid pilots and split audit; expanded model targets remain open',flush=True)


if __name__=='__main__':run()
