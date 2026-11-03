"""Add measured, bounded high-volume training anchors and untouched test scenes."""
import hashlib
import json
from pathlib import Path
import time
from force_program import make_scene

ROOT=Path(__file__).resolve().parents[2]
DOCS=ROOT/'docs/plans/full-ai-data-diversity-v1'


if __name__=='__main__':
    original=DOCS/'scene-reservation-02.json'
    pilot=DOCS/'pilot-and-coverage-results-01.json'
    result=json.loads(original.read_text())
    measured=json.loads(pilot.read_text())
    if len(measured['pilot_rows'])!=29 or not all(r['complete_native_advances']for r in measured['pilot_rows']):
        raise ValueError('Complete high-volume native pilots required before expansion')
    additions=[]
    specifications=[('train',(104,112,120),'continuous_varying','reflected_line'),
        ('train',(128,128,128),'regular_pulse','stationary'),
        ('reserved',(108,116,124),'irregular_pulse','oscillating'),
        ('reserved',(120,128,104),'continuous','stationary')]
    used={r['scene']['seed']for r in result['rows']}
    for index,(split,shape,profile,motion)in enumerate(specifications):
        seed=2026102901+index
        if seed in used:raise ValueError('Anchor seed overlaps initial groups')
        group=f'diversity-high-volume-{split}-g{index:02d}'
        result['groups'].append(dict(identifier=group,split=split,seed=seed,counts=list(shape),
            profile=profile,motion=motion,shape_role='high_volume_unseen_shape'if split=='reserved'else'high_volume_training_anchor'))
        for density in (50,100):
            scene=make_scene(seed,shape,density,profile,motion)
            packet=len((json.dumps(scene,separators=(',',':'))+'\n').encode())
            if packet>65536:raise ValueError('Bounded canonical scene packet required')
            additions.append(dict(identifier=f'{group}-cpm{density}',split=split,group=group,
                shape_role='high_volume_unseen_shape'if split=='reserved'else'high_volume_training_anchor',
                profile=profile,motion=motion,scene=scene,physical_extent_m=[n*(1/density)for n in shape],
                voxel_count=__import__('math').prod(shape),initial_packet_bytes=packet))
    result['rows'].extend(additions)
    result.update(created_unix=time.time(),supersedes_uncaptured_reservation=str(original.relative_to(ROOT)),
        high_volume_anchors_added_after_native_pilot=True,
        high_volume_limit=128**3,high_volume_density_subset=[50,100],
        all_above_density_training_domains_now_included=True)
    result['scene_counts_by_split']={s:sum(r['split']==s for r in result['rows'])for s in ('train','development','reserved')}
    result['independent_groups_by_split']={s:sum(g['split']==s for g in result['groups'])for s in ('train','development','reserved')}
    result['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in
        [original,pilot,Path(__file__).resolve(),Path(__file__).with_name('force_program.py').resolve()]}
    with (DOCS/'scene-reservation-03.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    plan=json.loads((DOCS/'plan-02.json').read_text())
    plan.update(supersedes=str((DOCS/'plan-02.json').relative_to(ROOT)),
        high_volume_anchor_pilots_passed=True)
    plan['splits'].update(reservation=str((DOCS/'scene-reservation-03.json').relative_to(ROOT)),
        training_scenes=148,development_scenes=24,reserved_scenes=52,independent_groups=40,
        reserved_references_before_frozen_model_selection=False)
    plan['large_volume_pilot'].update(status='all_planned_large_volume_pilots_completed',
        full_training_anchors={'counts':[[104,112,120],[128,128,128]],'densities':[50,100],
            'actual_steps':240,'new_training_scenes':4,'new_reserved_scenes':4})
    plan['collection'].update(maximum_train_capture_bytes=20*1024**3,
        maximum_development_capture_bytes=2*1024**3,
        pilot_actual_bytes=measured['total_pilot_stored_bytes'])
    for p in [DOCS/'scene-reservation-03.json',pilot,Path(__file__).resolve()]:
        plan['source_sha256'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    with (DOCS/'plan-03.json').open('x')as f:json.dump(plan,f,indent=2);f.write('\n')
    print('Reserved',len(result['rows']),'scenes',result['scene_counts_by_split'],flush=True)
