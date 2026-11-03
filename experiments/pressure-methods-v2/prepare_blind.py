"""Unlock reserved blind native scenes only after development selection is frozen."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def prepare():
    base=ROOT/'runs/pressure-methods-v2';docs=ROOT/'docs/plans/pressure-methods-v2'
    selection_path=docs/'selection.json';selection=read(selection_path)
    if selection['status']!='frozen_before_blind_access' or selection['blind_scenes_consumed']!=0:
        raise ValueError('Frozen development-only selection required before blind capture')
    reservation=read(docs/'reservation.json')
    if selection['reservation_sha256']!=sha(docs/'reservation.json'):
        raise ValueError('Blind reservation changed after selection')
    rows=[row for row in reservation['scenes'] if row['population']=='blind']
    training=read(base/'shape-training-view-01/manifest.json')
    if len(rows)!=8 or {r['trajectory_id'] for r in rows}&{r['trajectory_id'] for r in training['parents']}:
        raise ValueError('Eight whole-scene blind identities disjoint from fitting required')
    spec=importlib.util.spec_from_file_location('v2_blind_native_owner',ROOT/'experiments/pressure-methods-v1/native_training.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    output=base/'blind-native-01';owner.prepare(output)
    write(output/'reservation.json',dict(reservation,scenes=rows,
        scope='Eight pre-reserved blind validation trajectories; no optimizer access; selection already frozen'))
    snapshot=output/'blind-preparer.py';snapshot.write_bytes(Path(__file__).read_bytes())
    binding=read(output/'binding.json');prior=read(base/'native-shapes-01/binding.json')
    if binding['original_source_sha256']!=prior['original_source_sha256'] or binding['rheon_source']!=prior['rheon_source']:
        raise ValueError('Blind producer must retain the same native source and selected dependency')
    binding.update(reservation_sha256=sha(output/'reservation.json'),selection_sha256=sha(selection_path),
        original_blind_reservation_sha256=sha(docs/'reservation.json'))
    binding['source_sha256'][str(snapshot)]=sha(snapshot)
    write(output/'binding.json',binding)
    builder_spec=importlib.util.spec_from_file_location('v2_blind_native_builder',ROOT/'experiments/pressure-methods-v1/native_prepare.py')
    builder=importlib.util.module_from_spec(builder_spec);builder_spec.loader.exec_module(builder)
    builder.build(output)
    commands=[]
    for row in rows:
        commands.append(dict(id=row['trajectory_id'],arguments=[str(output/'capture_development.py'),'--build',str(output),
            '--trajectory',row['trajectory_id'],'--output',str(base/'blind-native-data'/row['trajectory_id'])],timeout=660))
    write(base/'blind-capture-specification.json',dict(commands=commands,selection_sha256=sha(selection_path)))
    write(docs/'blind-unlock.json',dict(status='unlocked_after_selection',selection_sha256=sha(selection_path),
        reservation_sha256=sha(docs/'reservation.json'),source_sha256=sha(__file__),
        training_parent_ids=[r['trajectory_id'] for r in training['parents']],
        blind_parent_ids=[r['trajectory_id'] for r in rows],no_training_overlap=True))
    print('Unlocked and built eight pre-reserved blind scenes after selection',flush=True)

if __name__=='__main__':prepare()
