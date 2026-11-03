"""Version the streaming drift diagnostic for longer, predeclared challenges."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def prepare():
    base=ROOT/'runs/pressure-methods-v2'
    original=ROOT/'experiments/pressure-methods-v1/prepare_drift.py'
    text=original.read_text().replace('3600','7200').replace('384','768')
    seam="    crate = output/'probe'"
    if text.count(seam)!=1:raise ValueError('Scene-test update seam changed')
    text=text.replace(seam,"    scene.write_text(replace(scene.read_text(), 's.steps = 512;', 's.steps = 1024;'))\n"+seam)
    capture=base/'harder-drift-preparer-02.py'
    if capture.exists():raise ValueError('Fresh captured drift preparer required')
    capture.write_text(text)
    spec=importlib.util.spec_from_file_location('v2_long_drift_builder',capture)
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    output=base/'harder-drift-build-02'
    builder.prepare(output)
    # Reuse only compiler cache; the failed first build has no qualified binary.
    (output/'target').symlink_to(base/'harder-drift-build-01/target',target_is_directory=True)
    builder.build(output)
    material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    scenes=[]
    for index,(counts,steps,acceleration,swirl,frequency) in enumerate([
        ([16,16,16],7200,.75,.9,2.1),
        ([23,17,11],7200,1.5,-.8,3.3),
        ([31,25,17],1200,2.,.95,.7)]):
        scene=dict(material['config']['scene'],counts=counts,steps=steps,seed=20261120+index,
            acceleration=acceleration,swirl=swirl,frequency=frequency,fluctuation=.7,
            center=[.35,.6,.45],radius=.2)
        path=base/f'harder-scene-{index:02d}.json';write(path,scene)
        scenes.append(dict(id=index,path=str(path.relative_to(ROOT)),sha256=sha(path)))
    write(ROOT/'docs/plans/pressure-methods-v2/drift-reservation.json',dict(status='reserved_before_measurement',
        scenes=scenes,relative_l2_levels=[1e-6,1e-5,1e-4,1e-3,1e-2],
        absolute_limits=dict(velocity_m_s=1e-5,tracer_concentration=1e-5),
        tolerance_scope='Diagnostic sensitivity levels; no application-specific tolerance supplied',
        native_qualification='Run native-only pilots before model policies; retain native physical failures as challenge limits',
        source_sha256=sha(__file__),original_preparer_sha256=sha(original),captured_preparer_sha256=sha(capture)))
    print('Reserved three harder force/shape trajectories up to 120 simulated seconds',flush=True)

if __name__=='__main__':prepare()
