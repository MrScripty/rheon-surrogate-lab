"""Adapt the qualified Project builder at explicit seams; never edit prior captures."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

ARMS=['v2-small16','v2-small24','v2-mixed16','v2-retain24','v2-headbase','v2-headshape']

def replace(text,before,after):
    if text.count(before)!=1:raise ValueError('Builder seam changed: '+before[:100])
    return text.replace(before,after)

def prepare(arm):
    if arm not in ARMS:raise ValueError('Declared arm required')
    original=ROOT/'experiments/pressure-methods-v1/prepare_domains.py'
    text=original.read_text()
    text=replace(text,'HERE = Path(__file__).resolve().parent',"HERE = ROOT/'experiments/pressure-methods-v1'")
    start=text.index('    views = {');end=text.index('    if arm not in views:',start)
    text=text[:start]+'''    views = {'v2-small16':(2,None),'v2-small24':(2,None),
             'v2-mixed16':(2,'shape'),'v2-retain24':(2,'shape'),
             'v2-headbase':(5,'shape'),'v2-headshape':(5,'shape')}
'''+text[end:]
    text=replace(text,"learned = arm in ['learned-coarse','coarse-head-rate']","learned = arm in ['v2-headbase','v2-headshape']")
    text=replace(text,"head_rate = arm == 'coarse-head-rate'","head_rate = learned")
    updates=2400 if arm in ['v2-small24','v2-retain24','v2-headbase','v2-headshape'] else 1600
    text=replace(text,'updates=1600, learning_rate=1e-5,',f'updates={updates}, learning_rate=1e-5,')
    text=replace(text,'schedule=[dict(update=1201, learning_rate=1e-6), dict(update=1501, learning_rate=1e-7)],',
        f'schedule=[dict(update={int(updates*.75)+1}, learning_rate=1e-6), dict(update={int(updates*.9375)+1}, learning_rate=1e-7)],')
    text=replace(text,'training_seed=2026101803, sampling_seed=2026101804, wall_seconds=1800,',
        'training_seed=2026101820, sampling_seed=2026101804, wall_seconds=1800,')
    text=replace(text,"'mixed_order.py':HERE/'mixed_order.py',", "'mixed_order.py':ROOT/'experiments/pressure-methods-v2/shape_order.py',")
    text=replace(text,"directory = ROOT/'runs/pressure-methods-v1'/(view_name+'-training-view-01')",
        "directory = ROOT/'runs/pressure-methods-v2'/(view_name+'-training-view-01')")
    # Every new learned parameter receives the declared coarse-head rate.
    text=replace(text,"'head_optimizer.py':HERE/'head_optimizer.py'", "'head_optimizer.py':ROOT/'experiments/pressure-methods-v2/head_optimizer.py'")
    if arm=='v2-headshape':
        text=replace(text,"initialization = ROOT/'runs/pressure-methods-v1'/(kind+'-initialization-01')",
            "initialization = ROOT/'runs/pressure-methods-v2/shape-head-initialization-01'")
        text=replace(text,"'model.py':HERE/'learned_rate_model.py'", "'model.py':ROOT/'experiments/pressure-methods-v2/shape_model.py'")
        text=replace(text,"'physical_residual.py':HERE/'learned_physical.py'", "'physical_residual.py':ROOT/'experiments/pressure-methods-v2/shape_physical.py'")
    text=replace(text,"config_path = output.parent/(arm+'-experiment.json')", "config_path = output.parent/(arm+'-experiment-02.json')")
    text=replace(text,"captures = output.parent/(arm+'-source-overrides')", "captures = output.parent/(arm+'-source-overrides-02')")
    capture=ROOT/'runs/pressure-methods-v2'/(arm+'-builder-02.py')
    if capture.exists():raise ValueError('Builder already captured')
    capture.write_text(text)
    module_spec=importlib.util.spec_from_file_location('v2_captured_builder',capture)
    builder=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(builder)
    output=ROOT/'runs/pressure-methods-v2'/(arm+'-project-02')
    builder.build(arm,output)
    write(output/'v2-build.json',dict(status='prepared_not_admitted',arm=arm,
        source_sha256=sha(__file__),original_builder_sha256=sha(original),captured_builder_sha256=sha(capture),
        scope='Separate fixed-update and original-small-exposure controls; do not conflate optimizer and cell budgets'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',required=True,choices=ARMS)
    prepare(parser.parse_args().arm)
