"""Train the promising numerical hybrid on matched shape-retention data."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def prepare(nodes):
    arm=f'v2-num{nodes}'
    original=ROOT/'runs/pressure-methods-v2/v2-retain24-builder-02.py'
    text=original.read_text()
    start=text.index('    views = {');end=text.index('    if arm not in views:',start)
    text=text[:start]+f"    views = {{'{arm}':(3,'shape')}}\n"+text[end:]
    old="if arm in ['numerical-coarse','sparse-anchors']:"
    if text.count(old)!=1:raise ValueError('Numerical method seam changed')
    text=text.replace(old,f"if arm == '{arm}':")
    old='method = dict(kind=arm, coarse_nodes_per_axis=4'
    if text.count(old)!=1:raise ValueError('Numerical metadata seam changed')
    text=text.replace(old,f"method = dict(kind='numerical-coarse', coarse_nodes_per_axis={nodes}")
    text=text.replace("if arm == 'numerical-coarse':",f"if arm == '{arm}':")
    if nodes==6:
        path=ROOT/'experiments/pressure-methods-v2/coarse_six.py'
        if not path.exists():
            source=(ROOT/'experiments/pressure-methods-v1/coarse.py').read_text()
            if source.count('count = min(4,n)')!=1:raise ValueError('Coarse geometry seam changed')
            path.write_text(source.replace('count = min(4,n)','count = min(6,n)'))
        text=text.replace("'coarse.py':HERE/'coarse.py'", "'coarse.py':ROOT/'experiments/pressure-methods-v2/coarse_six.py'")
    capture=ROOT/'runs/pressure-methods-v2'/(arm+'-builder-02.py')
    if capture.exists():raise ValueError('Fresh numerical Project required')
    capture.write_text(text)
    spec=importlib.util.spec_from_file_location('v2_numerical_builder',capture)
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    output=ROOT/'runs/pressure-methods-v2'/(arm+'-project-02')
    builder.build(arm,output)
    write(output/'v2-build.json',dict(status='prepared_not_admitted',arm=arm,nodes=nodes,
        source_sha256=sha(__file__),retained_builder_sha256=sha(original),captured_builder_sha256=sha(capture),
        matched_data_and_sampling_with='v2-retain24',updates=2400,
        rationale='Six-node inference passed all 16 cost-screen cases; isolate trained four/six-node stages on retained shape data'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes',type=int,choices=[4,6],required=True)
    prepare(parser.parse_args().nodes)
