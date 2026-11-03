"""Use the unchanged native capture contract with a new frozen reservation."""
import argparse
from pathlib import Path
import capture_pressure_fresh as capture
from study_support import HERE,DOCS,read,write,sha


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in('selection','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    for name,digest in read(args.selection)['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Frozen source changed before capture: '+name)
    capture.DOCS=DOCS/'pressure-confirmation-v2'
    capture.run(args)
    write(args.output/'execution-adapter.json',dict(status='completed',source_sha256=sha(__file__),
        original_capture_sha256=sha(HERE/'capture_pressure_fresh.py'),
        change='Independent learned-tail reservation and frozen precise execution selection; native capture algorithm unchanged',
        optimizer_updates=0,reserved_temporal_scenes_consumed=False))
