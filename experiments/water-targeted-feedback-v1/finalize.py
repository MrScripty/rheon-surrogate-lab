"""Reuse the completed-study index with this experiment's source/process scope."""
import argparse
from importlib.util import module_from_spec,spec_from_file_location
import os
from pathlib import Path

from run import LAB,digest,read,write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',type=Path,required=True)
    parser.add_argument('--reports',type=Path,required=True)
    args=parser.parse_args();slug=Path(read(args.sequence)['root']).name
    alive=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:argv=[x.decode(errors='surrogateescape') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
        except OSError:continue
        if len(argv)>1 and Path(argv[0]).name.startswith('python') and any('/'+slug+'/' in x for x in argv[1:]):alive.append(int(p.name))
    assert not alive,alive
    spec=spec_from_file_location('retained_study_index',LAB/'experiments/water-coverage-feedback-v1/finalize.py')
    owner=module_from_spec(spec);spec.loader.exec_module(owner);owner.main()
    path=args.reports/'summary.json';summary=read(path)
    summary['owned_controllers_alive']=alive
    summary['source_sha256'].update({str(p.relative_to(LAB)):digest(p) for p in Path(__file__).parent.glob('*.py')})
    write(path,summary)


if __name__=='__main__':main()
