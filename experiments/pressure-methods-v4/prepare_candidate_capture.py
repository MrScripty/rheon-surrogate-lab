"""Freeze a separate native adapter for completed candidate checkpoints/executions."""
import argparse
from pathlib import Path
from v4_support import ROOT,HERE,read,write,sha

def prepare(output):
    if output.exists():raise ValueError('Fresh candidate adapter capture required')
    source=(HERE/'streamed_capture.py').read_text()
    source=source.replace('from v4_support import ROOT,BASE,DOCS,read,write,sha,load',
        'sys.path.insert(0,'+repr(str(HERE))+')\nfrom v4_support import ROOT,BASE,DOCS,read,write,sha,load')
    before='        state,wrapper,parity=load();inventory=state.data.helpers().state_inventory(state.network)'
    after='''        import inference,scalable
        entry=read(args.entry)
        material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
        state=inference.load(dict(material,models=[entry]),entry['id'])
        wrapper,parity=scalable.qualify(state)
        inventory=state.data.helpers().state_inventory(state.network)
        if args.execution=='large-channels-last':
            from execution import LargeChannelsLast
            wrapper=LargeChannelsLast(wrapper).eval()'''
    if source.count(before)!=1:raise ValueError('Native checkpoint load seam changed')
    source=source.replace(before,after)
    source=source.replace("read(DOCS/'scene-reservation.json')","read(args.reservation)")
    source=source.replace("sha(DOCS/'scene-reservation.json')","sha(args.reservation)")
    source=source.replace("row['population']=='blind'","row['population'].startswith('blind')")
    source=source.replace("if len(frames)in row['snapshots']:","if args.record_requests and len(frames)in row['snapshots']:")
    source=source.replace("if index in row['snapshots']:","if args.record_requests and index in row['snapshots']:")
    source=source.replace("entry=state.entry if args.policy!='spectral'else None,", "entry=state.entry if args.policy!='spectral'else None,execution=args.execution,")
    seam="    parser.add_argument('--timeout',type=int,default=1800);parser.add_argument('--states',action='store_true')"
    replacement=seam+"\n    parser.add_argument('--entry',type=Path,required=True)\n    parser.add_argument('--execution',choices=['eager','large-channels-last'],default='eager')\n    parser.add_argument('--reservation',type=Path,default=DOCS/'scene-reservation.json')\n    parser.add_argument('--record-requests',action='store_true')"
    if source.count(seam)!=1:raise ValueError('Native argument seam changed')
    source=source.replace(seam,replacement);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(source)
    source=source.replace("label_semantics='Full spectral solve of each actual input equation, including model-generated states; no reference-trajectory endpoint substitution.',",
        "request_labels_exported=args.record_requests,label_semantics='Same-equation spectral labels'if args.record_requests else'No request labels computed/exported; diagnostics disabled for timing.',")
    output.write_text(source)
    write(output.with_suffix('.json'),dict(status='captured_adapter',source_sha256=sha(output),base_source_sha256=sha(HERE/'streamed_capture.py'),
        builder_source_sha256=sha(__file__),execution_source_sha256=sha(HERE/'execution.py'),original_sources_changed=False))
    print('Captured independent candidate native adapter',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)
