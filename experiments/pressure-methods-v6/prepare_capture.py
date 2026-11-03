"""Capture native transport separately; retain the qualified Rust binary and gates."""
import ast
from v6_support import ROOT,HERE,BASE,read,write,sha
def run():
    out=BASE/'native-capture-02.py'
    if out.exists():raise ValueError('Fresh native adapter required')
    prior=ROOT/'runs/pressure-methods-v5/native-capture-03.py';s=prior.read_text()
    s=s.replace(str(ROOT/'experiments/pressure-methods-v5'),str(HERE)).replace('from v5_support','from v6_support').replace('from solver_v3 import','from solver import')
    s=s.replace("retention=args.retention)","retention=args.retention,variant=args.variant,local_execution=args.local_execution,adaptive_reuse=args.adaptive_reuse)")
    s=s.replace("['solver_v3.py','recycling_v3.py','fast_multilevel.py']", "['solver.py','recycling.py','native_local.py','local_native.cpp','local_native_stream.cpp']")
    s=s.replace("choices=['tail','contribution']", "choices=['tail','contribution','coefficient']")
    needle="    parser.add_argument('--record-requests',action='store_true')"
    if s.count(needle)!=1:raise ValueError('Native CLI seam changed')
    s=s.replace(needle,needle+"\n    parser.add_argument('--variant',choices=['all','first','first2','stagnation','coarse1'],default='all')\n    parser.add_argument('--local-execution',choices=['torch','native','streaming'],default='torch')\n    parser.add_argument('--adaptive-reuse',action='store_true')")
    s=s.replace('policy=args.method,max_passes=', 'variant=args.variant,local_execution=args.local_execution,adaptive_reuse=args.adaptive_reuse,policy=args.method,max_passes=')
    s=s.replace('import resource','import resource\nimport shutil')
    s=s.replace("    scene=row['scene'];cells=math.prod(scene['counts'])", "    scene=row['scene'];cells=math.prod(scene['counts'])\n    expected_state_bytes=cells*24*scene['steps']if args.states else 0\n    expected_request_bytes=cells*32*len(row['snapshots'])if args.record_requests else 0\n    if shutil.disk_usage(BASE).free<10*1024**3+expected_state_bytes+expected_request_bytes:raise OSError('Insufficient space for requested full diagnostic capture plus ten GiB reserve')")
    s=s.replace('    scene=row[', "    closure={str((HERE/name).relative_to(ROOT)):sha(HERE/name)for name in ['solver.py','recycling.py','native_local.py','local_native.cpp','local_native_stream.cpp']}\n    adapter_sha=sha(__file__)\n    scene=row[",1)
    s=s.replace("    write(args.output/'receipt.json'", "    if adapter_sha!=sha(__file__)or any(sha(ROOT/name)!=h for name,h in closure.items()):raise ValueError('Native adapter/solver changed during execution')\n    write(args.output/'receipt.json'")
    ast.parse(s);out.write_text(s)
    write(out.with_suffix('.json'),dict(status='captured',source_sha256=sha(out),predecessor_sha256=sha(prior),builder_sha256=sha(__file__),native_build_unchanged=True))
    print('Captured v6 native solver transport; host gates unchanged',flush=True)
if __name__=='__main__':run()
