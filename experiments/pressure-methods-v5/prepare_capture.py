"""Capture the qualified native transport with new policy and host-commit seams."""
import ast
from v5_support import ROOT,HERE,BASE,DOCS,read,write,sha
def change(text,before,after):
    if text.count(before)!=1:raise ValueError('Native capture seam changed: '+before[:100])
    return text.replace(before,after)
def prepare():
    output=BASE/'native-capture-01.py'
    if output.exists():raise ValueError('Fresh native driver required')
    prior=ROOT/'runs/pressure-methods-v4/candidate-capture-02.py';source=prior.read_text()
    source=source.replace(str(ROOT/'experiments/pressure-methods-v4'),str(HERE))
    source=change(source,'from v4_support import ROOT,BASE,DOCS,read,write,sha,load','from v5_support import ROOT,BASE,DOCS,read,write,sha')
    source=source.replace("row['population'].startswith('blind')","'blind'in row['population']")
    begin=source.index("    if args.policy!='spectral':")
    end=source.index('    from large_native import available,read_field',begin)
    source=source[:begin]+'''    from solver import load,Policy
    state,wrapper=load(args.entry)
    inventory=state.data.helpers().state_inventory(state.network)
    policy=Policy(state,wrapper,method=args.method,neural=not args.numerical,reuse=args.reuse,
                  optimized=args.optimized,channels_last=args.execution=='large-channels-last')
    import numerics
'''+source[end:]
    begin=source.index("                        if args.policy=='spectral':")
    end=source.index('                        pressure=pressure-pressure',begin)
    source=source[:begin]+'''                        pressure,work=policy.solve(sample,args.trajectory,max_passes=args.passes)
'''+source[end:]
    source=change(source,"                    shape=(1,1,*scene['counts'][::-1]);base=", "                    transport_tick=time.perf_counter()\n                    shape=(1,1,*scene['counts'][::-1]);base=")
    source=change(source,"                    rhs=residual+numerics.fine_action(base,h,rho)","                    input_read_seconds=time.perf_counter()-transport_tick;assembly_tick=time.perf_counter()\n                    rhs=residual+numerics.fine_action(base,h,rho)\n                    assembly_seconds=time.perf_counter()-assembly_tick")
    source=change(source,"                    child.stdin.write(memoryview(np.ascontiguousarray(pressure.numpy(),dtype='<f8')).cast('B'));child.stdin.flush()", "                    send_tick=time.perf_counter()\n                    child.stdin.write(memoryview(np.ascontiguousarray(pressure.numpy(),dtype='<f8')).cast('B'));child.stdin.flush()\n                    send_seconds=time.perf_counter()-send_tick")
    source=change(source,"                    r=residual-numerics.fine_action(pressure-base,h,rho)\n                    callbacks.append(dict(index=len(frames),method_seconds=method_seconds,work=work,\n                        relative_residual=float(torch.linalg.vector_norm(r))/max(float(torch.linalg.vector_norm(rhs)),1e-300)))",'''                    audit_tick=time.perf_counter()
                    if args.audit_callback:
                        r=residual-numerics.fine_action(pressure-base,h,rho)
                        measured=float(torch.linalg.vector_norm(r))/max(float(torch.linalg.vector_norm(rhs)),1e-300)
                        if abs(measured-work['relative_residual'])>1e-10:raise ValueError('Duplicate residual check disagrees')
                    else:measured=work['relative_residual']
                    callbacks.append(dict(index=len(frames),method_seconds=method_seconds,work=work,
                        relative_residual=measured,input_read_seconds=input_read_seconds,assembly_seconds=assembly_seconds,
                        output_send_seconds=send_seconds,duplicate_audit_seconds=time.perf_counter()-audit_tick))''')
    source=change(source,"                    frame=packet['frame'];index=len(frames);frames.append", "                    frame=packet['frame'];policy.commit(frame['advanced']);index=len(frames);frames.append")
    source=source.replace("if args.policy!='spectral'and inventory!=",'if inventory!=')
    source=source.replace("policy=args.policy,max_passes=args.passes,","policy=args.method,max_passes=args.passes,neural=not args.numerical,reuse_capacity=args.reuse,optimized=args.optimized,duplicate_callback_audit=args.audit_callback,")
    source=source.replace("entry=state.entry if args.policy!='spectral'else None,",'entry=state.entry,')
    source=source.replace('args.trajectory,args.policy,','args.trajectory,args.method,')
    source=change(source,"parser.add_argument('--policy',choices=['spectral','learned','vcycle-neural','vcycle-only','gcr-learned','gcr-multilevel','gcr-multilevel-neural'],required=True)","parser.add_argument('--method',choices=['spectral','gcr','vcycle'],required=True)")
    source=source.replace('choices=[8,16]','choices=[8]')
    source=change(source,"    parser.add_argument('--record-requests',action='store_true')",'''    parser.add_argument('--record-requests',action='store_true')
    parser.add_argument('--reuse',type=int,choices=[0,2,4],default=0)
    parser.add_argument('--numerical',action='store_true')
    parser.add_argument('--optimized',action='store_true')
    parser.add_argument('--audit-callback',action='store_true')''')
    source=change(source,'source_sha256=sha(__file__),build_receipt_sha256=',"source_sha256=sha(__file__),dependency_sources={str((HERE/name).relative_to(ROOT)):sha(HERE/name)for name in ['solver.py','recycling.py','fast_multilevel.py']},build_receipt_sha256=")
    source=change(source,'from v5_support import ROOT,BASE,DOCS,read,write,sha','from v5_support import ROOT,HERE,BASE,DOCS,read,write,sha')
    ast.parse(source);output.write_text(source)
    write(output.with_suffix('.json'),dict(status='captured',source_sha256=sha(output),predecessor_sha256=sha(prior),builder_sha256=sha(__file__),native_build_unchanged=True,
        optimization='Optional removal of duplicate callback diagnostic residual. Solver true-residual and independent native acceptance gates remain unchanged.'))
    commands=[]
    for row in read(DOCS/'scene-reservation.json')['rows']:
        if row['population']!='development':continue
        commands.append(dict(id=row['trajectory_id'],runtime='torch',timeout=600,arguments=[str(output.relative_to(ROOT)),'--trajectory',row['trajectory_id'],'--method','spectral','--numerical',
            '--build','runs/pressure-methods-v4/streamed-build-01','--output',str((BASE/'native-data'/(row['trajectory_id']+'-spectral')).relative_to(ROOT)),
            '--entry','runs/pressure-methods-v3/entries/v3-native.json','--record-requests','--cpu','21']))
    write(BASE/'native-development-plan-01.json',dict(commands=commands))
    print('Captured v5 native adapter and five-scene development capture plan',flush=True)
if __name__=='__main__':prepare()
