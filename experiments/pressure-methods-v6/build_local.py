"""Capture compiler identity, command, source and binary for the native kernel."""
import argparse,platform,subprocess
from v6_support import HERE,BASE,read,write,sha
def run(streaming=False):
    out=BASE/('local-native-stream-build-01'if streaming else'local-native-build-01')
    if out.exists():raise ValueError('Fresh native build required')
    out.mkdir();target=out/'local_native.so';source=HERE/('local_native_stream.cpp'if streaming else'local_native.cpp')
    cmd=['g++','-O3','-march=native','-ffp-contract=off','-std=c++17','-shared','-fPIC',str(source),'-o',str(target)]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    (out/'compiler.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Native compile failed; see compiler.log')
    write(out/'build.json',dict(status='built',source_sha256=sha(source),binary_sha256=sha(target),command=cmd,
        compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],machine=platform.machine(),
        source=source.relative_to(HERE.parents[1]).as_posix(),source_sha256_builder=sha(__file__),threads=1,fast_math=False))
    print('Built fused local inference kernel',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--streaming',action='store_true');run(p.parse_args().streaming)
