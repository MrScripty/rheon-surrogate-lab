"""Compare a current request with independently retained little-endian f64 bytes.

Historical dataset decoding belongs to the explicitly invoked Q4 tool. This
verifier has no legacy deployment or scientific-record reader.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dependency-migration-v1'))
from model_adapter import strict, measure, require


def verify(deployment, request, expected, output):
    descriptor = strict(deployment.read_bytes())
    require(descriptor['schema'] == 'rheon-pressure-adapter-v1', 'Current adapter manifest required')
    value = strict(request.read_bytes())
    expected_bytes = expected.read_bytes()
    require(len(expected_bytes) > 0 and len(expected_bytes) % 8 == 0, 'Expected little-endian f64 pressure bytes required')
    result = subprocess.run([descriptor['python']['executable'], '-I', '-S', '-B',
        str(deployment.parent / descriptor['entrypoint']), '--deployment', str(deployment)],
        input=json.dumps(value, allow_nan=False).encode()+b'\n', capture_output=True, timeout=120,
        env={'PATH': '/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': '', 'TULDOK_TORCH_DEVICE': 'cpu',
             'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    replies = [strict(line) for line in result.stdout.splitlines()]
    candidates = [reply for reply in replies if reply['kind'] == 'candidate']
    ready = [reply for reply in replies if reply['kind'] == 'adapter_ready']
    require(result.returncode == 0 and len(candidates) == len(ready) == 1, 'Adapter did not complete one current prediction')
    require(ready[0]['manifest_sha256'] == measure(deployment)['sha256']
            and ready[0]['id'] == descriptor['id'] and ready[0]['version'] == descriptor['version'], 'Startup manifest identity mismatch')
    candidate = candidates[0]
    require(candidate['role'] == 'full_pressure' and candidate['id'] == value['id']
            and candidate['device'] == 'cpu' and candidate['optimizer_updates'] == 0, 'CPU full pressure reply required')
    pressure_bytes = struct.pack('<'+'d'*len(candidate['pressure']), *candidate['pressure'])
    require(pressure_bytes == expected_bytes, 'Saved-input exact pressure parity failed')
    report = {'status': 'passed', 'scope': 'one current saved-input pressure comparison',
        'manifest': measure(deployment), 'request': measure(request), 'expected': measure(expected),
        'pressure_sha256': hashlib.sha256(pressure_bytes).hexdigest(), 'exact_prediction_bytes': True,
        'device': 'cpu', 'optimizer_updates': 0, 'stderr_bytes': len(result.stderr)}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('deployment', 'request', 'expected', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.deployment.absolute(), args.request.absolute(), args.expected.absolute(), args.output.absolute())))
