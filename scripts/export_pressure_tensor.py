"""Export recorded variable-pressure model inputs for the HTML tensor explorer.

No network inference, optimization, padding or synthetic field generation occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

LAB = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--normalization', type=Path, required=True)
    parser.add_argument('--parent', type=int, default=1)
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 23])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-profile', choices=('variable-pressure-v2', 'variable-pressure-v1'), default='variable-pressure-v2',
                        help='Current seven-channel model, or the preserved ten-channel model')
    args = parser.parse_args()
    sys.path.insert(0, str(LAB / 'experiments' / args.model_profile))
    import data
    import model
    import torch
    if not 1 <= len(args.frames) <= 4 or any(not 0 <= f < 24 for f in args.frames):
        parser.error('Select one to four frame indices in 0..23.')
    raw = args.packet.read_bytes()
    manifest = json.loads(args.manifest.read_bytes())
    _, parents = data.load(raw, manifest)
    if not 0 <= args.parent < len(parents):
        parser.error('Parent index is outside this packet.')
    normalization_raw = args.normalization.read_bytes()
    normalization = json.loads(normalization_raw)
    if normalization != data.fit_normalization(parents):
        parser.error('Normalization must match the train-only statistics of this packet.')
    parent = parents[args.parent]
    tensors = []
    samples = []
    for frame in args.frames:
        sample = data.sample(parent, frame, labels=False)
        geometry = sample['cell_size_m'] if args.model_profile == 'variable-pressure-v2' else sample['spacing']
        tensors.append(model.build_inputs(sample['native_input_residual'], sample['base'],
                                         geometry, sample['rho'], sample['dt'], normalization))
        samples.append({'label': f"{parent['entry']['trajectory_id']} · frame {frame}",
                        'frame': frame, 'input_sha256': parent['frames'][frame]['sha256']})
    tensor = torch.cat(tensors, dim=0).contiguous()
    export = {'schema': 'rheon-pressure-tensor-view-v1', 'shape': list(tensor.shape),
              'channels': model.CHANNELS,
              'spacing_m': [parent['cell_size_m']]*3 if args.model_profile == 'variable-pressure-v2' else parent['spacing_m'],
              'samples': samples, 'model_profile': args.model_profile,
              'values': tensor.reshape(-1).tolist(),
              'source': f"Recorded model inputs · {parent['entry']['trajectory_id']} · "
                        f"{parent['entry']['split']} parent · frames {', '.join(map(str, args.frames))}. "
                        'Exact float32 preprocessing of qualified saved fields; no model inference or updates.',
              'provenance': {'dataset_sha256': manifest['dataset_sha256'],
                             'normalization_sha256': hashlib.sha256(normalization_raw).hexdigest(),
                             'model_source_sha256': hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),
                             'tensor_f32_sha256': hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(export, stream, allow_nan=False, separators=(',', ':'))
    print(json.dumps({'output': str(args.output), 'shape': export['shape'], 'provenance': export['provenance']}))


if __name__ == '__main__':
    main()
