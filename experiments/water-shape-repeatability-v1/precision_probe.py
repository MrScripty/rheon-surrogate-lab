"""Measure rounded ideal corrections on learned inputs without applying them."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from execution import LAB, read, write

sys.path.insert(0, str(LAB/'experiments/water-residual-learning-v1'))
import rollout
import audit
from learned_iteration import equation_quality, stopping_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--scene', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    scene = config['scenes'][args.scene]
    model = config['models'][0]
    policy = stopping_policy(read(LAB/config['profile']))
    original = rollout.correct
    records = []
    def observe(network_model, network, normalization, sample, budget,
                stopping=None, trace=None, single_evaluation=False):
        predicted = original(network_model, network, normalization, sample, budget,
                             stopping, trace, single_evaluation)
        torch = audit.TORCH
        base = sample['base']
        rhs = sample['native_input_residual']+network_model.full_operator(
            base, sample['cell_size_m'], sample['rho'])
        scale = normalization['scales'][normalization.get('output_scale','pressure')]['value']
        ideal = base+audit.spectral_correction(sample['native_input_residual'],sample)
        ideal_quality = equation_quality(network_model,ideal,sample,rhs,policy)
        pressure = base
        rounded = []
        for _ in range(2):
            residual = rhs-network_model.full_operator(pressure,sample['cell_size_m'],sample['rho'])
            delta = audit.spectral_correction(residual,sample)
            raw = (delta/scale).to(torch.float32)
            pressure = pressure+network_model.owner.frozen.gauge_correction(raw,scale)
            rounded.append(equation_quality(network_model,pressure,sample,rhs,policy))
        records.append({'step': scene['warmup_steps']+len(records)+1,
                        'ideal_float64': ideal_quality, 'rounded_float32_corrections': rounded,
                        'learned': trace[-1]})
        # Only the original learned pressure reaches the native engine.
        return predicted
    rollout.correct = observe
    try:
        receipt = rollout.run(LAB/config['binary'],scene,args.output.absolute(),
                              LAB/model['project'],LAB/model['artifacts'],
                              proposals=32,stopping=policy,single_evaluation=True)
    finally:
        rollout.correct = original
    result = {'schema': 'rheon-learned-input-precision-probe-v1', 'scene': scene,
        'weights_sha256': hashlib.sha256((LAB/model['artifacts']/'weights.npz').read_bytes()).hexdigest(),
        'records': records, 'cases': len(records),
        'ideal_float64_passed': sum(r['ideal_float64']['equation_passed'] for r in records),
        'one_rounded_correction_passed': sum(r['rounded_float32_corrections'][0]['equation_passed'] for r in records),
        'two_rounded_corrections_passed': sum(r['rounded_float32_corrections'][1]['equation_passed'] for r in records),
        'native_process': receipt['native_process'],
        'scope': 'spectral oracle diagnosed offline on each actual learned input; original learned output alone advances native simulation; no oracle refinement or timing claim'}
    write(args.output/'precision.json',result)
    print({k:result[k] for k in ('cases','ideal_float64_passed','one_rounded_correction_passed','two_rounded_corrections_passed')})


if __name__ == '__main__':
    main()
