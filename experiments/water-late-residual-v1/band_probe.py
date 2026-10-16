"""Measure spatial cross-band contamination using a frozen CPU pressure model.

The first three cosine modes per axis match the existing low-mode diagnostic.
This is a spatial index cutoff, not a temporal frequency or a common physical
wavenumber across unequal domains. It does not implement a training curriculum.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB/'experiments/water-pressure-accuracy-v1'))
import diagnose
from learned_iteration import correct


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calls', type=int, nargs='+', default=[4, 8])
    args = parser.parse_args()
    assert args.calls and all(1 <= n <= 32 for n in args.calls)
    assert not args.output.exists(), 'preserve previous attempts'
    started = time.perf_counter()
    study = diagnose.read(args.study)
    source = study['models'][args.model]
    audit = diagnose.audit
    audit.initialize(LAB/source['project'], LAB/source['artifacts'], sorted(os.sched_getaffinity(0)))
    current = int(next(x for x in Path('/proc/self/status').read_text().splitlines()
                       if x.startswith('VmSize:')).split()[1])*1024
    resource.setrlimit(resource.RLIMIT_AS, (current+512*1024**2, current+512*1024**2))
    import numpy as np
    torch = audit.TORCH
    records = []
    with torch.no_grad():
        for witness in study['witnesses']:
            path = LAB/study['root']/(witness['name']+'-'+args.model+'-original')/'capture.json'
            capture = diagnose.read(path)
            assert capture['model'] == source
            for item in (capture['samples'][0], capture['samples'][-1]):
                path = LAB/item['path']
                assert digest(path) == item['sha256']
                with np.load(path, allow_pickle=False) as arrays:
                    sample = {k: torch.from_numpy(arrays[k].copy()) for k in arrays.files}
                sample['base'] = torch.zeros_like(sample['base'])
                residual = sample['native_input_residual']
                residual = residual-residual.mean()
                exact = audit.spectral_correction(residual, sample)
                bases, _ = audit.BASES[residual.shape[2:]]
                coefficients = audit.cosine_transform(residual[0, 0], bases)
                mask = torch.zeros_like(coefficients, dtype=torch.bool)
                mask[:3, :3, :3] = True
                mask[0, 0, 0] = False
                low = audit.cosine_transform(torch.where(mask, coefficients, 0.), bases, inverse=True)[None, None]
                rest = residual-low
                partition_error = audit.vector_norm(residual-low-rest)/audit.vector_norm(residual)
                assert partition_error < 1e-14

                def low_coefficients(pressure):
                    values = audit.cosine_transform(pressure[0, 0], bases)
                    return values[mask]

                exact_low = low_coefficients(exact)
                results = []
                for calls in args.calls:
                    predictions = {name: correct(audit.MODEL, audit.NETWORK, audit.NORM,
                        {**sample, 'native_input_residual': field}, calls, single_evaluation=True)
                        for name, field in [('low', low), ('rest', rest), ('combined', residual)]}
                    intrinsic = low_coefficients(predictions['low'])-exact_low
                    leakage = low_coefficients(predictions['rest'])
                    combined = low_coefficients(predictions['combined'])-exact_low
                    remainder = combined-intrinsic-leakage
                    low_norm = max(audit.vector_norm(exact_low), 1e-30)
                    superposition_error = audit.vector_norm(remainder)/low_norm
                    assert superposition_error < 1e-5, 'frozen linear-map decomposition must qualify'
                    results.append({'calls': calls,
                        'combined_low_pressure_relative_l2': audit.vector_norm(combined)/low_norm,
                        'low_input_only_pressure_relative_l2': audit.vector_norm(intrinsic)/low_norm,
                        'other_band_to_low_leakage_relative_l2': audit.vector_norm(leakage)/low_norm,
                        'leakage_over_intrinsic_error': audit.vector_norm(leakage)/max(audit.vector_norm(intrinsic), 1e-30),
                        'intrinsic_leakage_cosine': float((intrinsic*leakage).sum())/max(
                            audit.vector_norm(intrinsic)*audit.vector_norm(leakage), 1e-60),
                        'superposition_discrepancy_over_exact_low': superposition_error,
                        'removing_other_band_improves_low_error': audit.vector_norm(intrinsic) < audit.vector_norm(combined)})
                records.append({'sample': item, 'counts_xyz': list(residual.shape[2:][::-1]),
                    'partition_relative_defect': partition_error, 'results': results})
    result = {'schema': 'rheon-pressure-spatial-band-probe-v1',
        'scope': 'six consumed saved residuals; frozen checkpoint, zero-base correction map; no trajectory/label-noise or curriculum claim',
        'argv': [sys.executable, *sys.argv], 'study_sha256': digest(args.study), 'source': source,
        'weights_sha256': digest(LAB/source['artifacts']/'weights.npz'),
        'normalization_sha256': digest(LAB/source['artifacts']/'normalization.json'),
        'source_witness_sha256': digest(LAB/source['project']/'source-witness.json'),
        'source_sha256': {str(Path(__file__).relative_to(LAB)): digest(__file__),
            str(Path(audit.__file__).relative_to(LAB)): digest(audit.__file__)},
        'low_band': 'cosine mode indices 0..2 on each axis, excluding constant gauge mode',
        'identity': 'low output error approximately equals intrinsic low-input solve error plus leakage from other input modes',
        'cases': records, 'optimizer_updates': 0, 'simulations_advanced': 0,
        'cuda_initialized': torch.cuda.is_initialized(), 'cpu_threads': torch.get_num_threads(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'elapsed_seconds': time.perf_counter()-started}
    assert not result['cuda_initialized']
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'cases': len(records), 'elapsed_seconds': result['elapsed_seconds'],
        'removing_other_band_improves_low_error': sum(r['removing_other_band_improves_low_error']
            for c in records for r in c['results']), 'comparisons': sum(len(c['results']) for c in records),
        'leakage_over_intrinsic_error': [r['leakage_over_intrinsic_error'] for c in records for r in c['results']]}))


if __name__ == '__main__':
    main()
