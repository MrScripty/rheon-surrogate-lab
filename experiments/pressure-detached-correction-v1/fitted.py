"""Freeze one train-calibrated fitting bank and evaluate every checkpoint on it."""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, source_inputs, write


def load(material_path, identifier, cpu):
    os.sched_setaffinity(0, {cpu})
    sys.path.insert(0, str(ROOT / 'experiments/pressure-pass-sweep-v1'))
    import inference
    return inference, inference.load(read(material_path), identifier)


def freeze(config_path, material_path, cpu, output):
    _, state = load(material_path, 'C53', cpu)
    import analytical
    import numpy as np
    inputs = source_inputs(ROOT / state.entry['project'])
    _, parents = state.data.load(tuple(inputs[k].read_bytes() for k in ['dataset','dataset_extra']), read(inputs['splits']))
    if not parents or any(parent['entry']['split'] != 'train' for parent in parents):
        raise ValueError('Common bank requires the current train-only calibration source')
    samples = [state.data.sample(parent, slot) for parent in parents for slot in range(len(parent['frames']))]
    amplitude = analytical.training_correction_rms(samples)
    owner_path = ROOT / 'experiments/pressure-fundamental-bank-fit-v1/fit_bank.py'
    captured = ROOT / 'runs/pressure-detached-correction-v1/c53-u-project-02/original_fit_bank.py'
    if sha(owner_path) != sha(captured):
        raise ValueError('Fitting generator differs from the actual captured trainer')
    spec = importlib.util.spec_from_file_location('common_fitting_bank_owner', owner_path)
    bank = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bank)
    settings = read(config_path)['fitting_bank']
    tensors, cases, residency = bank.build(samples[0], settings, amplitude)
    actual = read(ROOT / 'runs/pressure-detached-correction-v1/c53-u-study-01/receipt.json')['training_receipt']['fine_tuning_statistics']['fundamental_bank']
    if cases != actual['cases'] or amplitude != actual['amplitude_rms_pa']:
        raise ValueError('Frozen bank must match the actual optimizer cases and calibration')
    arrays = {f'case_{index:03d}_{name}': tensor.cpu().numpy()
              for index, sample in enumerate(tensors) for name, tensor in sample.items()}
    archive = output.with_suffix('.npz')
    with archive.open('xb') as stream:
        np.savez(stream, **arrays)
    write(output, dict(schema='rheon-detached-common-fitting-bank-v1', status='passed',
        cases=cases, amplitude_rms_pa=amplitude, resident_tensor_bytes=residency,
        arrays=state.data.inventory(arrays), archive=archive.name, archive_sha256=sha(archive),
        generator_sha256=sha(owner_path), config_sha256=sha(config_path), material_sha256=sha(material_path),
        inputs={k:dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for k,p in inputs.items()},
        calibration_population='Original C53 train-only states, shared by all nine actual fits',
        source_sha256=sha(__file__)))
    print('Frozen shared eight-case training bank', flush=True)


def evaluate(config_path, material_path, identifier, cpu, bank_path, output):
    started = time.perf_counter()
    inference, state = load(material_path, identifier, cpu)
    torch, np = state.torch, state.np
    bank = read(bank_path)
    archive = bank_path.parent / bank['archive']
    if (bank['status'] != 'passed' or bank['config_sha256'] != sha(config_path)
            or bank['material_sha256'] != sha(material_path) or sha(archive) != bank['archive_sha256']):
        raise ValueError('Frozen common bank identity changed')
    rows, defects = [], []
    with np.load(archive, allow_pickle=False) as arrays, torch.no_grad():
        if state.data.inventory({name:arrays[name] for name in arrays.files}) != bank['arrays']:
            raise ValueError('Common bank tensor inventory changed')
        for index, case in enumerate(bank['cases']):
            prefix = f'case_{index:03d}_'
            sample = {name.removeprefix(prefix): torch.from_numpy(arrays[name].copy())
                      for name in arrays.files if name.startswith(prefix)}
            denominator = torch.linalg.vector_norm(sample['native_input_residual'])
            defect = float(torch.linalg.vector_norm(state.model.full_operator(sample['reference'],sample['cell_size_m'],sample['rho'])-sample['rhs'])/denominator)
            if not math.isfinite(defect) or defect > 1e-10:
                raise ValueError('Independent common label disagrees with the checkpoint operator')
            defects.append(defect)
            pressure = sample['base']
            for passes in range(1, 9):
                pressure = inference.step(state, sample, pressure)
                if passes in (1,2,4,8):
                    remaining = sample['native_input_residual'] - state.model.full_operator(
                        pressure-sample['base'], sample['cell_size_m'], sample['rho'])
                    value = float(torch.linalg.vector_norm(remaining)/denominator)
                    if not math.isfinite(value):
                        raise ValueError('Nonfinite fitted residual')
                    rows.append(dict(case=index, passes=passes, relative_equation_residual=value, **case))
    summaries = []
    for passes in (1,2,4,8):
        values = [row['relative_equation_residual'] for row in rows if row['passes']==passes]
        summaries.append(dict(passes=passes, cases=8, equation_passes=sum(v<=1e-6 for v in values),
            worst_equation_residual=max(values), median_equation_residual=statistics.median(values)))
    if torch.cuda.is_initialized():
        raise ValueError('CPU evaluation cannot initialize CUDA')
    write(output, dict(status='passed', model=identifier, cases=bank['cases'],
        amplitude_rms_pa=bank['amplitude_rms_pa'], independent_operator_defects=defects,
        summary=summaries, measurements=rows, elapsed_s=time.perf_counter()-started,
        resident_tensor_bytes=bank['resident_tensor_bytes'],
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cpu=cpu, threads=torch.get_num_threads(), cuda_initialized=False,
        material_sha256=sha(material_path), config_sha256=sha(config_path),
        weights_sha256=state.entry['weights_sha256'], bank_manifest_sha256=sha(bank_path),
        bank_source_sha256=bank['generator_sha256'], source_sha256=sha(__file__)))
    print(identifier, summaries, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze','evaluate'])
    for name in ('config','material','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--cpu', required=True, type=int)
    parser.add_argument('--model')
    parser.add_argument('--bank', type=Path)
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.config.resolve(), args.material.resolve(), args.cpu, args.output.resolve())
    else:
        if args.model is None or args.bank is None:
            parser.error('--model and --bank are required for evaluation')
        evaluate(args.config.resolve(),args.material.resolve(),args.model,args.cpu,args.bank.resolve(),args.output.resolve())
