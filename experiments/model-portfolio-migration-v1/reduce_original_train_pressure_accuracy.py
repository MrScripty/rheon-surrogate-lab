"""Independently rescore the23CUDA pressure outputs with NumPy, not Torch."""
from collections import Counter
import hashlib
import math
from pathlib import Path
import time

import numpy as np

from import_common_generator_measurements import ROOT,RUN,DOC,read,save,binding,require

METRICS = ('common_equation_relative_l2','pressure_mean_free_relative_l2','pressure_absolute_rms','pressure_absolute_max')
REPORT = DOC/'reports/original-train-pressure-CUDA-accuracy-01.json'


def equation(pressure,h,rho):
    """Closed-boundary finite-volume negative Laplacian, original F64 order."""
    spacing = np.repeat(np.asarray([h],dtype='<f8')[:,None],3,axis=1)
    density = np.asarray([rho],dtype='<f8')
    result = np.zeros_like(pressure)
    for physical_axis,tensor_axis in enumerate((4,3,2)):
        other = [axis for axis in range(3) if axis != physical_axis]
        area = spacing[:,other[0]]*spacing[:,other[1]]
        weight = (area*(1.0/(density*spacing[:,physical_axis]))).reshape(-1,1,1,1,1)
        difference = np.diff(pressure,axis=tensor_axis)
        lower,upper = [(0,0)]*5,[(0,0)]*5
        lower[tensor_axis],upper[tensor_axis] = (1,0),(0,1)
        result = result+weight*(np.pad(difference,lower)-np.pad(difference,upper))
    return result


def metrics(pressure,reference,rhs,h,rho):
    difference = pressure-reference
    difference = difference-difference.mean()
    centered = reference-reference.mean()
    residual = equation(pressure,h,rho)-rhs
    return dict(common_equation_relative_l2=float(np.linalg.norm(residual))/max(float(np.linalg.norm(rhs)),1e-30),
        pressure_mean_free_relative_l2=float(np.linalg.norm(difference))/max(float(np.linalg.norm(centered)),1e-30),
        pressure_absolute_rms=float(np.sqrt(np.mean(difference**2))),pressure_absolute_max=float(np.max(np.abs(difference))))


def run():
    require(not REPORT.exists(),'Preserve original independent pressure reduction')
    batch_path = RUN/'remaining-original-train-pressure-CUDA-qualification-01/status.json'
    batch = read(batch_path)
    require(batch['status'] == 'all22_original_train_CUDA_accuracy_attempts_terminal' and batch['all22_qualified']
            and not batch['pending'] and not batch['adapter_gaps'] and batch['active_child'] is None
            and not Path(f'/proc/{batch["pid"]}').exists(),'All original22workers must be terminal and qualified')
    pilot = RUN/'original-train-pressure-CUDA-pilot-02/fit-8992e19d389ea016/result.json'
    result_paths = [pilot,*[ROOT/r['result']['path'] for r in batch['qualified']]]
    material_path = RUN/'remaining-pressure-material-01.json'
    material = read(material_path)
    fits = {r['fit_item_id']:r for r in material['models']}
    require(len(result_paths) == len(fits) == 23,'Exact23original fit identities required')
    inputs = ROOT/'runs/water-early-pressure-v1/control/project/inputs'
    manifest_path = inputs/'validation_splits.json'
    manifest = read(manifest_path)
    require(manifest['population'] == 'validation','Frozen consumed validation bank required')
    arrays = {}
    for name,spec in zip(['validation_dataset.npz','validation_dataset_extra.npz'],manifest['shards'],strict=True):
        path = inputs/name
        require(path.stat().st_size == spec['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256'],
                'Original native shard changed')
        with np.load(path,allow_pickle=False) as archive:
            require(set(archive.files) == set(spec['keys']),'Original native shard inventory differs')
            for key in archive.files:
                value = archive[key]
                require(key not in arrays and value.dtype.str == manifest['arrays'][key]['dtype']
                        and hashlib.sha256(value.tobytes()).hexdigest() == manifest['arrays'][key]['sha256'],
                        'Native array byte identity differs')
                arrays[key] = value
    cases = [(parent_index,parent,slot,frame) for parent_index,parent in enumerate(manifest['parents'])
        for slot,frame in enumerate(parent['frames'])]
    require(len(cases) == 128,'Complete common128case population required')
    rows,max_relative_difference = [],0.
    aliases = Counter()
    for path in result_paths:
        result = read(path)
        fid = result['fit_item_id']
        require(fid in fits and result['status'] == 'original_train_pressure_CUDA_accuracy_qualified'
                and result['cases'] == len(result['rows']) == 128 and not Path(f'/proc/{result["pid"]}').exists()
                and result['learned_tensors_exact_and_unchanged'] and result['captured_architecture_unchanged']
                and result['device_observation']['CPU_neural_forwards'] == 0
                and not result['device_observation']['forbidden_forward_attempts'] and not result['unbound_lab_Python_modules']
                and result['maximum_captured_common_equation_difference'] == 0,'Complete original CUDA source/parameter/operator qualification required')
        archive_spec = result['output_archive']
        archive_path = ROOT/archive_spec['path']
        require(binding(archive_path) == {k:archive_spec[k] for k in ('path','sha256')},'Retained prediction archive differs')
        maxima = {name:0. for name in METRICS}
        with np.load(archive_path,allow_pickle=False) as archive:
            require(set(archive.files) == {f'case{i:03d}_pressure' for i in range(128)},'Complete original prediction population required')
            for index,((parent_index,parent,slot,frame),prior) in enumerate(zip(cases,result['rows'],strict=True)):
                shape = (1,1,*parent['counts_xyz'][::-1])
                pressure = archive[f'case{index:03d}_pressure']
                require(pressure.dtype.str == '<f8' and pressure.shape == shape and np.isfinite(pressure).all()
                        and hashlib.sha256(pressure.tobytes()).hexdigest() == prior['output_sha256'],
                        'Original CUDA pressure field identity differs')
                require(prior['parent'] == parent_index and prior['native_frame'] == frame['index']
                        and prior['trajectory_id'] == parent['entry']['trajectory_id'] and prior['counts_xyz'] == parent['counts_xyz'],
                        'Original128case identity/order differs')
                for field in ('base','native_input_residual'):
                    require(prior['input_sha256'][field] == frame['sha256'][field],'Original pressure input identity differs')
                scalars = dict(cell_size_m=parent['cell_size_m'],rho=parent['density_kg_m3'],dt=parent['entry']['dt_s'])
                for field,value in scalars.items():
                    require(hashlib.sha256(np.asarray([value],dtype='<f8').tobytes()).hexdigest() == prior['input_sha256'][field],
                            'Original physical metadata input differs')
                reference = arrays[f'p{parent_index:04d}_reference'][slot][None,None]
                rhs = arrays[f'p{parent_index:04d}_rhs'][slot][None,None]
                require(hashlib.sha256(reference.tobytes()).hexdigest() == prior['reference_pressure_sha256'] == frame['sha256']['reference']
                        and hashlib.sha256(rhs.tobytes()).hexdigest() == prior['reference_rhs_sha256'] == frame['sha256']['rhs'],
                        'Original reference labels differ')
                values = metrics(pressure,reference,rhs,scalars['cell_size_m'],scalars['rho'])
                for name,value in values.items():
                    expected = prior['metrics'][name]
                    require(math.isfinite(value) and ((value == 0 == expected) or math.isclose(value,expected,rel_tol=2e-12,abs_tol=1e-300)),
                            'Independent NumPy pressure metric differs')
                    if value or expected:
                        max_relative_difference = max(max_relative_difference,abs(value-expected)/max(abs(value),abs(expected),1e-300))
                    maxima[name] = max(maxima[name],expected)
        require(maxima == result['worst'],'Qualified worst pressure metric differs from every-case maximum')
        aliases[archive_spec['kind']] += 1
        rows.append(dict(fit_item_id=fid,name=fits[fid]['name'],cases=128,worst=maxima,result=binding(path),
            prediction_archive=archive_spec,source_modules_bound=len(result['actual_bound_lab_Python_modules']),
            control_comparisons=result['same_weight_control_comparisons'],dtype='CUDAF32network/F64pressure',
            CPU_neural_forwards=0,latency_qualified=False))
    require({r['fit_item_id'] for r in rows} == fits.keys() and len(rows) == 23,'No original fit omitted or duplicated')
    save(REPORT,dict(status='all23_original_training_pressure_fits_GPU_accuracy_independently_reduced',created_unix=time.time(),
        fits=23,cases_per_fit=128,qualified_unique_pressure_requests=2944,rows=rows,NumPy_independent_metrics_checked=11776,
        maximum_relative_metric_difference=max_relative_difference,metric_relative_tolerance=2e-12,metric_absolute_tolerance=1e-300,
        all_input_reference_prediction_hashes_verified=True,all_original_final_tensors_and_architectures_exact=True,
        all_captured_common_operators_agree=True,CPU_neural_forwards=0,prediction_archive_roles=dict(aliases),
        same_weight_C66pilot_and_five_other_control_populations_exact=True,latency_qualified=False,new_parameter_updates=0,
        full_simulation_step=False,independent_holdout=False,own_history_drift=False,new_reference_generation=False,
        scope='23previously unbenchmarked default pressure fits; same128consumed old-solver equations as127named fits. Source-observer failed128case run and one-case diagnostic are retained separately, not counted again. Independent reduction imports NumPy only; no model or Torch.',
        evidence=[binding(batch_path),binding(material_path),binding(manifest_path),
            binding(DOC/'reports/original-train-pressure-CUDA-pilot-qualification-02.json')],producer=binding(Path(__file__)),full_portfolio_complete=False))
    print('All23original pressure fits/2944cases/11776metrics independently reduced with NumPy; timing remains pending.',flush=True)


if __name__ == '__main__':
    run()
