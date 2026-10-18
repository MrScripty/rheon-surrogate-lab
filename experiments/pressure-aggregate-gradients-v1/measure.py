"""Measure frozen captured gradients and counterfactual common directions."""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import resource
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, source_inputs, write
from aggregation import summarize

spec = importlib.util.spec_from_file_location('captured_component_owner',
    ROOT/'experiments/pressure-loss-components-v1/measure.py')
component_owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(component_owner)
TERMS = component_owner.TERMS


def measure(config_path, identifier, cpu, output, vectors):
    started = time.perf_counter()
    config = read(config_path)
    if output.exists() or vectors.exists() or identifier not in config['models']:
        raise ValueError('Fresh outputs and explicit checkpoint required')
    os.sched_setaffinity(0, {cpu})
    sys.path.insert(0, str(ROOT/'experiments/pressure-pass-sweep-v1'))
    import inference
    material_path = ROOT/config['material']
    material = read(material_path)
    state = inference.load(material, config['objective_models'][identifier])
    torch, np = state.torch, state.np
    import analytical
    import fine_tuning
    import fit_bank
    project = ROOT/state.entry['project']
    weights = ROOT/state.entry['restore']/'weights.npz'
    if identifier == 'C53':
        weights = ROOT/next(row['path'] for row in read(project/'project.json')['data'] if row['id'] == 'initial_weights')
        assert sha(weights) == next(row['weights_sha256'] for row in material['models'] if row['id'] == 'C53')
        fine_tuning.initialize(state.network, weights, state.data.helpers())
    control = ROOT/next(row['project'] for row in material['models'] if row['id'] == 'C66')
    objective, original = read(project/'fine-tuning.json'), read(control/'fine-tuning.json')
    objective.pop('arm'); original.pop('arm')
    assert objective == original
    witness, prior_witness = read(project/'source-witness.json'), read(control/'source-witness.json')
    for name, identity in prior_witness['sources'].items():
        if name not in ('config.json', 'fine-tuning.json', 'fit_bank.py'):
            assert witness['sources'][name] == identity, name
    if identifier == 'C69':
        assert witness['sources']['original_fit_bank.py'] == prior_witness['sources']['fit_bank.py']
        assert witness['sources']['fit_bank.py'] == sha(ROOT/'experiments/pressure-balanced-shapes-v1/fit_bank.py')
    else:
        assert witness['sources']['fit_bank.py'] == prior_witness['sources']['fit_bank.py']
    assert witness['normalization_sha256'] == prior_witness['normalization_sha256']
    inputs = source_inputs(project)
    _, parents = state.data.load(tuple(inputs[key].read_bytes() for key in ('dataset', 'dataset_extra')), read(inputs['splits']))
    assert all(parent['entry']['split'] == 'train' for parent in parents)
    actual = [state.data.sample(parent, j) for parent in parents for j in range(len(parent['frames']))]
    amplitude = analytical.training_correction_rms(actual)
    samples, cases, residency = fit_bank.build(actual[0], fine_tuning.SPEC['fitting_bank'], amplitude)
    fitted_path = ROOT/config['fitting_results'][identifier]
    fitted = read(fitted_path)
    assert sha(weights) == fitted['weights_sha256'] and cases == fitted['cases']
    assert amplitude == fitted['amplitude_rms_pa']
    assert len(samples) == 8 and cases[config['target_case']]['counts_xyz'] == [16, 7, 11]
    pressure_rows = fit_bank.measure(state.network, state.normalization, samples)
    for a, b in zip(pressure_rows, fitted['measurements'], strict=True):
        assert a['case'] == b['case'] and a['passes'] == b['passes']
        assert a['relative_equation_residual'] == b['relative_equation_residual']
    parameters = list(state.network.parameters())
    count = sum(p.numel() for p in parameters)
    workspace, disk = 40*count, len(TERMS)*len(samples)*4*count
    assert workspace <= config['vector_workspace_max_bytes'] and disk <= config['vector_disk_max_bytes']
    assert residency <= 8*1024**2
    vectors.mkdir(parents=True)
    before = state.data.helpers().state_inventory(state.network)
    teacher_calls = fine_tuning.STATS['teacher_forwards']
    state.network.train()
    rows = []
    def path(index, term):
        return vectors/(str(index)+'-'+term+'.npy')
    def gradient(loss, retain=False):
        values = torch.autograd.grad(loss, parameters, retain_graph=retain)
        assert all(bool(torch.isfinite(value).all()) for value in values)
        return torch.cat([value.detach().reshape(-1) for value in values])
    for index, (sample, case) in enumerate(zip(samples, cases, strict=True)):
        defect = float(torch.linalg.vector_norm(state.model.full_operator(sample['reference'], sample['cell_size_m'], sample['rho'])-sample['rhs'])/torch.linalg.vector_norm(sample['rhs']))
        assert defect < 1e-10
        loss = fit_bank.loss(state.network, sample, state.normalization)
        full_value = float(loss.detach())
        full = gradient(loss)
        np.save(path(index, 'full'), full.numpy())
        del full, loss
        terms, joint_loss = component_owner.components(state, sample)
        values = {name:float(value.detach()) for name, value in terms.items()}
        assert math.isclose(sum(values.values()), full_value, rel_tol=config['value_relative_tolerance'], abs_tol=1e-14)
        full = torch.from_numpy(np.load(path(index, 'full'), allow_pickle=False))
        joint = gradient(joint_loss, retain=True)
        if not torch.equal(joint, full):
            raise ValueError('Reconstructed joint gradient differs from captured full gradient')
        del joint, full
        without = gradient((terms['canonical']+terms['final']).mean(), retain=True)
        np.save(path(index, 'without_early'), without.numpy())
        del without
        accumulator = torch.zeros(count, dtype=torch.float64)
        for term in ('canonical', 'final', 'early'):
            vector = gradient(terms[term], retain=term != 'early')
            np.save(path(index, term), vector.numpy())
            accumulator.add_(vector)
            del vector
        del terms, joint_loss
        full = torch.from_numpy(np.load(path(index, 'full'), allow_pickle=False)).double()
        error = float(torch.linalg.vector_norm(accumulator-full)/torch.linalg.vector_norm(full))
        assert math.isfinite(error)
        del accumulator, full
        rows.append(dict(case=index, **case, values=values, full_value=full_value,
            operator_defect=defect, joint_gradient_exact=True, separate_gradient_relative_l2=error,
            separate_gradient_precision_check_passed=error <= config['gradient_relative_tolerance'],
            pressure_residual_by_pass={str(row['passes']):row['relative_equation_residual'] for row in pressure_rows if row['case'] == index}))
    across = {term:component_owner.geometry([path(i, term) for i in range(len(samples))]) for term in TERMS}
    cross = {}
    for term in TERMS:
        matrix = []
        for i in range(len(samples)):
            left = torch.from_numpy(np.load(path(i, term), allow_pickle=False)).double()
            row = []
            for j in range(len(samples)):
                right = torch.from_numpy(np.load(path(j, 'full'), allow_pickle=False)).double()
                row.append(float(left@right))
                del right
            matrix.append(row)
            del left
        cross[term] = matrix
    policies = {}
    for cap in [None, *config['per_case_norm_caps']]:
        policy = summarize(across['full']['dot_matrix'], cross,
            {term:across[term]['norms'] for term in TERMS}, cap=cap,
            global_cap=config['global_norm_cap'], epsilon=config['norm_epsilon'])
        # Independent direct-vector reconstruction checks the Gram-based projection.
        mean = torch.zeros(count, dtype=torch.float64)
        for i, case in enumerate(policy['cases']):
            vector = torch.from_numpy(np.load(path(i, 'full'), allow_pickle=False)).double()
            mean.add_(vector, alpha=case['pre_mean_cap_scale']/len(samples))
            del vector
        assert math.isclose(float(torch.linalg.vector_norm(mean)), policy['mean_l2'], rel_tol=1e-10, abs_tol=1e-12)
        for term in TERMS:
            for i in range(len(samples)):
                vector = torch.from_numpy(np.load(path(i, term), allow_pickle=False)).double()
                direct = float(vector@mean)
                error = abs(direct-policy['cases'][i]['terms'][term]['dot_mean'])
                assert error <= 1e-10*max(1., across[term]['norms'][i]*policy['mean_l2'])
                del vector
        del mean
        policy['direct_vector_projection_check'] = True
        policies['raw_mean' if cap is None else 'per_case_cap_'+str(cap)] = policy
    prior_path = ROOT/config['prior_components']/(identifier+'-components.json')
    replay = None
    if prior_path.exists():
        previous = read(prior_path)
        assert previous['weights_sha256'] == sha(weights) and previous['across'] == across
        assert [row['full_value'] for row in rows] == [row['full_value'] for row in previous['cases']]
        replay = dict(source=str(prior_path.relative_to(ROOT)), sha256=sha(prior_path), exact_component_geometry_and_full_losses=True)
    with zipfile.ZipFile(weights) as archive:
        inventory = archive.namelist()
    assert set(inventory) == {name+'.npy' for name in before}
    assert before == state.data.helpers().state_inventory(state.network)
    assert teacher_calls == fine_tuning.STATS['teacher_forwards'] == 0
    assert not torch.cuda.is_initialized() and torch.get_num_threads() == 1
    sources = [config_path, material_path, fitted_path, weights, project/'config.json',
        project/'fine-tuning.json', project/'source-witness.json', HERE/'aggregation.py',
        Path(component_owner.__file__), Path(__file__).resolve()]
    result = dict(status='passed', model=identifier, cases=rows, terms=TERMS, across=across,
        component_to_full_dots=cross, policies=policies, prior_replay=replay,
        weights_sha256=sha(weights), parameter_count=count, parameter_order=[dict(name=n, shape=list(p.shape)) for n,p in state.network.named_parameters()],
        normalization_sha256=witness['normalization_sha256'], amplitude_rms_pa=amplitude,
        optimizer_updates=0, inventory_unchanged=True, teacher_forwards=0, cuda_initialized=False,
        fitted_pressure_replay_exact=True, optimizer_direction=dict(status='unavailable', reason='Archive matches network arrays only; exact Adam moments were not retained.', archive=inventory),
        fitting_tensor_bytes=residency, vector_workspace_bound_bytes=workspace, vector_disk_bytes=disk,
        vectors={str(p.relative_to(ROOT)):sha(p) for p in sorted(vectors.glob('*.npy'))},
        cpu=cpu, threads=torch.get_num_threads(), elapsed_s=time.perf_counter()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources})
    write(output, result)
    print(identifier, 'target raw/capped cosine', [policy['cases'][config['target_case']]['cosine_mean'] for policy in policies.values()], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'output', 'vectors'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--cpu', type=int, required=True)
    args = parser.parse_args()
    measure(args.config.resolve(), args.model, args.cpu, args.output.resolve(), args.vectors.resolve())
