"""Assemble matched, immutable managed Projects for the domain-data comparison."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT/'experiments/pressure-four-pass-distillation-v1'
sys.path.insert(0, str(PREVIOUS))
from common import read, write, sha, environment
from prepare import prepare, replace_once


def transform(text):
    start = text.index('def order(parents, config):')
    end = text.index('def finite_tensors(', start)
    text = text[:start]+'def order(parents, config):\n    import mixed_order\n    return mixed_order.order(parents, config)\n\n\n'+text[end:]
    start = text.index('        analytical_samples=None;analytical_evidence=None')
    end = text.index('        profile_report = ', start)
    text = text[:start]+'''        # This paired comparison changes physical training cases only.
        analytical_samples = supplemental_samples = direction_samples = current_samples = None
        analytical_evidence = supplemental_evidence = direction_evidence = None
        feedback = supplemental = direction = None
        current = config.get('current_feedback')
        current_evidence = {'status':'inherited_auxiliary_banks_disabled',
                            'sampling':__import__('mixed_order').STATS}
'''+text[end:]
    text = replace_once(text, "'predictions.npz': ('predictions', 2*MIB)", "'predictions.npz': ('predictions', 16*MIB)")
    return replace_once(text, "helpers.read_npz(raw_trained['trained_predictions_npz'], prediction_inventory, 2*MIB)",
                        "helpers.read_npz(raw_trained['trained_predictions_npz'], prediction_inventory, 16*MIB)")


def head_rate_transform(text):
    text = transform(text)
    text = replace_once(text,"optimizer = torch.optim.Adam(network.parameters(), lr=config['training']['learning_rate'],",
        "optimizer = torch.optim.Adam(__import__('head_optimizer').groups(network,config['training']['learning_rate']), lr=config['training']['learning_rate'],")
    text = replace_once(text,"for group in optimizer.param_groups: group['lr'] = change['learning_rate']",
        "for group in optimizer.param_groups: group['lr'] = change['learning_rate']*group['lr_multiplier']")
    return text


def build(arm, output):
    views = {'domain-small-control':(1,None), 'domain-expanded':(1,'domain'),
             'teacher-coarse':(3,'teacher-coarse'), 'teacher-half':(3,'teacher-half'),
             'teacher-quarter':(3,'teacher-quarter'), 'spacing-data-control':(7,'spacing'),
             'numerical-coarse':(4,'domain'), 'sparse-anchors':(5,'domain'),
             'learned-coarse':(6,'domain'), 'spacing-zero-feature':(7,'spacing'),
             'spacing-conditioned':(7,'spacing'), 'coarse-head-rate':(6,'domain')}
    if arm not in views:
        raise ValueError('Declared paired domain arm required')
    cohort, view_name = views[arm]
    learned = arm in ['learned-coarse','coarse-head-rate']
    head_rate = arm == 'coarse-head-rate'
    if len('rheon.pressure.fourpass.'+arm+'.'+'0'*12)>64:
        raise ValueError('Arm would exceed the managed extension namespace bound')
    config = dict(schema='rheon-pressure-domain-training-v1',
        baseline_material='runs/pressure-pass-sweep-v1/cpu-01/material.json',
        runtime='runs/dependency-migration-v1/runtime/current-manifest-v2.json',
        source_lab='runs/dependency-migration-v1/lab', train_initialization='C53', baselines=['C53'],
        updates=1600, learning_rate=1e-5,
        schedule=[dict(update=1201, learning_rate=1e-6), dict(update=1501, learning_rate=1e-7)],
        training_seed=2026101803, sampling_seed=2026101804, wall_seconds=1800,
        arms=[dict(id=arm, model_id=arm, steps=2, teacher_weight=0.)],
        early_feedback_weight=.05, teacher_passes=8, teacher_cache_bytes=268435456,
        target_relative_residual=1e-6, validation_passes=[1, 2, 4, 8])
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path = output.parent/(arm+'-experiment.json')
    if output.exists() or config_path.exists():
        raise ValueError('Fresh Project and configuration required')
    write(config_path, config)
    source = ROOT/'runs/pressure-current-policy-v1/current-late-project-02'
    captures = output.parent/(arm+'-source-overrides')
    captures.mkdir()
    for name, before, after in [('policy.py', 'MAX_CELLS=65536', 'MAX_CELLS=262144'),
                               ('portable_policy.py', 'MAX_CELLS = 65536', 'MAX_CELLS = 262144')]:
        text = (source/name).read_text()
        if text.count(before) != 1:
            raise ValueError('Shape-ceiling source seam changed')
        (captures/name).write_text(text.replace(before, after))
    overrides = {
        'model.py':HERE/'domain_model.py', 'data.py':HERE/'domain_data.py',
        'historical_data.py':source/'data.py', 'methods_view.py':HERE/'view.py',
        'mixed_order.py':HERE/'mixed_order.py',
        'fixed-normalization.json':ROOT/'runs/pressure-current-policy-v1/current-late-study-03/restore/normalization.json',
        'policy.py':captures/'policy.py', 'portable_policy.py':captures/'portable_policy.py'}
    initialization = None
    objective_overrides = None
    if learned or arm in ['spacing-zero-feature','spacing-conditioned']:
        kind = 'learned-coarse' if learned else 'spacing-two-channel'
        initialization = ROOT/'runs/pressure-methods-v1'/(kind+'-initialization-01')
        initial = read(initialization/'initialization.json')
        if (initial['status'] != 'qualified_initialization' or not initial['surviving_c53_tensor_bytes_preserved']
                or sha(initialization/'weights.npz') != initial['weights_sha256']):
            raise ValueError('Declared surviving-tensor-preserving initialization required')
        objective_overrides = dict(initial_weights_sha256=initial['weights_sha256'],initial_inventory=initial['inventory'])
        overrides.update({'retained_domain_model.py':HERE/'domain_model.py',
                          'historical_physical.py':source/'physical_residual.py'})
        if learned:
            write(captures/'hybrid-method.json',dict(kind=arm,
                qualification='Fine-equation-gated positive learned low-mode gains; 63 extra parameters; no analytic inverse'))
            overrides.update({'model.py':HERE/'learned_model.py','physical_residual.py':HERE/'learned_physical.py',
                              'coarse.py':HERE/'coarse.py','hybrid-method.json':captures/'hybrid-method.json'})
            if head_rate:
                overrides.update({'model.py':HERE/'learned_rate_model.py','learned_coarse_model.py':HERE/'learned_model.py',
                                  'head_optimizer.py':HERE/'head_optimizer.py'})
        else:
            write(captures/'spacing-method.json',dict(explicit_spacing=arm=='spacing-conditioned',
                same_architecture_and_initialization=True,zero_added_convolution_column=True,
                feature='q/train_q_RMS times log2(h/(1/8 m))' if arm=='spacing-conditioned' else 'Identically zero second channel'))
            overrides.update({'model.py':HERE/'spacing_model.py','physical_residual.py':HERE/'spacing_physical.py',
                              'spacing-method.json':captures/'spacing-method.json'})
    if arm in ['numerical-coarse','sparse-anchors']:
        method = dict(kind=arm, coarse_nodes_per_axis=4, placement='Numerical stage before each neural pass',
            numerical_operator='Same sealed fine equation; no reference labels',
            cost='Count geometry setup, restriction/transform, solve/selected inverse, prolongation, residual and neural inference')
        if arm == 'numerical-coarse':
            method.update(transfer='Clamped trilinear interpolation at coarse aggregate centers; conservative transpose restriction',
                          operator='Galerkin Ac=P^T L P, float64 Cholesky with fixed coarse gauge')
        else:
            method.update(anchors='At most 4 selected fine cells per axis, including domain endpoints',
                computation='Full global DCT and all-mode division, selected inverse only; no local-solve speed claim',
                constraint='Fine neural correction is zero at every exact anchor')
        write(captures/'hybrid-method.json',method)
        overrides.update({'model.py':HERE/'hybrid_model.py', 'retained_domain_model.py':HERE/'domain_model.py',
            'physical_residual.py':HERE/'hybrid_physical.py', 'historical_physical.py':source/'physical_residual.py',
            'coarse.py':HERE/'coarse.py', 'hybrid-method.json':captures/'hybrid-method.json'})
    prepare(config_path, arm, output, source_overrides=overrides,
        trainer_transform=head_rate_transform if head_rate else transform,
        objective_overrides=objective_overrides)
    # Rebind physical data before anything is registered or executed.
    project = read(output/'project.json')
    if initialization is not None:
        for row in project['data']:
            if row['id'] == 'initial_weights':
                path = initialization/'weights.npz'
                row.update(path=str(path.relative_to(ROOT)),sha256=sha(path),label='C53 plus declared new parameters')
    if view_name is not None:
        directory = ROOT/'runs/pressure-methods-v1'/(view_name+'-training-view-01')
        replacements = dict(dataset=directory/'dataset-00.npz', dataset_extra=directory/'dataset-01.npz', splits=directory/'manifest.json')
        for row in project['data']:
            if row['id'] in replacements:
                path = replacements[row['id']]
                row.update(path=str(path.relative_to(ROOT)), sha256=sha(path), label=arm+' '+row['id'])
    write(output/'project.json', project)
    slots = {row['id']:ROOT/row['path'] for row in project['data']}
    runtime = read(ROOT/config['runtime'])
    probe = '''import sys,json
sys.path.insert(0,sys.argv[1])
import model,data,policy
model.configure('cpu');model.select(json.load(open(sys.argv[1]+'/config.json')))
packet,parents=data.load(tuple(open(p,'rb').read() for p in sys.argv[2:4]),json.load(open(sys.argv[4])))
norm=model.fit_normalization(parents,data.fit_normalization)
print(json.dumps(dict(packet_sha256=policy.sha(policy.canonical(packet)),normalization_sha256=policy.sha(policy.canonical(norm)),parents=[p['entry']['trajectory_id'] for p in parents])))
'''
    result = subprocess.run([runtime['python'], '-B', '-c', probe, str(output),
        str(slots['dataset']), str(slots['dataset_extra']), str(slots['splits'])],
        env=environment(runtime), capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError('Rebound data/normalization qualification failed: '+result.stderr[-2500:])
    selected = json.loads(result.stdout)
    witness = read(output/'source-witness.json')
    witness['packet_experiment_sha256'] = selected['packet_sha256']
    witness['normalization_sha256'] = selected['normalization_sha256']
    for slot in ['dataset', 'dataset_extra', 'splits', 'initial_weights']:
        witness['inputs'][slot] = dict(bytes=slots[slot].stat().st_size, sha256=sha(slots[slot]))
    write(output/'source-witness.json', witness)
    assembly = read(output/'assembly.json')
    sources = assembly['sources']
    for row in sources:
        row['sha256'] = sha(ROOT/row['path'])
    # Bind every fitting member to the actual newly selected input bytes.
    fit_members = [dict(input=slot, member_id=identifier, sha256=sha(slots[slot]), split='train')
                   for identifier in selected['parents'] for slot in ['dataset', 'dataset_extra']]
    sys.path.insert(0, str(ROOT/config['source_lab']/'experiments/dependency-migration-v1'))
    from typed_pressure_project import policy as field_policy
    pressure_probe = read(ROOT/'runs/dependency-migration-v1/receipts/typed-pressure-preparation-01/pressure-probe.json')
    for mode in ['train', 'restore']:
        path = output/(mode+'-recipe.json')
        recipe = read(path)
        recipe['sources'] = sources
        for row in recipe['outputs']:
            if row['path'] == 'predictions.npz':
                row['max_bytes'] = 16*1024**2
        if mode == 'restore':
            recipe['inputs']['trained_predictions_npz']['max_bytes'] = 16*1024**2
        if mode == 'train':
            original = recipe['model_evidence']
            bundle = dict(definitions={role:read(output/(role+'-contribution.json')) for role in ['reader','task','model','trainer','evaluator','view']},
                architecture=witness['architecture'], physical_context=original['model_specification']['physical_context'],
                preprocessing_policy=original['preprocessing_policy'], output_policy=original['output_policy'])
            policy = field_policy(recipe, bundle, pressure_probe, fit_members)
            policy['profile']['capture']['interval'] = config['updates']
            policy['field_outputs'][0]['clock'] = config['updates']
            recipe['model_evidence'] = policy
            assembly.update(model_specification=policy['model_specification'], preprocessing_policy=policy['preprocessing_policy'], output_policy=policy['output_policy'])
        write(path, recipe)
    provider = read(output/'provider.json')
    provider['sources'] = [dict(row, bytes=(ROOT/row['path']).stat().st_size) for row in sources]
    write(output/'provider.json', provider)
    project['name'] = 'Pressure methods · '+arm+' · C53'
    for row in project['recipes']:
        row['sha256'] = sha(ROOT/row['path'])
    for row in project['extensions']:
        row['sha256'] = sha(ROOT/row['path'])
    write(output/'project.json', project)
    assembly['project']['sha256'] = sha(output/'project.json')
    assembly['initialization_weights_sha256'] = sha(slots['initial_weights'])
    assembly['sources'] = sources
    write(output/'assembly.json', assembly)
    sys.path.insert(0, str(ROOT/'runs/dependency-migration-v1/Tuldok'))
    import projects, recipe_contract, extension_contract
    from dependency_config import canonical
    projects.manifest(project)
    extension_contract.declaration(canonical(provider), 'extension')
    for mode in ['train','restore']:
        recipe_contract.decode_definition(canonical(read(output/(mode+'-recipe.json'))))
    write(output/'domain-assembly.json', dict(status='prepared_not_admitted', cohort=cohort, arm=arm,
        config_sha256=sha(config_path), project_sha256=sha(output/'project.json'), assembly_sha256=sha(output/'assembly.json'),
        source_sha256=sha(__file__), fitting_parents=selected['parents'],
        normalization='Frozen original C53 train-only scales', supervision='Native physical cases; inherited extra banks disabled',
        prediction_export_budget=16*1024**2, optimizer_updates_executed=0))
    print('Rebound and validated', arm, len(selected['parents']), 'physical training parents', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.arm, args.output.resolve())
