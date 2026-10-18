"""Assemble an unchanged C69 replay with declared, bounded optimizer evidence."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT/'experiments/pressure-four-pass-distillation-v1'
sys.path.insert(0, str(PREVIOUS))
from common import read, sha, write
from prepare import prepare, replace_once


def owner():
    spec = importlib.util.spec_from_file_location('bank_prepare', ROOT/'experiments/pressure-fundamental-bank-fit-v1/prepare.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def instrument_bank(text):
    text = replace_once(text, '    gradient = clip_l2_(network.parameters(), 1.)',
        "    import adam_trace\n    adam_trace.gradients(network, 'gradient_raw')\n    gradient = clip_l2_(network.parameters(), 1.)\n    adam_trace.gradients(network, 'gradient_clipped')")
    return text


def transform(text):
    text = owner().transform(text)
    text = replace_once(text, '    return result\n\n\ndef trained_inputs',
        "    result.update({'adam-trace-%03d.npz'%u: ('training_diagnostics', 8*MIB) for u in (1,10,100,150,200)})\n    return result\n\n\ndef trained_inputs")
    text = replace_once(text, '        optimizer_work={',
        "        import adam_trace\n        adam_trace.start(output/'adam-trace.npz', network)\n        optimizer_work={")
    text = replace_once(text, "                optimizer_work['enabled']=True",
        "                adam_trace.before_update(network, optimizer, normalization, update)\n                optimizer_work['enabled']=True")
    text = replace_once(text, "                optimizer_work['enabled']=False",
        "                optimizer_work['enabled']=False\n                adam_trace.after_update(network, optimizer, normalization, update)")
    text = replace_once(text, "        optimizer_work.pop('enabled')",
        "        adam_observation = adam_trace.finish()\n        optimizer_work.pop('enabled')\n        optimizer_work['observer_elapsed_s'] = adam_observation['observer_elapsed_s']\n        optimizer_work['diagnostic_network_forwards'] = adam_observation['diagnostic_network_forwards']")
    return replace_once(text, "for name in ('weights.npz', 'predictions.npz'",
        "for name in ('adam-trace-001.npz', 'adam-trace-010.npz', 'adam-trace-100.npz', 'adam-trace-150.npz', 'adam-trace-200.npz', 'weights.npz', 'predictions.npz'")


def build(config_path, output):
    config = read(config_path)
    control = ROOT/config['control_project']
    arm = config['arms'][0]
    if len(config['arms']) != 1 or arm['id'] != 'adam-replay':
        raise ValueError('One diagnostic replay required')
    settings = owner().selected(config, arm)
    reference = read(ROOT/config['control_settings'])
    settings.update(updates=reference['updates'], examples=reference['examples'], schedule=reference['schedule'])
    frozen = output.parent/(arm['id']+'-experiment.json')
    bank = output.parent/'observed-fit-bank.py'
    budget_policy = output.parent/'observed-budget-policy.py'
    if output.exists() or frozen.exists() or bank.exists():
        raise ValueError('Fresh replay Project and inputs required')
    output.parent.mkdir(parents=True, exist_ok=True)
    write(frozen, settings)
    bank.write_text(instrument_bank((control/'fit_bank.py').read_text()))
    budget_policy.write_text(replace_once((control/'portable_policy.py').read_text(),
        "resources['max_output_bytes']<=32*MIB", "resources['max_output_bytes']<=64*MIB"))
    resources = {**read(control/'config.json')['resources'], 'max_output_bytes':64*1024**2}
    prepare(frozen, arm['id'], output, source_overrides={
        'fine_tuning.py':control/'fine_tuning.py', 'original_fine_tuning.py':control/'original_fine_tuning.py',
        'original_fit_bank.py':control/'original_fit_bank.py', 'fit_bank.py':bank, 'adam_trace.py':HERE/'adam_trace.py', 'portable_policy.py':budget_policy},
        config_overrides={'resources':resources}, objective_overrides={'fitting_bank':settings['fitting_bank']},
        trainer_transform=transform)
    for mode in ('train','restore'):
        path = output/(mode+'-recipe.json')
        recipe = read(path)
        recipe['resources']['max_output_bytes'] = resources['max_output_bytes']
        if mode == 'train':
            recipe['model_evidence']['profile']['limits']['total_output_bytes'] = resources['max_output_bytes']
        for update in config['capture_updates']:
            name = 'adam-trace-%03d.npz'%update
            recipe['outputs'].append(dict(path=name, role='training_diagnostics', max_bytes=8*1024**2, required=True))
            if mode == 'restore':
                slot = 'trained_adam_trace_%03d_npz'%update
                recipe['inputs'][slot] = dict(max_bytes=8*1024**2)
                recipe['arguments'] += [dict(literal='--'+slot.replace('_','-')), dict(input=slot)]
        write(path, recipe)
    provider = read(output/'provider.json')
    provider['resources']['output_bytes'] = resources['max_output_bytes']
    write(output/'provider.json', provider)
    project = read(output/'project.json')
    for row in project['extensions']:
        row['sha256'] = sha(ROOT/row['path'])
    project['name'] = 'C69 Adam diagnostic replay · unchanged training policy'
    for row in project['recipes']:
        row['sha256'] = sha(ROOT/row['path'])
    write(output/'project.json', project)
    assembly = read(output/'assembly.json')
    assembly['project']['sha256'] = sha(output/'project.json')
    write(output/'assembly.json', assembly)
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/Tuldok'))
    import projects, recipe_contract, extension_contract
    from dependency_config import canonical
    projects.manifest(project)
    extension_contract.declaration(canonical(provider),'extension')
    for mode in ('train','restore'):
        recipe_contract.decode_definition(canonical(read(output/(mode+'-recipe.json'))))
    print('Sealed replay', assembly['project']['id'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    build(args.config.resolve(), args.output.resolve())
