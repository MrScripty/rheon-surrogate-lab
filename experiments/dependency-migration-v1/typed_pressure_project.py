"""Assemble one prospective pressure project; never execute or admit its recipes.

The pressure declarations/probe own numerical meaning. Current Projects,
extension and recipe decoders own validation; ordinary custody owns restore
inputs. Source files are copied exactly, and every changed identity is recorded.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

from dependency_config import canonical, decode, digest, require


def document_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write(path, value):
    path.write_bytes(canonical(value))
    return {'path': path.name, 'sha256': digest(path), 'bytes': path.stat().st_size}


def policy(recipe, bundle, probe, fit_members):
    """Author the concrete existing validation-field policy, with no new dialect."""
    adapter = document_sha(sorted(recipe['sources'], key=lambda item: item['path']))
    shape = list(reversed(probe['geometry']['shape_xyz']))
    gauge = {'kind': 'cell_zero', 'pressure_units': 'Pa',
             'correction': 'subtract first correction cell; add supplied solved base exactly once',
             'prior': probe['prior']}
    gauge_sha = document_sha(gauge)
    def field(name, units, stage, gauge_identity):
        return dict(field_id=name, dtype='<f8', shape_zyx=shape, centering='cell', stage=stage,
                    units=units, gauge_policy_sha256=gauge_identity)
    inputs = [field('initial_residual', 'm3/s2', 'native_preprojection',
                    document_sha({'kind':'not_a_pressure_gauge','field':'initial_residual'})),
              field('previous_pressure', 'Pa', 'solved_previous_interval', gauge_sha)]
    output = field('full_pressure', 'Pa', 'train', gauge_sha)
    frozen = {'input': 'pressure_probe', 'sha256': probe['sha256'],
              'member_id': probe['member_id'], 'split': 'validation'}
    population = document_sha([frozen])
    specification = {'format':'rheon_pressure_model_v1', 'architecture':bundle['architecture'],
                     'physical_context':bundle['physical_context'], 'configuration':{}}
    metric = {'id':'same_rhs_mean_free_relative_l2', 'semantic_version':1, 'split':'validation',
              'units':'dimensionless', 'direction':'minimize',
              'reduction':'single_fixed_member_mean_free_pressure_relative_l2',
              'population_sha256':population, 'domain':{'minimum':0, 'maximum':1e300},
              'undefined_rule':'not_applicable', 'evaluator_sha256':adapter,
              'preprocessing_sha256':document_sha(bundle['preprocessing_policy'])}
    return {'format':'extension_field_policy_v1',
        'task':{'id':bundle['definitions']['task']['id'], 'revision':1,
                'definition':bundle['definitions']['task']},
        'adapter_sha256':adapter,
        'profile':{'id':'rheon.pressure.fixed-consumed-validation', 'revision':1,
            'probe_selection':{'role':'validation','algorithm':'sha256_rank_v1',
                'namespace':'rheon-pressure-consumed-validation','seed':0,'count':1},
            'capture':{'clock':'optimizer_update','interval':1,'retain_selected':True,'max_snapshots':1},
            'required_capabilities':['checkpoints','fixed_validation_predictions','scalar_history','field_pairs'],
            'optional_capabilities':[],
            'limits':{'manifest_bytes':1024*1024,'shard_bytes':8*1024**2,
                      'total_output_bytes':recipe['resources']['max_output_bytes']}},
        'probe':{'data_input':'pressure_probe','candidates':[{'frozen_reference':frozen,
            'input_sha256':document_sha([{'field_id':item['field_id'],
                'payload_sha256':probe['array_sha256'][item['field_id']]} for item in inputs]),
            'target_sha256':document_sha([{'field_id':'pressure','payload_sha256':probe['array_sha256']['native_pressure']}])}]},
        'fit_members':fit_members,
        'fields':[{'id':'pressure', 'contract':{'format':'field_adapter_contract_v1',
            'owner':'Rheon pressure lab','adapter':{'name':'rheon_pressure_correction','version':1},
            'source_sha256':adapter,'inputs':inputs,'output':output,'time_mode':'standalone'},
            'geometry':probe['geometry'],'gauge_policy':gauge}],
        'field_targets':[{'member_id':probe['member_id'],'field_id':'pressure',
                          'payload_sha256':probe['array_sha256']['native_pressure']}],
        'field_outputs':[{'clock':1,'field_id':'pressure','member_id':probe['member_id'],
                          'prediction':'pressure-prediction.npy','reference':'pressure-reference.npy'}],
        'probe_inputs':[{'member_id':probe['member_id'],'fields':[
            {'field_id':'initial_residual','artifact':'probe-residual.npy'},
            {'field_id':'previous_pressure','artifact':'probe-base.npy'}]}],
        'metric_definitions':[metric], 'model_specification':specification,
        'preprocessing_policy':bundle['preprocessing_policy'],'output_policy':bundle['output_policy']}


def assemble(root, bundle, probe, destination, tuldok_root):
    """Create fresh current material from explicit source/probe identities only."""
    root = Path(root).resolve(strict=True)
    destination = Path(destination)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    require(destination.is_relative_to(root) and not destination.exists(), 'fresh project beneath artifact root required')
    tuldok_root = Path(tuldok_root).resolve(strict=True)
    prefix = destination.relative_to(root).as_posix()
    original = root / bundle['manifest_path']
    require(decode(original.read_bytes())==bundle['project'], 'selected Q3 source project changed')
    probe_path = Path(probe['path']).resolve(strict=True)
    require(probe_path.is_relative_to(root) and digest(probe_path)==probe['sha256']
            and probe['population']=='validation' and probe['optimizer_updates_executed']==0,
            'exact previously consumed validation probe required')
    destination.mkdir(parents=True, mode=0o700)
    sources = {}
    mapping = []
    for entry in bundle['recipes']['train']['sources']:
        source = root / entry['path']
        require(digest(source)==entry['sha256'], 'selected Q3 source changed')
        if source.name == 'runtime-manifest.json':
            mapping.append({'source':entry,'current':{'input':'runtime_manifest','path':entry['path'],
                            'sha256':entry['sha256']},'treatment':'exact_bytes_current_data_reference'})
            continue
        target = destination / source.name
        require(target.name not in sources, 'flat source names must be unique')
        shutil.copyfile(source, target)
        path = prefix+'/'+target.name
        sources[target.name] = {'path':path,'sha256':digest(target)}
        mapping.append({'source':entry,'current':sources[target.name],'treatment':'exact_bytes'})
    wrapper = Path(__file__).with_name('typed_pressure_evidence.py')
    for source, relative in ((wrapper,'typed_pressure_evidence.py'),
            (tuldok_root/'tuldok_extension_sdk/__init__.py','tuldok_extension_sdk/__init__.py'),
            (tuldok_root/'tuldok_extension_sdk/evidence.py','tuldok_extension_sdk/evidence.py')):
        target = destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        sources[relative] = {'path':prefix+'/'+relative,'sha256':digest(target)}
        mapping.append({'source':{'path':str(source),'sha256':digest(source)},
                        'current':sources[relative],'treatment':'exact_bytes'})
    base_sources = sorted(sources.values(),key=lambda item:item['path'])
    definitions = copy.deepcopy(bundle['definitions'])
    for value in definitions.values():
        if 'implementation' in value:
            value['implementation']['source'] = prefix+'/'+Path(value['implementation']['source']).name
    for role in ('trainer','evaluator'):
        definitions[role]['implementation']['source'] = prefix+'/typed_pressure_evidence.py'
    provider_refs = []
    for role,value in definitions.items():
        target = destination/(role+'-contribution.json')
        row = write(target,value)
        row.update(path=prefix+'/'+target.name,id=value['id'],version=value['version'],kind=value['kind'])
        provider_refs.append(row)
    reader = next(row for row in provider_refs if row['kind']=='reader')
    definitions['trainer']['requires']=[{key:reader[key] for key in ('id','version','sha256')}]
    trainer = next(row for row in provider_refs if row['kind']=='trainer')
    fresh = write(destination/'trainer-contribution.json',definitions['trainer'])
    trainer.update(sha256=fresh['sha256'],bytes=fresh['bytes'])
    selections = {row['kind']:{key:row[key] for key in ('id','version','sha256')}
                  for row in provider_refs if row['kind'] in ('task','model','trainer','evaluator','view')}
    recipes = {}
    for mode in ('train','restore'):
        recipe = copy.deepcopy(bundle['recipes'][mode])
        recipe.update(id='rheon-pressure-typed-'+mode,name='Bounded pressure '+mode,
                      entrypoint=prefix+'/typed_pressure_evidence.py',sources=base_sources)
        recipe['inputs']['runtime_manifest']={'max_bytes':32*1024**2}
        for argument in recipe['arguments']:
            if 'source' in argument:
                if Path(argument['source']).name=='runtime-manifest.json':
                    argument.clear();argument['input']='runtime_manifest'
                else:
                    argument['source'] = prefix+'/'+Path(argument['source']).name
        if mode=='train':
            recipe.update(schema_version=3,format='external_model_evidence_recipe_v2',
                          extension_selection={'selections':selections})
            recipe['inputs']['pressure_probe']={'max_bytes':1024*1024}
            recipe['arguments'] += [{'literal':'--pressure-probe'},{'input':'pressure_probe'},
                                    {'literal':'--model-evidence-context'},{'model_evidence_context':True}]
            for entry in recipe['outputs']:
                if entry['path']=='profile.json':
                    entry['path']='training-profile.json'
            common = ['execution-context.json','model-evidence.json','model-definition.json','profile.json',
                'probe.json','metric-definitions.json','checkpoints.json','observations.json','field-index.json',
                'fitted-transform.json','output-policy.json','checkpoint-1.json','pressure-diagnostic.json']
            for name in common:
                recipe['outputs'].append({'path':name,'role':'model_context' if name=='execution-context.json' else 'model_evidence',
                                         'max_bytes':65536 if name=='execution-context.json' else 1024*1024,'required':True})
            for name in ('probe-residual.npy','probe-base.npy','pressure-prediction.npy','pressure-reference.npy'):
                recipe['outputs'].append({'path':name,'role':'model_evidence','max_bytes':8*1024**2,'required':True})
            refs={row['id']:row for row in bundle['project']['data']}
            splits=decode((root/refs['splits']['path']).read_bytes())
            fit=[{'input':slot,'sha256':refs[slot]['sha256'],
                  'member_id':row['entry']['trajectory_id'],'split':'train'}
                 for row in splits['parents'] for slot in ('dataset','dataset_extra')]
            current_bundle=dict(bundle,definitions=definitions)
            recipe['model_evidence']=policy(recipe,current_bundle,probe,fit)
        write(destination/(mode+'-recipe.json'),recipe)
        recipes[mode]=recipe
    provider={'format':'tuldok_extension_v1','package':{'id':'rheon.pressure','version':'1.0.0'},
        'label':'Prospective pressure diagnostic; no quality promotion',
        'license':{'id':'private-lab-authored','notices':[]},'sources':[
            dict(row,bytes=(root/row['path']).stat().st_size) for row in base_sources],
        'host_contracts':[{'id':'tuldok:extension','version':'1.0.0'}, {'id':'tuldok:project','version':'2.0.0'}],
        'runtime':{'profile':'managed_torch_python_v1','snapshot_sha256':bundle['runtime']['manifest_sha256']},
        'effects':['filesystem'],'resources':{'wall_seconds':recipes['train']['resources']['wall_seconds'],
            'source_bytes':8*1024**2,'output_bytes':recipes['train']['resources']['max_output_bytes']},
        'contributions':provider_refs}
    write(destination/'provider.json',provider)
    project=copy.deepcopy(bundle['project'])
    project.update(schema_version=2,id=uuid.uuid4().hex,name='Prospective pressure Model and bounded diagnostic Trial')
    project['recipes']=[{'id':mode,'label':'Pressure '+mode,'path':prefix+'/'+mode+'-recipe.json',
                        'sha256':digest(destination/(mode+'-recipe.json'))} for mode in ('train','restore')]
    runtime_source = next(row for row in bundle['recipes']['train']['sources']
                          if Path(row['path']).name=='runtime-manifest.json')
    project['data'].append({'id':'runtime_manifest','label':'Exact measured managed runtime manifest',
                           'path':runtime_source['path'],'sha256':runtime_source['sha256']})
    project['data'].append({'id':'pressure_probe','label':'Previously consumed fixed validation frame',
                           'path':probe_path.relative_to(root).as_posix(),'sha256':probe['sha256']})
    project['extensions']=[{'id':'pressure','label':provider['label'],'path':prefix+'/provider.json',
                            'sha256':digest(destination/'provider.json')}]
    write(destination/'project.json',project)
    sys.path.insert(0,str(tuldok_root))
    import extension_contract
    import projects
    import recipe_contract
    projects.manifest(project)
    for value in definitions.values():
        extension_contract.declaration(canonical(value),'contribution')
    extension_contract.declaration(canonical(provider),'extension')
    for recipe in recipes.values():
        recipe_contract.decode_definition(canonical(recipe))
    provenance={'format':'rheon_pressure_project_assembly_v1','source_project':{
        'id':bundle['project']['id'],'path':bundle['manifest_path'],'sha256':digest(original)},
        'current_project':{'id':project['id'],'path':prefix+'/project.json','sha256':digest(destination/'project.json')},
        'runtime':bundle['runtime'],'source_mapping':mapping,'probe':probe,
        'builder_sha256':digest(Path(__file__)),'model_specification':recipes['train']['model_evidence']['model_specification'],
        'preprocessing_policy':bundle['preprocessing_policy'],'output_policy':bundle['output_policy'],
        'limits':bundle['limits'],'trial_lineage':'prospective root; historical ancestry remains unknown',
        'execution_submitted':False,'optimizer_updates_executed':0,'scientific_acceptance':'not_asserted'}
    write(destination/'assembly.json',provenance)
    return provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--declarations',type=Path,required=True)
    parser.add_argument('--probe-metadata',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--tuldok-root',type=Path,required=True)
    args=parser.parse_args()
    value=assemble(args.root,decode(args.declarations.read_bytes()),decode(args.probe_metadata.read_bytes()),
                   args.destination,args.tuldok_root)
    print(json.dumps(value['current_project'],sort_keys=True))


if __name__=='__main__':
    main()
