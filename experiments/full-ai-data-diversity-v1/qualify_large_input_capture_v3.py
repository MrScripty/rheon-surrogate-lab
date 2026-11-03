"""Exercise real full-size managed capture without starting a fit or a model.

Uses a new test catalog, the registered installation's real asset manifest and
the existing training-only archive. Temporary retained/staged copies are removed
after exact verification; the source archive and live catalog are untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from capture_memory_telemetry_v1 import process_memory
import shutil
import sys
import time
import uuid


ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        while chunk:=handle.read(1024**2):digest.update(chunk)
    return digest.hexdigest()


def read(path):return json.loads(path.read_text())


def write(path,value):
    with path.open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False)
        handle.write('\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    protocol=read(args.protocol)
    for name,expected in protocol['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen qualification source changed: '+name)
    regressions=read(ROOT/protocol['regression_report'])
    if sha(ROOT/protocol['regression_report'])!=protocol['regression_report_sha256'] or regressions['status']!='passed':
        raise ValueError('Actual affected regression suites must pass first')
    if args.output.exists():raise ValueError('Fresh qualification output required')
    started=time.monotonic()
    args.output=args.output.resolve()
    args.output.mkdir()
    scratch=args.output/'temporary-catalog-and-captures'
    source=args.output/'source';source.mkdir()
    phases=[]
    dataset=None
    result=dict(status='running',new_parameter_fits=0,model_execution=False,cpu_model_forward=False,
        expanded_reserved52_references_opened=0,live_catalog_accessed=False,latency_claim=False,
        full_generator_target_pass=False)
    def phase(name):
        memory=process_memory()
        phases.append(dict(name=name,elapsed_seconds=time.monotonic()-started,**memory))
        print('Large-input qualification:',name,memory,flush=True)
        if memory['peak_since_exec_bytes']>protocol['maximum_peak_rss_bytes']:raise ValueError('Capture memory qualification failed')
        if time.monotonic()-started>protocol['maximum_wall_seconds']:raise ValueError('Capture wall budget exceeded')
    try:
        branch=ROOT/protocol['tuldok_source']
        sys.path.insert(0,str(branch))
        sys.path.insert(0,str(branch/'tests'))
        from app import Dataset
        from auth_helpers import ExplicitOwnerFixture
        from model_definitions import ModelDefinitions
        from torch_runtimes import TorchRuntimes
        import recipe_contract
        import recipe_process
        import torch_runtime
        view=read(ROOT/protocol['training_view_manifest'])
        archive=(ROOT/protocol['training_view_manifest']).parent/view['archive']['path']
        expected=view['archive']['sha256']
        if archive.stat().st_size!=view['archive']['bytes']:raise ValueError('Training archive size changed')
        # Two additional copies: retained material and exact execution staging.
        required=2*archive.stat().st_size+protocol['disk_reserve_bytes']
        if shutil.disk_usage(args.output).free<required:raise ValueError('Insufficient measured disk headroom')
        dataset=Dataset(scratch,project_roots={'fixture':ROOT})
        owner=ExplicitOwnerFixture.bootstrap(dataset)
        runtime_path=ROOT/protocol['runtime_manifest']
        runtime_raw=runtime_path.read_bytes()
        runtime_manifest=torch_runtime.decode(runtime_raw)
        runtime_sha=hashlib.sha256(runtime_raw).hexdigest()
        with owner.bound(dataset):
            runtime=TorchRuntimes(dataset.db,dataset.access,scratch).create(runtime_manifest,runtime_sha)
            phase('real_runtime_asset_inventory_registered_in_new_test_catalog')
            del runtime_raw,runtime_manifest
            entry=source/'main.py';entry.write_text('raise RuntimeError("Capture qualification never executes a model or recipe")\n')
            prefix=str(source.relative_to(ROOT))+'/'
            project_id=uuid.uuid4().hex
            policy=protocol['capture_policy']
            static=dict(part_00=(archive,expected),manifest=(ROOT/protocol['training_view_manifest'],sha(ROOT/protocol['training_view_manifest'])),
                normalization=(archive.parent/'network-normalization.json',sha(archive.parent/'network-normalization.json')),
                optimization_normalization=(archive.parent/'optimization-normalization.json',sha(archive.parent/'optimization-normalization.json')),
                runtime_manifest=(runtime_path,runtime_sha))
            definition=dict(schema_version=1,format='external_recipe_v1',id='large-capture',name='Full training archive capture qualification',version=1,
                runtime=recipe_contract.TORCH_PROFILE,runtime_device='cuda',runtime_snapshot_sha256=runtime_sha,
                capture_policy=policy,sources=[dict(path=prefix+'main.py',sha256=sha(entry))],entrypoint=prefix+'main.py',
                arguments=[*[dict(input=name) for name in static],dict(input_evidence=True),dict(output_dir=True)],parameters={},
                inputs={name:dict(max_bytes=policy['max_input_file_bytes'] if name=='part_00' else 16*1024**2) for name in static},
                outputs=[dict(path='unused.json',role='result',max_bytes=1024,required=True)],environment={},
                resources=dict(wall_seconds=30,max_output_bytes=1024,cpu_cores=1,memory_bytes=protocol['maximum_peak_rss_bytes']))
            recipe=source/'recipe.json';write(recipe,definition)
            project=dict(schema_version=1,id=project_id,name='Isolated full-size capture qualification',
                data=[dict(id=name,label=name,path=str(path.relative_to(ROOT)),sha256=digest) for name,(path,digest) in static.items()],
                recipes=[dict(id='large-capture',label='Capture only',path=prefix+'recipe.json',sha256=sha(recipe))],evidence=[],requirements=[])
            project_path=source/'project.json';write(project_path,project)
            dataset.projects.open(dict(root='fixture',manifest_path=str(project_path.relative_to(ROOT))))
            selector=dict(project_id=project_id,recipe_id='large-capture',expected_sha256=sha(recipe))
            graph=dataset.pipelines.create_external(dict(name='Full-size capture only',**selector,
                inputs={name:dict(reference_id=name,expected_sha256=digest) for name,(_path,digest) in static.items()}))
            models=ModelDefinitions(dataset)
            draft=models.create(dict(name='Capture only; no fitted model',kind='captured_source',
                task=dict(kind='task',catalog_id=dataset.db.dataset_id,id='neutral.bytes',revision=1,sha256='a'*64),
                specification=dict(format='large_capture_qualification_v1'),source_selector=selector,
                preprocessing_policy=dict(kind='identity'),output_policy=dict(kind='exact_bytes')))
            model=models.publish(draft['id'],dict(revision=1))
            prepared=dataset.pipelines.prepare_external_material(graph['id'],1,model['ref'])
            phase('entire_archive_captured_to_owned_material')
            with dataset.lock,dataset.db.transaction():retained=dataset.pipelines._retain_material(prepared)
            dataset.pipelines.recipe_material(retained['ref'],verify=True)
            phase('retained_full_inventory_stream_verified')
            execution=dataset.pipelines.capture_recipe_material(retained['ref'],scratch/'execution-staging')
            phase('retained_archive_recaptured_to_exact_execution_staging')
            spec=dict(execution['spec'],attempt_id='b'*32,request_sha256='c'*64)
            recipe_process.validate_spec(spec)
            recipe_process.verify_captures(spec)
            phase('complete_execution_spec_sources_inputs_and_real_runtime_verified')
            observed=execution['material']['captures']['input/part_00']
            from managed_input_stream_v1 import validate_inputs
            from types import SimpleNamespace
            inputs=scratch/'execution-staging/input'
            fitting_args=SimpleNamespace(mode='train',input_evidence=scratch/'execution-staging'/recipe_contract.INPUT_EVIDENCE,
                manifest=inputs/'manifest',normalization=inputs/'normalization',optimization_normalization=inputs/'optimization_normalization',
                runtime_manifest=inputs/'runtime_manifest',shards=[inputs/'part_00'])
            _evidence,identities=validate_inputs(fitting_args)
            if identities['part_00']!=observed:raise ValueError('Portable fitting input identity differs')
            phase('portable_fitting_validator_verified_all_five_real_captured_Project_inputs')
            if observed!=dict(bytes=archive.stat().st_size,sha256=expected):raise ValueError('Complete archive capture differs')
            if dataset.runs.list():raise ValueError('Capture qualification unexpectedly created a Run')
            result.update(status='passed_real_fullsize_streamed_project_material_and_execution_capture',
                typed_fitting_static_inputs_verified=identities,archive_bytes=observed['bytes'],archive_sha256=observed['sha256'],capture_policy=policy,
                training_scenes=view['all148training_scenes_in_objective'] and 148,
                source_runtime_manifest_sha256=runtime_sha,runtime_asset_bytes=runtime['total_bytes'],
                runtime_asset_files=runtime['file_count'],recipe_material_sha256=retained['ref']['sha256'],
                execution_material_sha256=execution['material_sha256'],admitted_preparation_seconds=recipe_process.preparation_seconds(spec),
                admitted_verification_seconds=recipe_process.verification_seconds(spec),
                peak_rss_bytes=process_memory()['peak_since_exec_bytes'],memory_counters=process_memory(),
                protocol_sha256=sha(args.protocol),live_service_restarted=False,production_capture_limit_changed=False)
            for name,expected_source in protocol['source_sha256'].items():
                if sha(ROOT/name)!=expected_source:raise ValueError('Qualification source changed during execution')
            write(args.output/'verified-execution-material.json',execution['material'])
            write(args.output/'verified-execution-spec.json',spec)
    except BaseException as error:
        result.update(status='failed',error=repr(error))
        raise
    finally:
        if dataset is not None:dataset.close()
        shutil.rmtree(scratch,ignore_errors=False) if scratch.exists() else None
        result.update(phases=phases,elapsed_seconds=time.monotonic()-started,final_memory_counters=process_memory(),
            temporary_material_and_staging_removed=not scratch.exists())
        write(args.output/'results.json',result)


if __name__=='__main__':main()
