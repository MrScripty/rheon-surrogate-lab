"""Retain one source-list and one packet-metadata read, without request retries.

Native API wall time includes transport and canonical checks. This does not
measure browser rendering, project Open, model inference or GPU latency.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from import_common_generator_measurements import ROOT, CONTROLLER, PROJECT, read, save, binding, require


def run(job, expected_commit):
    require(not job.exists() and job.is_relative_to(ROOT/'runs/model-portfolio-migration-v1'),
            'Fresh task-owned native read directory required; do not replay an unknown request')
    state = read(CONTROLLER/'controller.json')
    source = Path(state['source'])
    require(state['status'] == 'ready' and state['source_commit'] == expected_commit
            and subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() == expected_commit
            and not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip(),
            'Exact clean installed native source required')
    job.mkdir()
    header = binding(PROJECT/'.tuldok/project.json')
    checkpoint = read(PROJECT/'.tuldok/project.json')['checkpoint']
    result = dict(status='native_measurement_read_owner_started', pid=os.getpid(), created_unix=time.time(),
        source_commit=expected_commit, canonical_header=header, checkpoint=checkpoint,
        producer=binding(Path(__file__)), scope=__doc__.strip(), calls=[], native_mutations=0,
        model_forwards=0, project_dataset_or_weight_copies=0, browser_rendering_measured=False,
        whole_Open_measured=False, load_before=list(os.getloadavg()))
    save(job/'status.json', result)
    sys.path.insert(0, str(source))
    import agent_api
    client = agent_api.ProjectFilesOwnerClient(state['origin'],
        (CONTROLLER/'private/owner-session.txt').read_text().strip(), state['catalog_id'], timeout=300)
    try:
        client.bootstrap()
        selected = client.call('project_files.status', {})
        save(job/'selection.response.json', selected)
        require(selected['status'] == 200 and selected['result']['state'] == 'open'
                and selected['result']['project']['checkpoint'] == checkpoint, 'Exact open checkpoint required')
        common = dict(binding=selected['result']['binding'], checkpoint=checkpoint)
        def once(name, operation, arguments):
            save(job/(name+'.request.json'), dict(operation=operation, arguments=arguments,
                source_commit=expected_commit, origin=state['origin'], catalog_id=state['catalog_id']))
            result['status'] = name+'_request_in_flight'
            save(job/'status.json', result)
            started = time.monotonic()
            response = client.call(operation, arguments)
            elapsed = time.monotonic()-started
            save(job/(name+'.response.json'), response)
            require(response['status'] == 200 and response['result']['checkpoint'] == checkpoint,
                    'Original native read must acknowledge its exact checkpoint')
            result['calls'].append(dict(name=name, operation=operation, wall_seconds=elapsed,
                response=binding(job/(name+'.response.json'))))
            save(job/'status.json', result)
            return response['result']['value']
        sources = once('sources', 'project_files.measurement_sources', dict(common, offset=0, limit=40))
        require(sources['total'] == len(sources['items']) == 31, 'All original 31 sources must remain present')
        target = sources['items'][0]
        metadata = once('metadata', 'project_files.measurement_metadata', dict(common,
            origin_catalog_id=target['origin_catalog_id'], packet_sha256=target['id']))
        require(metadata['source'] == {k: v for k, v in target.items() if k != 'availability'},
                'Read metadata must retain the exact originally selected source')
        require(binding(PROJECT/'.tuldok/project.json') == header, 'Native reads changed the project checkpoint')
        result.update(status='two_native_measurement_reads_acknowledged', finished_unix=time.time(),
            native_measurement_sources=31, target_packet_sha256=target['id'],
            target_origin_catalog_id=target['origin_catalog_id'], load_after=list(os.getloadavg()),
            project_left_open=True)
    except BaseException as error:
        result.update(status='native_read_stopped_for_explicit_reconciliation', error_type=type(error).__name__)
        raise
    finally:
        save(job/'status.json', result)
    print([(row['name'], row['wall_seconds']) for row in result['calls']], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    run(args.job.resolve(), args.expected_commit)
