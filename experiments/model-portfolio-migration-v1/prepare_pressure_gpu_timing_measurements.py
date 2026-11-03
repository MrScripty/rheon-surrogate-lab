"""Reduce the original ABBA timing samples and package native scientific scalars."""
import base64
import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics
import struct
import sys
import time

from prepare_portfolio_measurements import source

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/model-portfolio-migration-v1'
DOC = ROOT / 'docs/plans/model-portfolio-migration-v1'
JOB = RUN / 'pressure-GPU-timing-pilot-01'
OUTPUT = RUN / 'native-pressure-GPU-timing-pilot-01'
REPORT = DOC / 'reports/pressure-GPU-timing-pilot-01.json'
SOURCE = RUN / 'sources/Tuldok-composition-source-artifacts'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run():
    require(not OUTPUT.exists() and not REPORT.exists(), 'Fresh reduction/package outputs required')
    sys.path.insert(0, str(SOURCE))
    from retained_measurements import MeasurementTable, encode, digest
    from projects import manifest as validate_manifest
    status = read(JOB / 'status.json')
    protocol = read(JOB / 'protocol.json')
    require(status['status'] == 'four_original_ABBA_GPU_timing_blocks_complete'
            and len(status['blocks']) == 4
            and not Path('/proc', str(status['pid'])).exists(), 'Original ABBA pilot must be terminal')
    require(status['protocol_sha256'] == sha(JOB / 'protocol.json'), 'Protocol changed')
    for path, expected in protocol['source_sha256'].items():
        require(sha(ROOT / path) == expected, 'Preregistered source changed: ' + path)
    frozen = read(ROOT / protocol['frozen_cases']['path'])
    require(sha(ROOT / protocol['frozen_cases']['path']) == protocol['frozen_cases']['sha256'],
            'Frozen pressure cases changed')
    score = read(DOC / 'reports/pressure-common-score-01.json')
    captures = {}
    blocks = []
    raw_sources = []
    fixed_settings = None
    hardware = None
    for index, (entry, identifier) in enumerate(zip(status['blocks'], protocol['block_order'], strict=True)):
        path = ROOT / entry['result']
        require(entry['identifier'] == identifier and sha(path) == entry['result_sha256'], 'Original block changed')
        block = read(path)
        require(block['status'] == 'complete_pressure_GPU_timing_pilot' and block['cases'] == 128
                and block['all_output_hashes_exact'] and block['architecture_unchanged']
                and block['learned_parameters_unchanged'] and not block['CPU_model_forward']
                and block['device_qualification']['forbidden_module_calls'] == 0
                and block['task_calls_timed'] == 2560 and not block['timed_module_hooks_enabled'],
                'Block precision/device/coverage qualification differs')
        settings = {key: block[key] for key in ('gpu_model', 'torch_version', 'cuda_version', 'host_affinity',
            'torch_threads', 'interop_threads', 'tf32', 'network_dtype', 'pressure_dtype', 'autocast',
            'deterministic_algorithms', 'cudnn_benchmark')}
        require(fixed_settings is None or settings == fixed_settings, 'Hardware/precision settings differ across blocks')
        fixed_settings = settings
        for observation in block['resource_observations']:
            require(not observation['other_compute_pids'], 'Unrelated GPU compute was observed')
            devices = list(csv.reader(observation['gpu'].splitlines(), skipinitialspace=True))
            require(len(devices) == 1, 'Pilot must bind one GPU')
            identity = devices[0][:4]
            require(hardware is None or hardware == identity, 'GPU/driver identity changed')
            hardware = identity
        original = protocol['accuracy'][identifier]
        require(sha(ROOT / original['path']) == original['sha256'], 'Retained accuracy changed')
        captures[identifier] = read(ROOT / original['path'])
        require(block['weights_sha256'] == captures[identifier]['weights_sha256']
                == score['models'][identifier]['weights_sha256'], 'Another checkpoint was timed')
        for row_index, (row, case, baseline) in enumerate(zip(block['rows'], frozen['cases'], captures[identifier]['rows'], strict=True)):
            require(row['case_id'] == case['id'] and row['input_sha256'] == baseline['input_sha256']
                    == case['original_case']['input_sha256']
                    and row['output_sha256'] == baseline['output_sha256']
                    and row['output_independent_of_inputs']
                    and len(row['raw_wall_nanoseconds']) == 20
                    and all(type(value) is int and value > 0 for value in row['raw_wall_nanoseconds']),
                    'Block case identity, output parity or timing samples differ')
            require(row['median_ms'] == statistics.median(row['raw_wall_nanoseconds']) / 1e6,
                    'Recorded case median is not the raw timing reduction')
        blocks.append(block)
        raw_sources.append(source('block-' + str(index + 1), 'Original timing block ' + str(index + 1), path.read_bytes()))

    models = {}
    for identifier in protocol['models']:
        own = [block for block in blocks if block['identifier'] == identifier]
        require(len(own) == 2, 'Two balanced blocks per model required')
        rows = []
        for index, case in enumerate(frozen['cases']):
            samples = [value for block in own for value in block['rows'][index]['raw_wall_nanoseconds']]
            require(len(samples) == 40, 'Exactly40samples per pooled case required')
            rows.append(dict(case_id=case['id'], dimensions=case['dimensions'],
                input_sha256=case['original_case']['input_sha256'],
                output_sha256=captures[identifier]['rows'][index]['output_sha256'],
                raw_wall_nanoseconds=samples, median_ms=statistics.median(samples) / 1e6,
                minimum_ms=min(samples) / 1e6, maximum_ms=max(samples) / 1e6))
        density = {}
        for value in sorted({case['cells_per_metre'] for case in frozen['cases']}):
            selected = [row for row in rows if row['dimensions']['cells_per_metre'] == value]
            density[str(value)] = dict(cases=len(selected), median_case_median_ms=statistics.median(row['median_ms'] for row in selected),
                                      worst_case_median_ms=max(row['median_ms'] for row in selected))
        models[identifier] = dict(weights_sha256=own[0]['weights_sha256'], rows=rows,
            summary=dict(cases=128, median_case_median_ms=statistics.median(row['median_ms'] for row in rows),
                         worst_case_median_ms=max(row['median_ms'] for row in rows),
                         block_median_ms=[block['case_median_median_ms'] for block in own],
                         by_cells_per_metre=density, accuracy_worst=score['models'][identifier]['worst']))
    ratios = [b['median_ms'] / a['median_ms'] for a, b in zip(models['C106']['rows'], models['C107']['rows'], strict=True)]
    report = dict(schema='rheon.complete-pressure-GPU-timing-pilot.v1', created_unix=time.time(),
        status='four_ABBA_blocks_exact_outputs_and_raw_timing_reductions_verified',
        protocol=dict(path=str((JOB / 'protocol.json').relative_to(ROOT)), sha256=sha(JOB / 'protocol.json')),
        raw_blocks=status['blocks'], settings=fixed_settings, GPU_identity=hardware,
        timing_scope=protocol['timing_scope'], units='ms/pressure_request', cases_per_model=128,
        measured_calls=10240, new_model_fits=0, new_reference_generation=False,
        models=models, paired_C107_over_C106_case_median_ratio=dict(minimum=min(ratios), median=statistics.median(ratios),
            maximum=max(ratios), C107_faster_cases=sum(value < 1 for value in ratios), paired_cases=128),
        all_output_hashes_exact=True, all_source_bindings_exact=True,
        CPU_model_forward=False, resource_observations=[block['resource_observations'] for block in blocks],
        sampled_resources_not_continuous_exclusivity_proof=True, full_simulation_step_timing=False,
        full_portfolio_complete=False, independent_confirmation=False,
        producer=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(Path(__file__))))
    REPORT.write_bytes(encode(report))

    template_path = RUN / 'native-measurements-packet-05/measurements.json'
    require(sha(template_path) == 'c78bbc342cad9f3d9af5116f2c272e85f1d01344f7af0f33f4dce000de828400',
            'Qualified native pressure template changed')
    template = read(template_path)
    original_sources = {entry['id']: entry for entry in template['sources']}
    original_context = json.loads(base64.b64decode(original_sources['context']['data_base64']))
    scope = copy.deepcopy(original_context['scope'])
    scope.update(id='old-pressure-complete-GPU-call-128-v1',
        label='Complete GPU pressure calls · original128consumed requests',
        task='Complete pressure proposal from native CPU arrays to independent CPU output; not a simulation step.',
        timing_scope=protocol['timing_scope'], timing_statistic='Case median of40warm calls in two balanced20-sample blocks',
        hardware=dict(gpu_identity=hardware, host_affinity=[4], torch_threads=1, interop_threads=1))
    definitions = copy.deepcopy(original_context['metrics'])
    time_definition = next(item for item in definitions if item['id'] == 'pressure_call_ms')
    time_definition.update(label='Complete GPU pressure-call case median', units='ms/request',
        description='Median of40retained complete warm CPU-array to GPU pressure to independent CPU-output wall times.',
        source_kind='Derived median of retained raw wall-clock nanoseconds',
        formula='median(concatenated two20-sample raw_wall_nanoseconds arrays)/1e6')
    context = dict(scope=scope, metrics=definitions, cases=original_context['cases'],
                   original_packet_manifest_sha256=original_context['original_packet_manifest_sha256'],
                   common_score_report_sha256=sha(DOC / 'reports/pressure-common-score-01.json'))
    dimensions = copy.deepcopy(template['dimensions'])
    for identity, label, kind, scope_name in [('host_logical_cpu','Host logical CPU','integer','subject'),
        ('host_threads','Host threads','integer','subject'), ('execution_variant','Execution boundary','string','subject'),
        ('samples','Identical-output timed samples','integer','observation'),
        ('balanced_blocks','Balanced model blocks','integer','observation'), ('statistic','Observation statistic','string','observation')]:
        dimensions.append(dict(id=identity, label=label, type=kind, units=None, scope=scope_name))
    subjects = copy.deepcopy(template['subjects'])
    sources = [source('context','Shared case, metric and complete-call scope definitions',encode(context)),
               source('pilot','Reduced original timing samples and qualification',REPORT.read_bytes()), *raw_sources]
    for identifier in protocol['models']:
        sources.append(source(identifier,'Original exact GPU accuracy capture ' + identifier,
                              (ROOT / protocol['accuracy'][identifier]['path']).read_bytes()))
    for subject in subjects:
        subject.update(label=subject['id'] + ' · complete GPU pressure call',
                       reference=dict(source_id='pilot', pointer='/models/' + subject['id']))
        subject['dimensions'].update(host_logical_cpu=4, host_threads=1,
            execution_variant='captured_CUDA_eager_native_CPU_arrays_owned_CPU_pressure_v1')
    metrics = []
    for index, definition in enumerate(definitions):
        metrics.append(dict(id=definition['id'], label=definition['label'], units=definition['units'],
                            definition=dict(source_id='context', pointer='/metrics/' + str(index), sha256=digest(definition))))
    observations = []
    for subject in subjects:
        identifier = subject['id']
        for index, case in enumerate(template['cases']):
            for metric in metrics:
                timed = metric['id'] == 'pressure_call_ms'
                value = (models[identifier]['rows'][index]['median_ms'] if timed else
                         captures[identifier]['rows'][index]['metrics'][metric['id']])
                reference = dict(source_id='pilot' if timed else identifier,
                    pointer='/models/' + identifier + '/rows/' + str(index) + '/median_ms' if timed else
                            '/rows/' + str(index) + '/metrics/' + metric['id'])
                observations.append(dict(id='o-' + str(len(observations)).zfill(6), subject_id=identifier,
                    case_id=case['id'], metric_id=metric['id'], value=value, state='measured', reason=None,
                    reference=reference, dimensions=dict(samples=40, balanced_blocks=2,
                        statistic='median40warm_complete_calls' if timed else 'retained_accuracy_exact_output_hashes')))
    packet = dict(format=template['format'], schema_version=1, title=scope['label'],
        scope={key: scope[key] for key in ('id','label','population')} | dict(definition=dict(
            source_id='context', pointer='/scope', sha256=digest(scope))),
        dimensions=dimensions, subjects=subjects, cases=copy.deepcopy(template['cases']), metrics=metrics,
        observations=observations, sources=sources)
    raw = encode(packet)
    table = MeasurementTable(raw)
    require(table.metadata()['observation_count'] == 1280, 'Native scalar count differs')
    for metric in metrics:
        query = table.query(metric['id'])
        for summary in query['summary']:
            expected = max(row['value'] for row in observations if row['subject_id'] == summary['subject_id']
                           and row['metric_id'] == metric['id'])
            require(summary['states']['measured'] == 128 and struct.pack('!d', summary['maximum']['value']) == struct.pack('!d', expected),
                    'Native maximum or complete denominator differs')
    OUTPUT.mkdir()
    packet_path = OUTPUT / 'measurements.json'
    packet_path.write_bytes(raw)
    declaration = dict(schema_version=1, id=hashlib.sha256(raw).hexdigest()[:32],
        name=scope['label'], data=[dict(id='complete-GPU-pressure-timing', label=scope['label'],
            path=str(packet_path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest())],
        recipes=[], evidence=[], requirements=[])
    validate_manifest(declaration)
    (OUTPUT / 'source-project.json').write_bytes(encode(declaration))
    qualification = dict(status='complete_GPU_pressure_timing_native_packet_prepared_not_imported',
        created_unix=time.time(), packet_sha256=hashlib.sha256(raw).hexdigest(), packet_bytes=len(raw),
        source_manifest_sha256=sha(OUTPUT / 'source-project.json'), subjects=2, cases=128, measured=1280,
        original_raw_timed_calls=10240, report=dict(path=str(REPORT.relative_to(ROOT)), sha256=sha(REPORT)),
        existing_native_project_mutated=False, new_model_forwards=0, new_reference_generation=False,
        native_UI_qualified=False, full_portfolio_complete=False)
    (OUTPUT / 'preparation.json').write_bytes(encode(qualification))
    print(json.dumps(dict(status=qualification['status'], packet_bytes=len(raw),
                          models={name: value['summary'] for name,value in models.items()},
                          paired=report['paired_C107_over_C106_case_median_ratio'])))


if __name__ == '__main__':
    run()
