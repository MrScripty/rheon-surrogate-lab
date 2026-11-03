"""Freeze a passing learned divisor generator before fresh native confirmation."""
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT/'docs/plans/full-ai-data-diversity-v1'
BASE = ROOT/'runs/full-ai-data-diversity-v1'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2)+'\n').encode()


if __name__ == '__main__':
    names = ['learned-divisor-selection-02.json',
             'learned-divisor-confirmation-scene-reservation-02.json',
             'learned-divisor-confirmation-capture-protocol-02.json',
             'divisor-confirmation-resource-audit-02.json']
    if any((DOCS/name).exists() for name in names):
        raise ValueError('Fresh divisor confirmation registration required')
    reg_path = DOCS/'learned-divisor-consumed-regression-protocol-02.json'
    reg = read(reg_path)
    result_path = BASE/'learned-divisor-consumed-regression-02/results.json'
    result = read(result_path)
    if (result['protocol_sha256'] != digest(reg_path) or result['numerical_oracle_used']
        or result['pressure_execution'] != 'learned_divisor' or len(result['rows']) != 69120
        or not all(r['metrics']['all_accuracy_criteria_pass'] for r in result['rows'])):
        raise ValueError('All72consumed scenes and everyK1-4physical output must pass before selection')
    qualification_path = BASE/'callable-divisor-qualification-02/qualification.json'
    qualification = read(qualification_path)
    if qualification['status'] != 'passed_callable_divisor_generator_output_parity' or len(qualification['rows']) != 1920:
        raise ValueError('Actual callable output parity required')
    sources = dict(reg['source_sha256'])
    for path, expected in qualification['source_sha256'].items():
        if path in sources and sources[path] != expected:
            raise ValueError('Conflicting frozen source: '+path)
        sources[path] = expected
    for path, expected in sources.items():
        if digest(ROOT/path) != expected:
            raise ValueError('Selected learned source changed before capture: '+path)
    for path in (reg_path, result_path, qualification_path, Path(__file__).resolve(),
                 Path(__file__).with_name('capture_divisor_confirmation.py').resolve(),
                 Path(__file__).with_name('evaluate_divisor_generator.py').resolve(),
                 Path(__file__).with_name('measure_divisor_generator_v2.py').resolve()):
        sources[str(path.relative_to(ROOT))] = digest(path)
    selection = dict(status='frozen_before_new_native_labels', created_unix=time.time(),
        generator='experiments/full-ai-data-diversity-v1/divisor_generator.py:load_model',
        pair=reg['pair'], pressure_pair=reg['pressure_pair'], source_evidence=dict(sources),
        horizons=[1, 2, 3, 4], output_timestep_s=1/60,
        geometries_xyz=[[16, 16, 16], [9, 11, 11], [15, 6, 8], [32, 32, 32]],
        cells_per_metre=[8, 16],
        pressure_execution='Explicit cosine tensor transforms; divide by reciprocals of unchanged restored learned gains',
        forcing='Qualified scalar coordinate reuse; known controls only',
        numerical_pressure_solver=False, analytical_eigenvalues_computed_in_candidate=False,
        new_parameter_fits=0, all72_consumed_scenes_pass=True,
        latency_qualified=False, full_generator_targets_verified=False,
        worker_contract='One owning application caller per model; batch1; up to4physicalCPUcores',
        broader_reserved52_references_untouched=True)
    selection_bytes = encoded(selection)
    selection_sha = hashlib.sha256(selection_bytes).hexdigest()
    fingerprints, known = {}, set()
    for directory in ('full-ai-next-priorities-v1', 'full-ai-tight-targets-v1', 'full-ai-data-diversity-v1'):
        for path in sorted((ROOT/'docs/plans'/directory).rglob('*scene-reservation*.json')):
            document = read(path)
            for row in document.get('rows', []):
                seed = row.get('seed', row.get('scene', {}).get('seed'))
                if seed is not None:
                    known.add(seed)
            fingerprints[str(path.relative_to(ROOT))] = digest(path)
    rng = random.Random(2026102903)
    rows = []
    for counts in selection['geometries_xyz']:
        for density in (8, 16):
            for _ in range(2):
                seed = rng.randrange(1, 2**31)
                while seed in known:
                    seed = rng.randrange(1, 2**31)
                known.add(seed)
                rows.append(dict(identifier=f'learned-divisor-fresh-{len(rows):02d}', split='reserved',
                    seed=seed, counts=counts, cells_per_metre=density, steps=240, warmup_steps=0,
                    center=[rng.uniform(.3, .7) for _ in range(3)], acceleration=rng.uniform(.08, .18),
                    radius=rng.uniform(.16, .26), swirl=rng.uniform(-.65, .65),
                    fluctuation=rng.uniform(.1, .3), frequency=rng.uniform(1.5, 2.5), sgs=False))
    reservation = dict(status='reserved_before_fresh_confirmation_labels', created_unix=time.time(),
        seed=2026102903, rows=rows, expected_scenes=16, steps=240,
        source_evidence=fingerprints, generator_sha256=digest(__file__),
        frozen_selection_sha256=selection_sha, seeds_disjoint_from_listed_reservations=True,
        reserved_references_captured=False, optimizer_updates=0,
        scope='New seeds in original current-regime force distribution; broader data52reservedscenes remain separate')
    reservation_bytes = encoded(reservation)
    free = shutil.disk_usage(ROOT).free
    if free < 14*1024**3:
        raise ValueError('Independent2GiBcap must preserve12GiBfree')
    raw_bound = sum(241*(32*__import__('math').prod(r['counts'])+4*(r['counts'][0]*r['counts'][1]+
        r['counts'][0]*r['counts'][2]+r['counts'][1]*r['counts'][2])) for r in rows)
    audit = dict(status='prospective_bounded_independent_confirmation_extension', created_unix=time.time(),
        previous_finite_capture_estimate_gib=23.66, additional_confirmation_cap_gib=2.,
        new_finite_capture_estimate_gib=25.66, extended_program_ceiling_gib=28.,
        current_free_bytes=free, minimum_free_bytes=12*1024**3,
        raw_complete_state_bound_bytes=raw_bound,
        original_capture_protocols_and_caps_unchanged=True,
        reason='A changed learned pressure execution needs fresh independent references; retain all prior failures and evidence')
    audit_bytes = encoded(audit)
    capture_sources = dict(sources)
    capture_sources.update(fingerprints)
    for name, content in zip((names[0], names[1], names[3]),
                             (selection_bytes, reservation_bytes, audit_bytes)):
        capture_sources[str((DOCS/name).relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
    build = ROOT/'runs/full-ai-next-priorities-v1/temporal-build-02'
    assets = {str((build/name).relative_to(ROOT)): digest(build/name)
              for name in ('build-receipt.json', 'rheon-temporal-probe')}
    protocol = dict(status='registered_before_frozen_divisor_reference_capture', cpu=5,
        frozen_selection_sha256=selection_sha,
        original_scene_reservation_sha256=hashlib.sha256(reservation_bytes).hexdigest(),
        source_sha256=capture_sources, execution_assets_sha256=assets,
        expected_scenes=16, steps=240, reference_only_native_solver=True,
        maximum_wall_seconds=900, maximum_capture_bytes=2*1024**3,
        minimum_free_bytes=12*1024**3, optimizer_updates=0, latency_claim=False,
        full_generator_targets_verified=False)
    for name, content in zip(names, (selection_bytes, reservation_bytes, encoded(protocol), audit_bytes)):
        with (DOCS/name).open('xb') as output:
            output.write(content)
    print('Frozen callable learned divisor generator and16new disjoint scenes before reference generation')
