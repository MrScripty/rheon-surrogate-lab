"""Freeze the corrected learned generator and fresh current-regime confirmation scenes."""
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT/'docs/plans/full-ai-data-diversity-v1'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh(path, value):
    with path.open('x') as output:
        output.write(json.dumps(value, indent=2)+'\n')


if __name__ == '__main__':
    reg_path = DOCS/'table-force-regression-protocol-01.json'
    reg = read(reg_path)
    sources = dict(reg['source_sha256'])
    for p, expected in sources.items():
        if digest(ROOT/p) != expected:
            raise ValueError('Selected candidate source changed before confirmation: '+p)
    strict = read(ROOT/reg['output_identity_control'])
    if len(strict['rows']) != 15360 or not all(r['metrics']['all_accuracy_criteria_pass'] for r in strict['rows']):
        raise ValueError('Full corrected-force consumed regression required before selection')
    tables = read(ROOT/reg['table_force_qualification'])
    if tables['status'] != 'passed_known_force_table_bitwise_parity':
        raise ValueError('Exact scalar-table construction must qualify first')
    generator = str(Path(__file__).resolve().relative_to(ROOT))
    sources[generator] = digest(__file__)
    sources[str(reg_path.relative_to(ROOT))] = digest(reg_path)
    selection = dict(status='frozen_before_fresh_reference_capture', created_unix=time.time(),
        model='Existing restored learned flux/momentum/transport/pressure, causal pressure reuse',
        head_pair=reg['pair'], pressure_pair=reg['pressure_pair'],
        known_external_inputs='Qualified native-scalar external controls with identical coordinate-table reuse',
        horizons=[1, 2, 3, 4], output_timestep_s=1/60,
        geometries_xyz=[[16, 16, 16], [9, 11, 11], [15, 6, 8], [32, 32, 32]],
        cells_per_metre=[8, 16], source_evidence=sources,
        no_numerical_solver_in_candidate=True, optimizer_updates=0,
        fresh_confirmation_scope='New seeds in the original current-regime forcing distribution; separate from the broader density/grid/force dataset',
        table_complete_state_regression_must_pass_before_confirmation_scoring=True,
        latency_qualified=False, full_generator_targets_verified=False,
        original_frozen_studies_unchanged=True)
    selection_path = DOCS/'strict-force-selection-01.json'
    fresh(selection_path, selection)
    prior = []
    for directory in ('full-ai-next-priorities-v1', 'full-ai-tight-targets-v1', 'full-ai-data-diversity-v1'):
        prior.extend(sorted((ROOT/'docs/plans'/directory).rglob('*scene-reservation*.json')))
    known = set()
    fingerprints = {}
    for path in prior:
        document = read(path)
        for row in document.get('rows', []):
            seed = row.get('seed', row.get('scene', {}).get('seed'))
            if seed is not None:
                known.add(seed)
        fingerprints[str(path.relative_to(ROOT))] = digest(path)
    rng = random.Random(2026102609)
    rows = []
    for counts in selection['geometries_xyz']:
        for density in (8, 16):
            for _ in range(2):
                seed = rng.randrange(1, 2**31)
                while seed in known:
                    seed = rng.randrange(1, 2**31)
                known.add(seed)
                rows.append(dict(identifier=f'strict-force-fresh-{len(rows):02d}', split='reserved',
                    seed=seed, counts=counts, cells_per_metre=density, steps=240, warmup_steps=0,
                    center=[rng.uniform(.3, .7) for _ in range(3)], acceleration=rng.uniform(.08, .18),
                    radius=rng.uniform(.16, .26), swirl=rng.uniform(-.65, .65),
                    fluctuation=rng.uniform(.1, .3), frequency=rng.uniform(1.5, 2.5), sgs=False))
    reservation = dict(status='fresh_confirmation_reserved_before_native_labels',
        created_unix=time.time(), seed=2026102609, rows=rows,
        expected_scenes=16, steps=240, source_evidence=fingerprints,
        generator_sha256=digest(__file__), seeds_disjoint_from_all_listed_reservations=True,
        reserved_references_captured=False, frozen_selection_sha256=digest(selection_path),
        wider_population_reserved52_references_untouched=True, optimizer_updates=0,
        future_model_changes_consume_this_confirmation_and_require_new_fresh_confirmation=True)
    reservation_path = DOCS/'strict-force-confirmation-scene-reservation-01.json'
    fresh(reservation_path, reservation)
    build = ROOT/'runs/full-ai-next-priorities-v1/temporal-build-02'
    assets = {str((build/p).relative_to(ROOT)): digest(build/p)
              for p in ('build-receipt.json', 'rheon-temporal-probe')}
    sources.update(fingerprints)
    for p in (selection_path, reservation_path):
        sources[str(p.relative_to(ROOT))] = digest(p)
    protocol = dict(status='registered_before_new_frozen_candidate_reference_capture',
        cpu=4, frozen_selection_sha256=digest(selection_path),
        original_scene_reservation_sha256=digest(reservation_path),
        source_sha256=sources, execution_assets_sha256=assets, expected_scenes=16, steps=240,
        reference_only_native_solver=True, maximum_wall_seconds=900,
        maximum_capture_bytes=2*1024**3, minimum_free_bytes=12*1024**3,
        disk_free_before_bytes=shutil.disk_usage(ROOT).free,
        total_data_program_conservative_cap_gib=23.66, prior_data_ceiling_gib=24,
        optimizer_updates=0, latency_claim=False, full_generator_targets_verified=False)
    fresh(DOCS/'strict-force-confirmation-capture-protocol-01.json', protocol)
    print('Frozen corrected full-AI candidate and16new disjoint confirmation scenes; no references consumed')
