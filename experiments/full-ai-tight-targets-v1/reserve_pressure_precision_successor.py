"""Reserve new whole scenes before selecting or fitting a precision successor."""
import hashlib
import json
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs/plans/full-ai-tight-targets-v1'


if __name__ == '__main__':
    target = DOCS / 'pressure-successor-fresh-scene-reservation-10.json'
    if target.exists():
        raise ValueError('Fresh successor reservation required')
    sources = sorted((ROOT / 'docs/plans/full-ai-next-priorities-v1').rglob('*scene-reservation*.json'))
    sources += sorted(DOCS.glob('*scene-reservation*.json'))
    known = set()
    fingerprints = {}
    for path in sources:
        document = json.loads(path.read_text())
        known.update(row['seed'] for row in document.get('rows', []) if 'seed' in row)
        fingerprints[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rng = random.Random(2026102167)
    rows = []
    for counts in ([16, 16, 16], [9, 11, 11], [15, 6, 8], [32, 32, 32]):
        for density in (8, 16):
            for replica in range(2):
                seed = rng.randrange(1, 2**31)
                while seed in known:
                    seed = rng.randrange(1, 2**31)
                known.add(seed)
                rows.append(dict(identifier=f'pressure-successor-fresh-{len(rows):02d}', split='reserved',
                    seed=seed, counts=counts, cells_per_metre=density, steps=240, warmup_steps=0,
                    center=[rng.uniform(.3, .7) for _ in range(3)], acceleration=rng.uniform(.08, .18),
                    radius=rng.uniform(.16, .26), swirl=rng.uniform(-.65, .65),
                    fluctuation=rng.uniform(.1, .3), frequency=rng.uniform(1.5, 2.5), sgs=False))
    document = dict(status='reserved_before_successor_selection_or_learning', created_unix=time.time(),
                    seed=2026102167, rows=rows, expected_scenes=16, steps=240,
                    forcing_sampling_ranges=dict(center=[.3, .7], acceleration=[.08, .18], radius=[.16, .26],
                        swirl=[-.65, .65], fluctuation=[.1, .3], frequency=[1.5, 2.5]),
                    seeds_disjoint_from_all_listed_prior_reservations=True,
                    reserved_references_captured=False, consume_only_after_selected_weights_and_inference_sources_frozen=True,
                    earlier_precision_scenes_are_consumed_diagnostic_evidence=True,
                    higher_density_or_size_phase_started=False, source_evidence=fingerprints,
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with target.open('x') as output:
        output.write(json.dumps(document, indent=2, allow_nan=False) + '\n')
    print('Reserved 16 new current-regime whole scenes; no references captured or fitting executed', flush=True)
