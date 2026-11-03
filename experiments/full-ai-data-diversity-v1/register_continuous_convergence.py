"""Freeze a bounded reference-only frequency/discretization pilot before running it."""
import hashlib
import json
from pathlib import Path
import shutil
import time
from continuous_force_program_v2 import frequency_scene
from force_program import PROFILES, MOTIONS

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    output = ROOT/'docs/plans/full-ai-data-diversity-v1/continuous-reference-convergence-protocol-01.json'
    if output.exists():
        raise ValueError('Fresh prospective reference pilot registration required')
    qualification = 'runs/full-ai-data-diversity-v1/continuous-reference-qualification-02/qualification.json'
    q = json.loads((ROOT/qualification).read_text())
    if q['status'] != 'passed_fine_time_external_control_qualification':
        raise ValueError('Exact fine-time forcing and restart qualification required')
    sources = list(q['source_sha256'])+[qualification]
    sources += ['experiments/full-ai-data-diversity-v1/'+v for v in (
        'register_continuous_convergence.py', 'pilot_continuous_convergence.py', 'reference_difference.py')]
    cases = []
    for f_index, frequency in enumerate((.1, 1., 10., 100.)):
        for p_index, profile in enumerate(PROFILES):
            index = len(cases)
            motion = 'oscillating' if profile == 'continuous' else MOTIONS[(f_index+p_index) % 3]
            density = (25, 100)[(f_index+p_index) % 2]
            scene = frequency_scene(2026102083+index, [9, 11, 7], density,
                                    profile, motion, 60, frequency, frequency)
            cases.append(dict(identifier=f'frequency-pilot-{index:02d}', scene=scene,
                note='A finite training-reference diagnostic, not a reserved model evaluation or full frequency-coverage qualification'))
    reg = dict(status='registered_before_reference_frequency_pilot', created_unix=time.time(),
        cpu=6, qualification=qualification, cases=cases, substeps=[1, 4, 16, 64],
        source_sha256={p: digest(ROOT/p) for p in sorted(set(sources))},
        maximum_wall_seconds=900, maximum_output_bytes=512*1024**2,
        minimum_free_bytes=12*1024**3, disk_free_before_bytes=shutil.disk_usage(ROOT).free,
        storage_context='At most512MiB additional pilot output; previous finite capture estimate18.16GiB plus this pilot stays below24GiB. Preserve12GiB free floor.',
        output_timestep_s=1/60, physical_duration_s=1.,
        low_frequency_scope='0.1Hz and some1Hz components are only partially observed within one second. Random initial phase does not prove full-cycle or long-duration fidelity.',
        pulse_scope='Regular and irregular pulses have sharp edges; repetition/event rate is not a bandwidth limit. Report sensitivity rather than inferring convergence from samples per cycle alone.',
        comparators='Compare all common actual output states against the finest64-substep trial. Finest is a numerical comparator, not exact physical truth. Preserve native rejections.',
        inference_scope='No candidate inference or fitting; all native solves are offline reference generation.',
        model_targets_verified=False, optimizer_updates=0, reserved_references_consumed=False,
        long600_step_model_phase_started=False)
    output.write_text(json.dumps(reg, indent=2)+'\n')
    print('Registered', len(cases), 'reference frequency scenes and', len(cases)*4, 'bounded trials')
