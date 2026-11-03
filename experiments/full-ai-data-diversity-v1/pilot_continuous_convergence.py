"""Offline reference pilot: physical-frequency inputs and temporal refinement."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
from support import ROOT, read, write, digest
from continuous_force_program_v2 import sample_scene
from continuous_reference_client import ContinuousReferenceClient
from reference_difference import difference
import numpy as np
import torch


def run(args):
    if args.output.exists():
        raise ValueError('Fresh reference convergence pilot required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Registered convergence pilot changed: '+path)
    qualification = read(ROOT/reg['qualification'])
    if qualification['status'] != 'passed_fine_time_external_control_qualification':
        raise ValueError('Fine-time force transport qualification required first')
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    trials = []
    result = dict(status='running', trials=trials, protocol_sha256=digest(args.protocol),
        source_sha256=reg['source_sha256'], reference_only=True, optimizer_updates=0,
        reserved_references_consumed=False, model_targets_verified=False,
        pressure_time_semantics='Pressure is the last internal projection pressure; its forcing time precedes the published endpoint by one reference timestep.',
        comparison_scope='Descriptive discretization sensitivity on finite pilot scenes; finest trial is a comparator, not exact continuous-time truth.')
    for case in reg['cases']:
        states_by_factor = {}
        case_rows = []
        for factor in reg['substeps']:
            if time.monotonic()-started > reg['maximum_wall_seconds']:
                raise TimeoutError('Registered finite reference pilot deadline')
            if shutil.disk_usage(args.output).free < reg['minimum_free_bytes']:
                raise ValueError('Reference pilot disk floor reached')
            scene = sample_scene(case['scene'], factor)
            directory = args.output/case['identifier']/f'substeps-{factor}'
            client = ContinuousReferenceClient(scene, directory)
            states = {}
            frames = []
            error = None
            try:
                _, states[0] = client.inspect()
                with (directory/'fine-frames.jsonl').open('x') as fine_log:
                    for step in range(1, scene['steps']+1):
                        frame = client.advance()
                        fine_log.write(json.dumps(dict(output_step=step, **frame),
                                                  separators=(',', ':'))+'\n')
                        aggregate = frame['frame']
                        frames.append(dict(aggregate, output_step=step,
                            fine_steps_attempted=len(frame['fine_frames']),
                            fine_steps_accepted=sum(f['frame']['advanced'] for f in frame['fine_frames']),
                            all_fine_qualified=all(f['frame']['quality']['qualified'] for f in frame['fine_frames'])))
                        if not aggregate['advanced']:
                            break
                        packet, state = client.inspect()
                        if packet['step'] != step or packet['native_step'] != step*factor or packet['time_s'] != step/60:
                            raise ValueError('Actual reference clock mismatch')
                        states[step] = state
            except BaseException as failure:
                error = repr(failure)
                raise
            finally:
                owner = client.close()
            if not owner['reaped'] or owner['returncode'] != 0 or not owner['graceful_protocol_complete']:
                raise ValueError('Every reference pilot child must complete and reap')
            archive = directory/'states.npz'
            np.savez_compressed(archive, **{f's{step:03d}_{key}': value
                for step, state in states.items() for key, value in state.items()})
            row = dict(identifier=case['identifier'], substeps=factor,
                counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                force_profile=scene['forcing_program']['profile'], motion=scene['forcing_program']['motion'],
                force_frequency_parameter_hz=scene['forcing_program']['pulse_frequency_hz'],
                variance_frequencies_hz=scene['forcing_program']['variance_frequencies_hz'],
                motion_frequencies_hz=scene['forcing_program']['motion_frequency_hz'],
                sharp_pulse_edges=scene['forcing_program']['sharp_pulse_edges'],
                output_timestep_s=1/60, reference_timestep_s=1/(60*factor),
                complete=len(states) == scene['steps']+1 and all(f['all_fine_qualified'] for f in frames), accepted_output_steps=len(states)-1,
                accepted_fine_steps=sum(f['fine_steps_accepted'] for f in frames),
                worst_courant=max(f['quality']['courant'] for f in frames),
                worst_actual_divergence=max(f['quality']['actual_divergence'] for f in frames),
                states_archive=str(archive.relative_to(ROOT)), states_archive_sha256=digest(archive),
                fine_frames_sha256=digest(directory/'fine-frames.jsonl'),
                native_process=owner, frames=frames, error=error)
            states_by_factor[factor] = states
            case_rows.append(row)
            trials.append(row)
            write(args.output/'results.json', result)
            print(case['identifier'], 'factor', factor, 'accepted', len(states)-1, flush=True)
        finest = reg['substeps'][-1]
        reference = states_by_factor[finest]
        for row in case_rows:
            factor = row['substeps']
            comparisons = [dict(output_step=step, physical_time_s=step/60,
                                fields=difference(state, reference[step]))
                for step, state in states_by_factor[factor].items() if step and step in reference]
            row['compared_to_substeps'] = finest
            row['comparisons'] = comparisons
            row['worst_relative_linf_to_finest_peak'] = {field: max(
                (v['fields'][field]['relative_linf_to_reference_peak'] for v in comparisons), default=None)
                for field in ('pressure', 'velocity', 'tracer')}
        write(args.output/'results.json', result)
        used = sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
        if used > reg['maximum_output_bytes']:
            raise ValueError('Finite reference pilot storage cap exceeded')
    result.update(status='completed_reference_frequency_convergence_pilot',
                  elapsed_seconds=time.monotonic()-started,
                  temporal_convergence_qualified=False)
    write(args.output/'results.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
