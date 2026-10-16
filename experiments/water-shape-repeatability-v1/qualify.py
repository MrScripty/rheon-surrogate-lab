"""Qualify a restored checkpoint through existing native CPU procedures."""
import argparse
import fcntl
import sys

from execution import LAB, command, read, utc, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--trial', required=True)
    args = parser.parse_args()
    sequence = read(args.sequence)
    trial = next(t for t in sequence['trials'] if t['name'] == args.trial)
    root = LAB/sequence['root']
    directory = root/trial['name']
    # Independent numerical pools share cores, so only one qualification owns them.
    with (root/'cpu-qualification.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        receipt = read(directory/'managed-full/receipt.json')
        if receipt['status'] != 'completed_and_drained':
            raise ValueError('completed managed custody required')
        project = directory/'project'
        artifacts = directory/'managed-full/stage-01/artifacts'
        multigrid = LAB/'experiments/water-multigrid-learning-v1'
        timings = []
        for name, packet, population, extra in (
            ('fit', sequence['training_packet'], 'train', []),
            ('development', sequence['validation_packet'], 'validation', []),
            ('fixed-eight', sequence['validation_packet'], 'validation',
             ['--network-budget', '8', '--fixed-budget']),
        ):
            timings.append(command([sys.executable, multigrid/'evaluate.py',
                '--project', project, '--artifacts', artifacts, '--packet', LAB/packet,
                '--population', population, '--binary', LAB/sequence['native_binary'],
                '--output', directory/name, *extra], directory, name))
        timings.append(command([sys.executable, multigrid/'mode_probe.py',
            '--project', project, '--artifacts', artifacts,
            '--packet', LAB/sequence['probe_packet'], '--output', directory/'mode-probe'],
            directory, 'modes'))
        model = {'name': trial['name'], 'project': str(project.relative_to(LAB)),
                 'artifacts': str(artifacts.relative_to(LAB)), 'single_evaluation': True}
        for name, key, workers in (('grid-study', 'fresh-development', 8),
                                   ('prior-study', 'prior-grids', 7),
                                   ('long-study', 'long-study', 3)):
            study = read(LAB/sequence[key])
            study['models'] = [model]
            config = directory/(name+'.json')
            write(config, study)
            timings.append(command([sys.executable, multigrid/'study.py', '--config', config,
                '--output', directory/name, '--workers', str(workers)], directory, name))
        for name, packet, stage, extra in (
            ('raw-fit-native', sequence['training_packet'], 'stage-01', ['--training-only']),
            ('raw-validation-native', sequence['validation_packet'], 'stage-02', []),
        ):
            timings.append(command([sys.executable,
                LAB/'experiments/water-conditioning-v1/native_parallel.py', '--packet', LAB/packet,
                '--inference', directory/'managed-full'/stage/'artifacts',
                '--binary', LAB/sequence['native_binary'], '--output', directory/name, *extra],
                directory, name))
        fit = read(directory/'fit/report.json')['native']
        dev = read(directory/'development/report.json')['native']
        qualified = (fit['float32_qualified'] == fit['cases']
                     and dev['float32_qualified'] == dev['cases']
                     and all(row['qualified'] for name in ('grid-study','prior-study','long-study')
                             for row in read(directory/name/'report.json')['aggregate']))
        write(directory/'cpu-complete.json', {'status': 'complete', 'recorded_at_utc': utc(),
            'all_development_qualified': qualified, 'commands': timings,
            'scope': 'concurrent CPU qualification throughput, not isolated inference latency'})
        print(trial['name'], 'CPU complete; all development qualified:', qualified, flush=True)


if __name__ == '__main__':
    main()
