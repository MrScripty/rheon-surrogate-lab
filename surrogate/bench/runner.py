"""Score subjects on a suite of cases: own-history rollouts, one timing rule for all.

A case is a reference trajectory with its forcing. A subject is anything that can
advance a fluid state. For each case the runner

  1. times the subject's first pass: a few warm-up calls, then repeated calls on
     the same input, taking the median (the lab's earlier matched benchmarks timed
     a block the same way);
  2. rolls the subject out from the case's first state on its own outputs and
     scores every output against the reference.
"""
import statistics
import time
from dataclasses import dataclass, field

import torch

from ..fields import FluidState, Scene
from . import metrics, results

TIMING_WARMUPS = 3
TIMING_SAMPLES = 10


@dataclass(frozen=True)
class Case:
    """One reference trajectory to reproduce.

    reference   states at successive output intervals, fields [T+1, ...]; index 0 is the start.
    forcing     (fu, fv, fw), the velocity added during each interval, m/s, [T, ...].
    scene       physical constants, one value each (shape [1]).
    start       output step of reference index 0 within its recorded scene.
    parameters  the recorded scene's own parameter dictionary, for subjects that rebuild
                the forcing themselves (the solvers do).
    right_hand_sides  the reference pressure equation's right-hand side per state [T+1, Z, Y, X], if recorded.
    attributes  what Tuldok shows as columns for this case (scene name, grid, spacing, ...).
    """
    id: str
    reference: FluidState
    forcing: tuple
    scene: Scene
    start: int = 0
    parameters: dict = field(default_factory=dict)
    right_hand_sides: object = None
    attributes: dict = field(default_factory=dict)

    @property
    def intervals(self):
        return self.forcing[0].shape[0]


@dataclass(frozen=True)
class Suite:
    """A named set of cases. `cases` is any iterable, so large suites can load lazily."""
    id: str
    label: str
    population: str            # results.POPULATIONS: how these cases relate to training data
    description: str
    velocity_error_limit: float
    cases: object


class Unsupported(Exception):
    """Raised by a subject for a case outside what it can run; recorded as not applicable, not as a failure."""


class Subject:
    """Something that advances a fluid state. Subclasses set the attributes and `advance`."""
    id: str
    label: str
    outputs_per_pass: int = 1
    attributes: dict = {}

    def begin(self, case):
        """Prepare for a case (not timed). Raise Unsupported if the case cannot be run."""

    def advance(self, state, forcing, scene, step):
        """Return `outputs_per_pass` future states of `state`, which is at output step `step`.

        Fields arrive with a batch dimension of one in host memory, and outputs must
        be back in host memory on return: the runner times this whole call.
        """
        raise NotImplementedError

    def own_seconds(self):
        """Computation time of the last `advance` as measured by the subject itself, or None.

        For subjects that run in another process, where the runner's clock would mostly
        measure moving the state in and out.
        """
        return None

    def end(self):
        """Release whatever `begin` acquired."""


def _not_applicable():
    return {'values': dict.fromkeys(metrics.IDS), 'failure': None}


def _timed(subject, state, forcing, scene, step):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = subject.advance(state, forcing, scene, step)
    elapsed = time.perf_counter() - started
    own = subject.own_seconds()
    return outputs, elapsed if own is None else own


def run_case(subject, case, velocity_error_limit):
    """Time and roll out one subject on one case; returns {'values', 'failure'} for the result document."""
    k = subject.outputs_per_pass
    passes = case.intervals // k
    if passes < 1:
        return _not_applicable()   # the case is shorter than one pass of this subject
    start = case.reference.map(lambda tensor: tensor[:1])
    forcing_of = lambda p: tuple(component[p * k:(p + 1) * k].unsqueeze(0) for component in case.forcing)
    spacing, density = case.scene.spacing.item(), case.scene.density.item()
    per_output, seconds, failure = [], [], None
    try:
        subject.begin(case)
    except Unsupported:
        return _not_applicable()
    except Exception as error:
        return {'values': {**dict.fromkeys(metrics.IDS), 'valid_outputs': 0},
                'failure': f'{type(error).__name__}: {error}'}
    try:
        for _ in range(TIMING_WARMUPS):
            subject.advance(start, forcing_of(0), case.scene, case.start)
        seconds = [_timed(subject, start, forcing_of(0), case.scene, case.start)[1] for _ in range(TIMING_SAMPLES)]
        state = start
        for p in range(passes):
            outputs = subject.advance(state, forcing_of(p), case.scene, case.start + p * k)
            if len(outputs) != k:
                raise RuntimeError(f'returned {len(outputs)} outputs for a pass of {k}')
            for offset, output in enumerate(outputs):
                index = p * k + offset + 1
                if not metrics.is_finite(output):
                    failure = f'A field was not finite at output {index}.'
                    break
                rhs = None if case.right_hand_sides is None else case.right_hand_sides[index]
                per_output.append(metrics.compare(output.map(lambda t: t[0]),
                                                  case.reference.map(lambda t: t[index]), spacing, density, rhs))
            if failure:
                break
            state = outputs[-1]
    except Exception as error:   # one subject failing on one case must not lose the other results
        failure = f'{type(error).__name__}: {error}'
    finally:
        subject.end()
    values = {**dict.fromkeys(metrics.IDS), **metrics.summarise(per_output, velocity_error_limit)}
    if seconds:
        per_output_seconds = statistics.median(seconds) / k
        values['time_per_output_ms'] = 1000.0 * per_output_seconds
        values['simulated_seconds_per_second'] = case.scene.timestep.item() / per_output_seconds
    return {'values': values, 'failure': failure}


def run(subjects, suite, environment=None):
    """Score every subject on every case and return the result document."""
    subjects = list(subjects)
    cases, rows = [], []
    for case in suite.cases:
        cases.append({'id': case.id, **case.attributes})
        for subject in subjects:
            rows.append({'subject': subject.id, 'case': case.id,
                         **run_case(subject, case, suite.velocity_error_limit)})
    return results.validate({
        'format': results.FORMAT, **results.git_identity(), 'environment': environment or describe_environment(),
        'suite': {'id': suite.id, 'label': suite.label, 'population': suite.population,
                  'description': suite.description, 'velocity_error_limit': suite.velocity_error_limit,
                  'timing_warmups': TIMING_WARMUPS, 'timing_samples': TIMING_SAMPLES},
        'metrics': metrics.DEFINITIONS, 'cases': cases,
        'subjects': [{'id': s.id, 'label': s.label, 'outputs_per_pass': s.outputs_per_pass, **s.attributes}
                     for s in subjects],
        'results': rows})


def describe_environment():
    environment = {'torch': torch.__version__, 'cpu_threads': torch.get_num_threads(),
                   'cuda': torch.cuda.is_available()}
    if environment['cuda']:
        environment['gpu'] = torch.cuda.get_device_name(0)
    return environment
