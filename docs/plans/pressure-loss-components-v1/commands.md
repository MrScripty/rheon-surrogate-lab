# Commands and environment

All commands run from the repository root. The reusable configuration selects the
retained material and Pumas runtime manifest; it does not substitute new historical
hashes. Actual worker commands, source identities and available memory are retained
in reports/workers.json. Successful measurements are immutable and require fresh
output/vector directories; the first two failed preflights remain under runs.

Environment: Pumas-managed CPython 3.14.7, PyTorch 2.14.0+cu132, NumPy 2.5.3 on
Linux. CUDA_VISIBLE_DEVICES is empty for numerical workers; one Torch/BLAS thread,
CPU affinity 21/22/23. No CUDA initialization or optimizer step. All checkpoints
use the captured float32 network and float64 physical calculation. No Tuldok
training admission, authentication or private Run-storage access is needed.
The existing Tuldok activity file announces reporting/completion; no new catalog.

Final measurements and evidence projection:

```bash
python3 -B experiments/pressure-loss-components-v1/diagnose.py \
  --config experiments/pressure-loss-components-v1/config.json \
  --output runs/pressure-loss-components-v1/workers-03 \
  --results docs/plans/pressure-loss-components-v1/reports
python3 -B experiments/pressure-loss-components-v1/report.py \
  --directory docs/plans/pressure-loss-components-v1 \
  --execution runs/pressure-loss-components-v1/workers-03/execution.json
```

Focused numerical oracle and real negative fixture, using the same Pumas runtime:

```python
import subprocess, sys
sys.path.insert(0, 'experiments/pressure-four-pass-distillation-v1')
from common import read, environment
runtime = read('runs/dependency-migration-v1/runtime/current-manifest-v2.json')
commands = [
    ['check.py', '--output', 'docs/plans/pressure-loss-components-v1/reports/math-check.json'],
    ['check-joint-guard.py', '--config', 'experiments/pressure-loss-components-v1/config.json',
     '--output', 'docs/plans/pressure-loss-components-v1/reports/joint-guard.json',
     '--scratch', 'runs/pressure-loss-components-v1/joint-guard-02'],
]
for script, *arguments in commands:
    subprocess.run([runtime['python'], '-B', 'experiments/pressure-loss-components-v1/' + script,
                    *arguments], env=environment(runtime), check=True)
```

Cancellation fixture and final report checks:

```bash
python3 -B experiments/pressure-loss-components-v1/check-cancel.py \
  --config experiments/pressure-loss-components-v1/config.json \
  --scratch runs/pressure-loss-components-v1/cancel-02 \
  --output docs/plans/pressure-loss-components-v1/reports/cancellation.json
python3 -B experiments/pressure-loss-components-v1/verify.py \
  --directory docs/plans/pressure-loss-components-v1 \
  --output docs/plans/pressure-loss-components-v1/reports/verification.json
python3 -B experiments/pressure-low-mode-tail-v1/view.py \
  --directory docs/plans/pressure-loss-components-v1
node experiments/pressure-loss-components-v1/browser.cjs \
  /home/jeremy/.cache/ms-playwright/chromium_headless_shell-1246/chrome-headless-shell-linux64/chrome-headless-shell \
  runs/pressure-loss-components-v1/browser-01
```

The lifecycle test substitutes three sleepers for numerical children while using
the real supervisor. It sends SIGTERM only to its owned fixture parent and checks
that all three process groups terminate and are reaped. It does not test numerical
results or the forced-KILL branch. Actual preflight failures additionally exercise
the failed-worker result path. The headless browser owns and removes its temporary
profile, then terminates/reaps itself; it does not touch the user's browser.

The initial additivity failure was reproduced by an isolated C66 invocation of
measure.py with the same configuration and CPU 22. reports/additivity-probe.json
records repeated/full versus reconstructed joint equality on case 0. Final
measurements extend that exact joint equality check to all 24 cases and retain
the original separate-gradient precision flags. No threshold is presented as
universally passed when its recorded case checks failed.
