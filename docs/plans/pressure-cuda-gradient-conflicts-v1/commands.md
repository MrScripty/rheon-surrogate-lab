# Reproduction and environment

From the repository root, using fresh output directories for any repeat:

```bash
python3 -B experiments/pressure-cuda-gradient-conflicts-v1/run.py \
  --config experiments/pressure-cuda-gradient-conflicts-v1/config.json \
  --output runs/pressure-cuda-gradient-conflicts-v1/attempt-02
python3 -B experiments/pressure-cuda-gradient-conflicts-v1/report.py \
  --measurement runs/pressure-cuda-gradient-conflicts-v1/attempt-02/measurement.json \
  --execution runs/pressure-cuda-gradient-conflicts-v1/attempt-02/execution.json \
  --directory docs/plans/pressure-cuda-gradient-conflicts-v1
python3 -B experiments/pressure-cuda-gradient-conflicts-v1/lifecycle.py \
  --output runs/pressure-cuda-gradient-conflicts-v1/cancellation-01
node experiments/pressure-cuda-gradient-conflicts-v1/browser.cjs \
  /home/jeremy/.cache/ms-playwright/chromium_headless_shell-1246/chrome-headless-shell-linux64/chrome-headless-shell \
  runs/pressure-cuda-gradient-conflicts-v1/browser-01
```

The CUDA runner and cancellation fixture need access to the NVIDIA device;
Chromium also runs outside the filesystem sandbox. The parent owns and reaps
qualifier/CUDA workers with host RSS/wall/reserve guards; the child sets an
allocator fraction for its declared 4GiB GPU limit. Qualification uses literal
CPU geometry before any CUDA numerical work.

Runtime comes from the existing Pumas manifest named in config: CPython 3.14.7,
Torch 2.14.0+cu132, NumPy 2.5.3, NVIDIA RTX 5090 Laptop GPU. Environment owner
`experiments/pressure-four-pass-distillation-v1/common.py` selects installed
site-packages, fixes OMP/OPENBLAS/MKL/RAYON threads to 1, and sets explicit CPU
or CUDA device; CPU verification hides CUDA. The single GPU diagnostic pins
one host CPU and streams independent graphs, rather than running training.
Original source sets deterministic algorithms, no TF32 and no AMP.

Copy qualification/execution, cancellation and browser receipts to reports/;
the report embeds the scalar measurement. The initial failed config and
execution are retained as reports/attempt-01-config.json and
reports/attempt-01-execution.json. Full commands, PIDs, exit/reaping evidence
and config/runtime identities are in those receipts. No private Tuldok Run
directories are accessed: published replay downloads and their custody
receipts supply all training material. The existing linked project activity
shows diagnosis/completion; this does not create a training Run, catalog or
new Project.

CPU verification runs with the selected Pumas Python and environment:

```python
import subprocess, sys
from pathlib import Path
sys.path.insert(0, 'experiments/pressure-four-pass-distillation-v1')
from common import environment, read
runtime = read(Path('runs/dependency-migration-v1/runtime/current-manifest-v2.json'))
subprocess.run([runtime['python'], '-B',
    'experiments/pressure-cuda-gradient-conflicts-v1/verify.py',
    '--directory', 'docs/plans/pressure-cuda-gradient-conflicts-v1',
    '--output', 'docs/plans/pressure-cuda-gradient-conflicts-v1/reports/verification.json'],
    env=environment(runtime), check=True, timeout=60)
```

Coding standards use the previously resolved MCP snapshot
`snapshot:v1:ffcfc3f8-b52a-434e-b57f-d259a0355711`; independent-oracle guidance
was reread for this diagnosis. Exact historical gradient replay establishes
consistency with the actual training source/selected states. It does not
certify external physics or a replacement training policy. Stage only the new
study and discovery links; preserve all unrelated concurrent edits.
