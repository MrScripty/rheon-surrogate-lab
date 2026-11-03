#!/usr/bin/env bash
set -euo pipefail
cd -- '/media/jeremy/OrangeCream/Linux Software/repos/owned/graphics-3d/rheon-surrogate-lab'
python3 -B - <<'PY'
import json,time
from pathlib import Path
base=Path('runs/pressure-methods-v6')
names=['blind-capture-batch-01','blind-evaluation-controller-01-0','blind-evaluation-controller-01-1','long-drift-controller-01','long-fixed-controller-01']
deadline=time.monotonic()+7200
while True:
    pending=[]
    for name in names:
        path=base/name/'execution.json'
        if not path.exists():pending.append(name);continue
        item=json.loads(path.read_text())
        if item['status']=='stopped':raise RuntimeError('Required experiment stopped: '+name)
        if item['status']!='completed' or not item.get('owned_child_reaped') or item.get('active'):pending.append(name)
    if not pending:break
    if time.monotonic()>deadline:raise TimeoutError('Finite final timing prerequisite wait expired')
    time.sleep(5)
print('Every frozen evaluation and full-field drift controller is terminal and reaped.',flush=True)
PY
'runs/dependency-migration-v1/app-env/bin/python3.12' -B experiments/pressure-methods-v6/quiet_ready.py \
  --plan runs/pressure-methods-v6/final-native-timing-plan-01.json \
  --output runs/pressure-methods-v6/quiet-ready-final-native-01.json \
  --dependencies blind-capture-batch-01 blind-evaluation-batch-01-0 blind-evaluation-batch-01-1 long-rollout-batch-01 long-drift-batch-01 long-fixed-rollout-batch-01 long-fixed-drift-batch-01
exec python3 -B experiments/pressure-methods-v6/start_timing.py \
  --plan runs/pressure-methods-v6/final-native-timing-plan-01.json \
  --readiness runs/pressure-methods-v6/quiet-ready-final-native-01.json \
  --admission runs/pressure-methods-v6/timing-admission-final-native-01.json \
  --output runs/pressure-methods-v6/final-native-timing-batch-01 --timeout 14400
