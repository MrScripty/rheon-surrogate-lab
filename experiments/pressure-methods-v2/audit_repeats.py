"""Check the actual paired sampling repeats against admitted orders and optimizer work."""
import hashlib,json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];RUNS=ROOT/'runs/pressure-methods-v2';DOCS=ROOT/'docs/plans/pressure-methods-v2'
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha
import shape_order
manifest=read(RUNS/'shape-training-view-01/manifest.json');parents=[dict(p,entry=dict(split=p['split'])) for p in manifest['parents']]
rows=[];paired={}
for seed in [25,26]:
 for family in ['v2-headshape','v2-num6']:
  arm=family+f'-s{seed}';project=RUNS/(arm+'-project-01');config=read(project/'config.json')
  sequence=shape_order.order(parents,config);expected=shape_order.STATS.copy();digest=hashlib.sha256(sequence.tobytes()).hexdigest()
  history=read(RUNS/(arm+'-study-01/restore/history.json'));qualification=read(DOCS/(arm+'-qualification-01.json'))
  if history['current_feedback']['sampling']!=expected:raise ValueError('Recorded repeat sampling differs from replay')
  if qualification['matched_control']['sample_order_sha256']!=digest:raise ValueError('Repeat order differs from admission')
  work=history['optimizer_work']
  if work['network_forwards']!=4800 or work['forward_cells']!=2*expected['sampled_cells']:raise ValueError('Actual repeat neural work differs from sampled cells')
  if seed in paired and paired[seed]!=(digest,work):raise ValueError('Paired families changed sample order or fine-network work')
  paired[seed]=(digest,work)
  rows.append(dict(arm=arm,seed=2026101800+seed,order_sha256=digest,optimizer_work=work,sampling=expected,history_sha256=sha(RUNS/(arm+'-study-01/restore/history.json'))))
if len({row['order_sha256'] for row in rows})!=2:raise ValueError('Two distinct paired sampling orders required')
write(DOCS/'repeat-sampling-audit.json',dict(status='passed',rows=rows,same_sampling_and_fine_network_work_within_each_seed=True,distinct_orders=2,source_sha256=sha(__file__)))
print('Audited four actual sampling-repeat logs; paired families match work and each seed changes order')
