"""Reserve the complete five-track scope and unseen scenes before any results."""
import random,shutil
from v6_support import ROOT,HERE,BASE,DOCS,read,write,sha
def run():
    if (DOCS/'campaign.json').exists():raise ValueError('Existing reservation must be preserved')
    BASE.mkdir(parents=True,exist_ok=True);DOCS.mkdir(parents=True,exist_ok=True)
    attachment=Path('/home/jeremy/.codex/attachments/8a430111-2578-4546-a4dc-2e76519a1ff9/pasted-text-1.txt')
    shutil.copyfile(attachment,DOCS/'requested-scope.txt')
    tracks=[
      dict(id=1,name='Cheaper local inference',required=['Boundary-equivalent fused native execution of the existing model','Train and compare a simpler learned stencil and separable filters','Independent representation tests and complete solver gates','Repeated total pressure and native-fluid timing']),
      dict(id=2,name='Selective neural correction',required=['Fixed selected-iteration policy','Measured stagnation-triggered policy','Coarser-hierarchy neural application','Neural calls, saved hierarchy work and full solve cost including decisions']),
      dict(id=3,name='Several-iteration training',required=['Matched tiny replay and multi-iteration managed fits','Same initial tensors, sampling sequence and optimizer schedule','Finite CPU/CUDA objective/backward checks and fresh CPU restores','Coverage and cost after frozen selection, with actual forward counts']),
      dict(id=4,name='Adaptive correction reuse',required=['Cheaper retention and immutable shared basis storage','Deterministic admission based on expected savings, calibrated only on development','Cold, repeated, abrupt and context-reset independent tests','Fixed-zero/fixed-four/adaptive native work, cost and drift']),
      dict(id=5,name='Longer and larger evaluation',required=['At least 256 consecutive requested timesteps on two independent scenes','Full-field pressure, staggered velocity and tracer drift against independent reference','Multiple new 256-cells-per-axis scenes','Independent grid-dimension and cells-per-metre comparisons','Use observed failures to decide whether rollout training or further architecture changes are justified'])]
    write(DOCS/'campaign.json',dict(status='active',tracks=tracks,prior_publication_sha256=sha(ROOT/'docs/plans/pressure-methods-v5/publication.json'),
        requested_scope_sha256=sha(DOCS/'requested-scope.txt'),application_drift_limits_deferred=True,promotion=False,
        fits=['v6-replay','v6-unrolled','v6-stencil','v6-separable'],updates_per_fit=2400,small_updates=1600,
        training_seed=2026101910,sampling_seed=2026101804,learning_rate=.001,schedule=[dict(update=1801,learning_rate=.0001),dict(update=2251,learning_rate=.00001)]))
    rng=random.Random(202610191019);rows=[]
    specifications=[('blind',[64]*3,16,8),('blind',[81,55,39],8,8),('blind',[128]*3,8,4),('blind',[128]*3,16,4),
      ('large-blind',[256]*3,16,4),('large-blind',[256]*3,32,4),('large-blind',[256,193,129],16,4),
      ('long-blind',[64]*3,16,256),('long-blind',[65,49,33],8,256)]
    for i,(population,counts,cpm,steps) in enumerate(specifications):
        # Paired grids/spacings share the same forcing realization and normalized
        # source shape; other scenes vary all forcing parameters independently.
        scene=dict(seed=rng.randrange(2**32),counts=counts,cells_per_metre=cpm,steps=steps,warmup_steps=0,
          center=[rng.uniform(.2,.8) for _ in range(3)],acceleration=rng.uniform(.07,.21),radius=rng.uniform(.18,.4),
          swirl=rng.uniform(-.6,.6),fluctuation=rng.uniform(.15,.6),frequency=rng.uniform(.3,3),sgs=False)
        if i in [3,4,5]:
            original=rows[2]['scene'];scene={**original,'counts':counts,'cells_per_metre':cpm,'steps':steps}
        rows.append(dict(trajectory_id=f'v6-{population}-{i:02d}',population=population,snapshots=[0,steps-1],scene=scene))
    write(DOCS/'scene-reservation.json',dict(status='reserved_before_fits_and_capture',rows=rows,source_sha256=sha(__file__),
        paired_indices=[2,3,4,5],pairing='128³ at 8/16 cells per metre and 256³ at 16/32; same normalized forcing; matched physical extents 16m/8m.'))
    write(DOCS/'selection-rule.json',dict(status='frozen_before_new_results',max_iterations=8,relative_residual=1e-6,absolute_residual=1e-12,predicted_divergence=1e-7,
        development='All v5 development and blind inputs are now consumed development; no new v6 reserved scenes enter fitting or selection.',
        rank=['Coverage under both unchanged gates','Retained historical coverage','Repeated complete-policy time','Worst same-request pressure error'],
        architecture_selection='Select at most two learned configurations before new blind capture; retain strongest numerical and spectral controls.',
        reuse_selection='Use only development measurements to calibrate expected cost and admit/reject a bounded reusable subspace.',
        long_rollout='Diagnostic only; preserve rejection and compare matched accepted prefixes without claiming an application horizon.'))
    write(BASE/'work-state.json',dict(status='active',phase='native inference and solver qualification',completed=[],remaining=[t['name'] for t in tracks]))
    print('Reserved all five tracks, four fits, paired 128³/256³ scenes, and two 256-step trajectories',flush=True)
if __name__=='__main__':
    from pathlib import Path
    run()
