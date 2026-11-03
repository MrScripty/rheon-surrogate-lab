"""Freeze the requested experiment contrasts and fresh whole-scene populations."""
import random
from v5_support import ROOT,BASE,DOCS,read,write,sha

def reserve():
    if (DOCS/'campaign.json').exists():raise ValueError('Fresh reservation required')
    rng=random.Random(2026101908)
    rows=[]
    for population,shapes,steps in [
        ('development',[[32]*3,[35,27,19],[64]*3,[69,47,31],[128]*3],8),
        ('blind',[[32]*3,[39,25,17],[64]*3,[75,51,35],[128]*3],8),
        ('rollout-development',[[32]*3,[61,43,29],[64]*3],32),
        ('rollout-blind',[[32]*3,[67,45,31],[64]*3],32)]:
        for i,shape in enumerate(shapes):
            name=f'v5-{population}-{i:02d}'
            rows.append(dict(trajectory_id=name,population=population,snapshots=[0,steps-1],scene=dict(seed=rng.randrange(2**32),counts=shape,
                cells_per_metre=8 if i%2 else 16,steps=steps,warmup_steps=0,center=[rng.uniform(.15,.85)for _ in range(3)],
                acceleration=rng.uniform(.04,.22),radius=rng.uniform(.12,.44),swirl=rng.uniform(-.7,.7),
                fluctuation=rng.uniform(.1,.65),frequency=rng.uniform(.2,3.2),sgs=False)))
    spectrum=[]
    for population,shapes,bands in [
        ('train',[[32]*3,[33,25,17],[64]*3,[61,45,29]],[(1,3),(4,7),(8,12)]),
        ('development',[[35,27,19],[64]*3,[69,47,31]],[(1,3),(4,7),(8,12)]),
        ('blind',[[39,25,17],[64]*3,[75,51,35],[128]*3],[(1,4),(5,9),(10,15)])]:
        for i,shape in enumerate(shapes):
            for j,band in enumerate(bands):
                spectrum.append(dict(trajectory_id=f'v5-spectrum-{population}-{i:02d}-{j:02d}',population=population,counts_xyz=shape,
                    cells_per_metre=8 if i%2 else 16,mode_band=list(band),seed=rng.randrange(2**32),patterns=2,
                    mode_count=6,pressure_rms_pa=1.,definition='Six seeded separable Neumann cosine pressure modes; RHS from the unchanged finite-volume operator.'))
    write(DOCS/'scene-reservation.json',dict(status='reserved_before_capture_and_fitting',rows=rows,spectrum=spectrum,source_sha256=sha(__file__)))
    write(DOCS/'selection-rule.json',dict(status='frozen_before_new_evidence',student_priority=['most development requests passing both gates','retention of prior requests','fewest total full-grid corrections'],
        timing_selection='Select one trained model using development only. Freeze weights, hierarchy, reuse capacity and execution before fresh blind capture.',
        pressure_relative_residual=1e-6,pressure_absolute_residual=1e-12,predicted_divergence=1e-7,max_iterations=8,
        reuse_capacities=[0,2,4],promotion=False))
    tracks=[
        dict(id=1,name='Multilevel-conditioned lightweight model',requirements=['Three matched managed CUDA fits with fresh CPU restores','Tiny raw-residual and post-coarse controls with identical initial weights and native bank','Independent forward/objective qualification and deployed coarse-plus-fine parity','Full historical, new native and new spectral evaluation']),
        dict(id=2,name='Timestep correction reuse',requirements=['Bounded recycled direction space with true-residual checks and geometry reset','Independent tests covering repeated, changed, unrelated, zero and invalid equations','Same-input sequence screen and independently advanced native rollouts','Fresh versus recycled work and repeated full-fluid timing']),
        dict(id=3,name='Explicit spatial wavelengths and rectangular geometry',requirements=['Train/development/blind reservations before fitting','Controlled cosine mode bands with declared physical wavelengths and operator-verified labels','Native-only versus spectral-augmented training contrast retaining the original bank','Independent native and wavelength evaluations after frozen selection']),
        dict(id=4,name='Inference and solver execution',requirements=['Profile CNN, hierarchy, operators and native exchange on real requests','Implement and qualify measured optimization candidates against unchanged operator','Repeated same-core pressure and native-fluid timing including conversions, gates and reuse maintenance','Report accepted coverage separately from rejected attempt timing'])]
    write(DOCS/'campaign.json',dict(status='active',tracks=tracks,prior_publication_sha256=sha(ROOT/'docs/plans/pressure-methods-v4/publication.json'),
        arms=['v5-raw','v5-multilevel','v5-spectrum'],updates_per_fit=2400,small_updates=1600,larger_updates=800,
        shared_seed=2026101908,shared_sampling_seed=2026101804,learning_rate=.003,schedule=[dict(update=1801,learning_rate=.0003),dict(update=2251,learning_rate=.00003)],
        architecture='Three bias-free linear 3×3×3 replicated convolutions, 1→4→4→1, plus fixed 1/8 Jacobi skip; 648 trainable parameters.',
        controls=['v3-native-vcycle-neural','v4-gcr-gcr-learned','numerical multilevel with/without GCR','spectral'],
        application_drift_limits_deferred=True,promotion=False,source_sha256=sha(__file__)))
    print('Reserved four-track contrasts and independent native/spatial populations',flush=True)
if __name__=='__main__':reserve()
