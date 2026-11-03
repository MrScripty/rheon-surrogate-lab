"""Record all six requested mechanisms and evidence required before completion."""
import hashlib
from v4_support import ROOT, BASE, DOCS, read, write, sha

def reserve():
    source=ROOT/'docs/plans/pressure-methods-v3/publication.json'
    publication=read(source)
    for name, expected in publication['artifacts'].items():
        if sha(ROOT/name)!=expected: raise ValueError('Previous campaign publication changed: '+name)
    attachment=__import__('pathlib').Path('/home/jeremy/.codex/attachments/e15e5033-ee0d-49f7-90c1-0d9e974e17d5/pasted-text-1.txt')
    objective=attachment.read_bytes()
    tracks=[
        dict(id=1, name='Multilevel coarse-to-fine solver', hypothesis='A genuine size-adaptive Galerkin hierarchy reduces expensive fine-network corrections.',
             evidence=['Galerkin operator, nullspace and transfer qualification on odd/rectangular grids', 'Frozen multilevel and numerical-only controls on common requests', 'Coupled timing and accuracy of any selected candidate']),
        dict(id=2, name='Student distillation', hypothesis='Training smaller students on successful full-model corrections recovers better convergence than channel slicing plus fine-tuning.',
             evidence=['Intermediate and half-width students with an explicitly frozen completed teacher', 'Matched non-distillation controls and recorded optimizer/teacher work', 'Managed fits, fresh CPU restores and same-population accuracy/time']),
        dict(id=3, name='Model-generated state training', hypothesis='Training on states from model-driven native trajectories improves subsequent rollout behavior.',
             evidence=['Longer matched native/surrogate rollouts recording velocity/tracer/pressure drift', 'Separate training-only on-policy state captures with same-equation labels', 'Matched replay continuation and on-policy fit, then independent rollouts']),
        dict(id=4, name='Neural preconditioner', hypothesis='Residual minimization inside a conventional iterative solver uses neural proposals more efficiently than repeated raw corrections.',
             evidence=['Krylov or flexible residual-minimizing method with verified breakdown handling', 'Frozen neural and numerical preconditioner controls', 'Train for convergence inside the solver and compare with matched reconstruction objective']),
        dict(id=5, name='Data diversity and larger-grid diagnosis', hypothesis='More distinct forcing, rectangular shapes, frequencies and cold starts improve coverage more efficiently than grid size alone.',
             evidence=['New whole-scene train/development/blind reservations before capture', 'Early bounded 256-cubed diagnostic with explicit memory/time limits', 'Matched original and diversified training with per-tier work and independent evaluation']),
        dict(id=6, name='Inference execution', hypothesis='Changing inference execution and repeated operator work lowers total policy cost without retraining.',
             evidence=['Eager, traced/frozen and channels-last execution on identical real requests', 'Full forward/policy accuracy qualification and all conversions included', 'Repeated same-core inference and native IPC/fluid-step timing of selected execution'])]
    path=DOCS/'campaign.json'
    if path.exists(): raise ValueError('Fresh campaign reservation required')
    BASE.mkdir(parents=True,exist_ok=True)
    DOCS.mkdir(parents=True,exist_ok=True)
    write(DOCS/'requested-scope.json',dict(attachment_path=str(attachment),sha256=hashlib.sha256(objective).hexdigest(),text=objective.decode()))
    write(path,dict(status='active',tracks=tracks,prior_publication_sha256=sha(source),
        requested_scope_sha256=sha(DOCS/'requested-scope.json'),source_sha256=sha(__file__),
        controls=['v3-native','v2-num6','full-spectral','numerical-multilevel'],
        diagnostic_only=True,application_drift_limits_deferred=True,promotion=False,
        experiment_rules=['Preserve qualified historical sources and original checkpoints',
          'No use of consumed holdouts as blind evidence; reserve new whole scenes',
          'Count all training and teacher work, setup, conversions and solver work',
          'Pressure residual and predicted divergence gates remain unchanged',
          'No hidden solver fallback or advancement after rejection',
          'Do not declare the six-track campaign complete from a frozen screen alone']))
    print('Reserved six-track pressure-methods-v4 campaign')

if __name__=='__main__': reserve()
